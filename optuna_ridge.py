"""Automated hyperparameter search for Ridge-regression time-series forecasting.

Searches over scaling strategies, lookback windows, regularization, and data
augmentation with Optuna, and compares the resulting per-series/horizon "local"
models against a global baseline. See README.md for usage and CLI arguments.
"""

import argparse
import os
import math
import time

import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import optuna

# ==========================================
# 0. Environment Setup
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. Scaling Strategies & Scalers
# ==========================================
class ScalerStrategy:
    """Base class for computing center and scale statistics separately."""

    def compute_center(self, tensor, dim):
        raise NotImplementedError

    def compute_scale(self, tensor, dim):
        raise NotImplementedError


class StandardStrategy(ScalerStrategy):
    """Mean for center, std for scale."""

    def compute_center(self, tensor, dim):
        return torch.mean(tensor, dim=dim, keepdim=True)

    def compute_scale(self, tensor, dim):
        # Biased variance (correction=0) to match numpy/reference
        var = torch.var(tensor, dim=dim, keepdim=True, correction=0)
        return torch.sqrt(var + 1e-5)


class RobustStrategy(ScalerStrategy):
    """Median for center, IQR for scale."""

    def __init__(self, q_min=0.25, q_max=0.75):
        self.q_min = q_min
        self.q_max = q_max

    def compute_center(self, tensor, dim):
        return torch.quantile(tensor, 0.50, dim=dim, keepdim=True)

    def compute_scale(self, tensor, dim):
        q_l = torch.quantile(tensor, self.q_min, dim=dim, keepdim=True)
        q_h = torch.quantile(tensor, self.q_max, dim=dim, keepdim=True)
        return (q_h - q_l).clamp_min(1e-5)


class IdentityStrategy(ScalerStrategy):
    """No normalization."""

    def compute_center(self, tensor, dim):
        return torch.zeros(1, device=tensor.device)

    def compute_scale(self, tensor, dim):
        return torch.ones(1, device=tensor.device)


class GlobalScaler:
    """
    Global normalization: (X - center) / scale.

    Both center and scale are computed on the original (non-centered) data.
    Computes global (scalar) statistics over all data for dataset-level preprocessing.
    """

    def __init__(self, strategy: ScalerStrategy):
        self.strategy = strategy
        self.center = None
        self.scale = None

    def fit(self, X: torch.Tensor):
        """Compute global center and scale statistics.

        X: (S, T) series × time
        Computes scalar statistics over all values.
        """
        # Flatten to compute global statistics
        X_flat = X.reshape(-1, 1)  # (S*T, 1)
        self.center = self.strategy.compute_center(X_flat, dim=0)  # (1, 1) scalar
        self.scale = self.strategy.compute_scale(X_flat, dim=0)  # (1, 1) scalar

    def transform(self, X: torch.Tensor):
        """Apply (X - center) / scale."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        if isinstance(self.strategy, IdentityStrategy):
            return X
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        return (X - center) / scale

    def inv_transform(self, X: torch.Tensor):
        """Reverse: X * scale + center."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        return X * scale + center


class LocalNormScaler:
    """
    Local normalization (instance-norm style).

    Center is computed on original data, scale is computed on centered data.
    Transform subtracts center and appends scale as a feature.

    Args:
        strategy: ScalerStrategy (Standard, Robust, or Identity)
        lookback: context length
        last_k: trailing samples to compute stats from (default: lookback)
    """

    def __init__(self, strategy: ScalerStrategy, lookback: int, last_k: int = None):
        self.strategy = strategy
        self.lookback = lookback
        self.last_k = last_k or lookback
        self.center = None
        self.scale = None

    def fit(self, X: torch.Tensor):
        """Compute center on original data, scale on centered data."""
        wins = X[:, -self.last_k:] if X.dim() == 2 and X.shape[-1] >= self.last_k else X
        self.center = self.strategy.compute_center(wins, dim=-1)
        # Compute scale on centered data
        centered = wins - self.center
        # self.scale = self.strategy.compute_scale(centered, dim=-1)
        # the scale is std deviation regardless of strategy for local norm
        var = torch.var(centered, dim=-1, keepdim=True, correction=0)
        self.scale = torch.sqrt(var + 1e-5)

    def transform(self, X: torch.Tensor):
        """Subtract center, append scale as feature. Returns (N, L+1)."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        X_centered = X - center
        return torch.cat([X_centered, scale], dim=-1)

    def transform_target(self, Y: torch.Tensor):
        """Subtract center from target (no scale appending)."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(Y.device, non_blocking=True)
        return Y - center

    def inv_transform(self, pred: torch.Tensor):
        """Add center back to predictions."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(pred.device, non_blocking=True)
        return pred + center

    @property
    def output_dim(self):
        """Feature dimension after transform (lookback + 1)."""
        return self.lookback + 1


class Augmentor:
    def apply(self, X): return X

class TimeDomainNoise(Augmentor):
    def __init__(self, sigma): self.sigma = sigma
    def apply(self, X):
        if self.sigma <= 0: return X
        return X.add_(torch.randn_like(X), alpha=self.sigma)

class FreqDomainNoise(Augmentor):
    def __init__(self, sigma, mode='amplitude'):
        """
        mode: 'amplitude' (scale magnitude) or 'phase' (shift phase)
        """
        self.sigma = sigma
        self.mode = mode

    def apply(self, X):
        # X shape: (Batch, Time, Feat) or (Batch, Time)
        # 1. FFT
        # rfft computes the real-input FFT (faster, gives only positive freqs)
        X_freq = torch.fft.rfft(X, dim=1) 
        
        # 2. Perturb
        if self.mode == 'amplitude':
            # Perturb magnitude: Multiply by random scale ~ N(1, sigma)
            # Create noise for (Batch, Freq_Bins, Feat)
            noise = torch.randn_like(X_freq.abs()) * self.sigma
            # Apply to amplitude, keep phase
            # New_Complex = (Old_Abs + Noise) * e^(i * Old_Phase)
            # Efficient way: scale real and imaginary parts uniformly? 
            # Better: Scale magnitude directly
            scale = 1.0 + noise
            X_freq = X_freq * scale
            
        elif self.mode == 'phase':
            # Perturb phase: Add random angle ~ N(0, sigma)
            phase_noise = torch.randn_like(X_freq.angle()) * self.sigma
            # New_Complex = Old_Complex * e^(i * phase_noise)
            # Euler's formula: e^(ix) = cos(x) + i*sin(x)
            rotation = torch.polar(torch.ones_like(X_freq.abs()), phase_noise)
            X_freq = X_freq * rotation

        # 3. Inverse FFT
        X_aug = torch.fft.irfft(X_freq, n=X.shape[1], dim=1)
        return X_aug


def apply_augmentation(X, config=None):
    if config is None: return X
    X_aug = X.clone()
    if config['noise_type'] == 'time':
        X_aug = TimeDomainNoise(config['sigma']).apply(X_aug)
    elif config['noise_type'] == 'freq':
        X_aug = FreqDomainNoise(config['sigma']).apply(X_aug)
    return X_aug

# ==========================================
# 2. Solver
# ==========================================
class RidgeSolver:
    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def solve(self, X_train, Y_train, alphas, scaler=None, chunk_size=50000, aug_config=None, dtype=torch.float64):
        """
        Solve ridge regression for multiple alpha values.

        Args:
            X_train: (N, L) input features
            Y_train: (N, H) targets
            alphas: (K,) regularization strengths
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
                    - LocalNormScaler: per-sample normalization, appends scale feature, no intercept
                    - GlobalScaler: dataset-level normalization, adds intercept
                    - None: no normalization, adds intercept
            chunk_size: batch size for memory-efficient computation
            aug_config: augmentation configuration
            dtype: computation dtype

        Returns:
            Theta: (K, F, H) weight matrices for each alpha
        """
        N, L = X_train.shape
        _, H = Y_train.shape
        K = alphas.shape[0]

        # Determine feature dimension and intercept based on scaler type
        use_local_norm = isinstance(scaler, LocalNormScaler)
        if use_local_norm:
            F = L + 1  # context features + scale feature (no intercept)
            fit_intercept = False
        else:
            F = L + 1  # features + intercept
            fit_intercept = True

        XTX = torch.zeros(F, F, device=self.device, dtype=dtype)
        XTY = torch.zeros(F, H, device=self.device, dtype=dtype)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_chunk = X_train[start:end].to(dtype=dtype, device=self.device, non_blocking=True)
            Y_chunk = Y_train[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

            if use_local_norm:
                # LocalNormScaler: fit per-chunk, transform X (appends scale), transform target Y
                scaler.fit(X_chunk)
                X_chunk = scaler.transform(X_chunk)
                Y_chunk = scaler.transform_target(Y_chunk)
                # Data augmentation (on normalized features, excluding appended scale)
                X_chunk_features = X_chunk[:, :-1]
                X_chunk_features = apply_augmentation(X_chunk_features, aug_config)
                X_chunk = torch.cat([X_chunk_features, X_chunk[:, -1:]], dim=-1)
            else:
                # GlobalScaler or None: transform if scaler provided
                if scaler is not None:
                    X_chunk = scaler.transform(X_chunk)
                    Y_chunk = scaler.transform(Y_chunk)
                # Data augmentation
                X_chunk = apply_augmentation(X_chunk, aug_config)
                # Add intercept term
                X_chunk = torch.cat([torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device), X_chunk], dim=1)

            # Accumulate the gram matrix
            XTX.add_(X_chunk.T @ X_chunk)
            XTY.add_(X_chunk.T @ Y_chunk)

        XTX_expanded = XTX.unsqueeze(0).expand(K, -1, -1)

        # Create regularization matrix
        if fit_intercept:
            # Don't regularize the intercept term
            reg_diags = alphas.unsqueeze(1).expand(-1, F).clone()
            reg_diags[:, 0] = 0.0
        else:
            # Regularize all features (LocalNormScaler case)
            reg_diags = alphas.unsqueeze(1).expand(-1, F)
        I_reg = torch.diag_embed(reg_diags)

        A = XTX_expanded + I_reg
        B = XTY.unsqueeze(0).expand(K, -1, -1)

        # Solve Linear System
        try:
            Lchol = torch.linalg.cholesky(A)
            Theta = torch.cholesky_solve(B, Lchol)  # (K, F, H)
        except RuntimeError:
            try:
                Theta = torch.linalg.solve(A, B)
            except RuntimeError:
                XTX_inv = torch.linalg.pinv(A)
                Theta = torch.einsum('kij,kjh->kih', XTX_inv, B)

        torch.cuda.empty_cache()
        return Theta

    @torch.no_grad()
    def predict(self, X, theta, scaler=None, chunk_size=50000, dtype=torch.float64):
        """
        Make predictions using trained weights.

        Args:
            X: (N, L) input features (raw, before any normalization)
            theta: (K, F, H) weight matrices
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
                    - LocalNormScaler: fit, transform (appends scale), predict, inv_transform
                    - GlobalScaler: transform only, predict (caller handles inv_transform)
                    - None: add intercept, predict
            chunk_size: batch size for memory-efficient prediction
            dtype: computation dtype

        Returns:
            Y_pred: (K, N, H) predictions
        """
        N = X.shape[0]
        K, _, H = theta.shape
        theta = theta.to(dtype=dtype, device=self.device, non_blocking=True)

        use_local_norm = isinstance(scaler, LocalNormScaler)

        # Process in chunks for memory efficiency
        Y_pred_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_chunk = X[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

            if use_local_norm:
                # LocalNormScaler: fit, transform, predict, then inv_transform
                scaler.fit(X_chunk)
                X_transformed = scaler.transform(X_chunk)
                Y_pred_norm = torch.einsum('nf, kfh -> knh', X_transformed, theta)
                Y_pred_chunk = scaler.inv_transform(Y_pred_norm)
            else:
                # GlobalScaler or None: add intercept and predict
                if scaler is not None:
                    X_chunk = scaler.transform(X_chunk)
                ones = torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device)
                X_chunk = torch.cat([ones, X_chunk], dim=1)
                Y_pred_chunk = torch.einsum('nf, kfh -> knh', X_chunk, theta)

            Y_pred_chunks.append(Y_pred_chunk.cpu())

        Y_pred = torch.cat(Y_pred_chunks, dim=1).to(self.device)
        return Y_pred

    @torch.no_grad()
    def solve_batched(self, X_train_batch, Y_train_batch, alphas, scaler=None, chunk_size=50000, aug_config=None, dtype=torch.float64):
        """
        Solve ridge regression for multiple series simultaneously (batched).

        Each series gets its own separate Ridge model with unique weights.
        Adaptively batches over series and alphas to fit GPU memory.

        Args:
            X_train_batch: (S, N_s, L) input features for S series
            Y_train_batch: (S, N_s, H) targets for S series
            alphas: (K,) regularization strengths
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
            chunk_size: batch size for memory-efficient computation per series
            aug_config: augmentation configuration
            dtype: computation dtype

        Returns:
            Theta: (K, S, F, H) weight matrices on CPU
        """
        S, N_s, L = X_train_batch.shape
        _, _, H = Y_train_batch.shape
        K = alphas.shape[0]

        # Determine feature dimension and intercept based on scaler type
        use_local_norm = isinstance(scaler, LocalNormScaler)
        if use_local_norm:
            F = L + 1  # context features + scale feature (no intercept)
            fit_intercept = False
        else:
            F = L + 1  # features + intercept
            fit_intercept = True

        # Theta on CPU — results are read by predict_batched which also works per-series
        Theta = torch.zeros(K, S, F, H, dtype=dtype)

        # Determine series_batch: must fit Gram matrices + at least 1 alpha solve
        # Gram: Sb * (F*F + F*H) * 8 bytes for XTX + XTY
        # Solve (min): 1 * Sb * F*F * 8 * 3 bytes for A, L, workspace
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(self.device)
        gpu_budget = int(free_mem * 0.7)
        bytes_per_series = (F * F + F * H + F * F * 3) * 8  # Gram + 1-alpha solve
        series_batch = max(1, min(S, gpu_budget // bytes_per_series))

        for s_start in range(0, S, series_batch):
            s_end = min(s_start + series_batch, S)
            Sb = s_end - s_start

            # --- Phase 1: Accumulate Gram matrices for this series batch ---
            XTX = torch.zeros(Sb, F, F, device=self.device, dtype=dtype)
            XTY = torch.zeros(Sb, F, H, device=self.device, dtype=dtype)

            for si, s in enumerate(range(s_start, s_end)):
                X_s = X_train_batch[s]  # (N_s, L)
                Y_s = Y_train_batch[s]  # (N_s, H)

                for start in range(0, N_s, chunk_size):
                    end = min(start + chunk_size, N_s)
                    X_chunk = X_s[start:end].to(dtype=dtype, device=self.device, non_blocking=True)
                    Y_chunk = Y_s[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

                    if use_local_norm:
                        scaler.fit(X_chunk)
                        X_chunk = scaler.transform(X_chunk)
                        Y_chunk = scaler.transform_target(Y_chunk)
                        X_chunk_features = X_chunk[:, :-1]
                        X_chunk_features = apply_augmentation(X_chunk_features, aug_config)
                        X_chunk = torch.cat([X_chunk_features, X_chunk[:, -1:]], dim=-1)
                    else:
                        if scaler is not None:
                            X_chunk = scaler.transform(X_chunk)
                            Y_chunk = scaler.transform(Y_chunk)
                        X_chunk = apply_augmentation(X_chunk, aug_config)
                        X_chunk = torch.cat([torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device), X_chunk], dim=1)

                    XTX[si].add_(X_chunk.T @ X_chunk)
                    XTY[si].add_(X_chunk.T @ Y_chunk)

            # --- Phase 2: Solve for all alphas, adaptively batched ---
            # Re-query free memory after Gram allocation
            torch.cuda.empty_cache()
            free_mem2, _ = torch.cuda.mem_get_info(self.device)
            solve_budget = int(free_mem2 * 0.7)
            # Each alpha in this batch needs Sb * F * F * 8 * 3 bytes
            solve_bytes_per_alpha = Sb * F * F * 8 * 3
            alpha_batch = max(1, min(K, solve_budget // max(1, solve_bytes_per_alpha)))

            for k_start in range(0, K, alpha_batch):
                k_end = min(k_start + alpha_batch, K)
                Kb = k_end - k_start
                alpha_chunk = alphas[k_start:k_end]

                diag_vals = alpha_chunk.view(Kb, 1, 1).expand(Kb, Sb, F).clone()
                if fit_intercept:
                    diag_vals[:, :, 0] = 0.0
                I_reg = torch.diag_embed(diag_vals)

                A = XTX.unsqueeze(0).expand(Kb, -1, -1, -1) + I_reg  # (Kb, Sb, F, F)
                B = XTY.unsqueeze(0).expand(Kb, -1, -1, -1).clone()  # (Kb, Sb, F, H)

                try:
                    L = torch.linalg.cholesky(A)
                    Theta[k_start:k_end, s_start:s_end] = torch.cholesky_solve(B, L).cpu()
                except RuntimeError:
                    try:
                        Theta[k_start:k_end, s_start:s_end] = torch.linalg.solve(A, B).cpu()
                    except RuntimeError:
                        A_inv = torch.linalg.pinv(A)
                        Theta[k_start:k_end, s_start:s_end] = torch.einsum('ksij,ksjh->ksih', A_inv, B).cpu()

                del A, B, I_reg, diag_vals

            del XTX, XTY
            torch.cuda.empty_cache()

        return Theta

    @torch.no_grad()
    def predict_batched(self, X_batch, theta, scaler=None, chunk_size=50000, dtype=torch.float64):
        """
        Make predictions using batched trained weights (one model per series).

        Args:
            X_batch: (S, N_s, L) input features for S series
            theta: (K, S, F, H) weight matrices
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
            chunk_size: batch size for memory-efficient prediction per series
            dtype: computation dtype

        Returns:
            Y_pred: (K, S, N_s, H) predictions
        """
        S, N_s, L = X_batch.shape
        K, _, _, H = theta.shape
        # theta stays on CPU; load per-series to GPU to avoid large GPU allocation

        use_local_norm = isinstance(scaler, LocalNormScaler)

        Y_pred_all = []

        # Process each series
        for s in range(S):
            X_s = X_batch[s]  # (N_s, L)
            theta_s = theta[:, s, :, :].to(dtype=dtype, device=self.device, non_blocking=True)  # (K, F, H)

            # Process in chunks for memory efficiency
            Y_pred_chunks = []
            for start in range(0, N_s, chunk_size):
                end = min(start + chunk_size, N_s)
                X_chunk = X_s[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

                if use_local_norm:
                    # LocalNormScaler: fit, transform, predict, then inv_transform
                    scaler.fit(X_chunk)
                    X_transformed = scaler.transform(X_chunk)
                    Y_pred_norm = torch.einsum('nf, kfh -> knh', X_transformed, theta_s)
                    Y_pred_chunk = scaler.inv_transform(Y_pred_norm)
                else:
                    # GlobalScaler or None: add intercept and predict
                    if scaler is not None:
                        X_chunk = scaler.transform(X_chunk)
                    ones = torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device)
                    X_chunk = torch.cat([ones, X_chunk], dim=1)
                    Y_pred_chunk = torch.einsum('nf, kfh -> knh', X_chunk, theta_s)

                Y_pred_chunks.append(Y_pred_chunk.cpu())

            Y_pred_s = torch.cat(Y_pred_chunks, dim=1)  # (K, N_s, H)
            Y_pred_all.append(Y_pred_s)

        Y_pred = torch.stack(Y_pred_all, dim=1)  # (K, S, N_s, H) — stays on CPU
        return Y_pred

# ==========================================
# 3. Data Pipeline
# ==========================================
def get_context_and_horizons(data, lookback, horizons):
    if isinstance(horizons, int):
        H_max = horizons
        idx_tensor = torch.tensor([horizons], dtype=torch.long)
    else:
        H_max = max(horizons)
        idx_tensor = torch.tensor(horizons, dtype=torch.long)

    # Dynamic Window: lookback + max_horizon required
    window_size = lookback + H_max
    if data.shape[1] < window_size:
        return None, None

    data_windowed = data.unfold(1, window_size, 1)
    X = data_windowed[:, :, :lookback]
    target_indices = lookback + idx_tensor - 1
    Y = torch.index_select(data_windowed, dim=2, index=target_indices)
    return X, Y


def create_strategy(method: str, scaler_config: dict) -> ScalerStrategy:
    """Create a ScalerStrategy based on method name."""
    if method == "mean":
        return StandardStrategy()
    elif method == "robust":
        q_min = scaler_config.get('q_min', 0.25)
        q_max = scaler_config.get('q_max', 0.75)
        return RobustStrategy(q_min, q_max)
    elif method == "none":
        return IdentityStrategy()
    else:
        raise ValueError(f"Unknown scaler method: {method}")


def create_scaler(strategy: ScalerStrategy, scope: str, lookback: int,
                  scaler_config: dict):
    """Create GlobalScaler or LocalNormScaler based on scope."""
    if scope == 'global':
        return GlobalScaler(strategy)
    elif scope == 'local':
        ratio = scaler_config.get('local_ratio', 1.0)
        last_k = max(1, int(lookback * ratio))
        return LocalNormScaler(strategy, lookback, last_k)
    else:
        raise ValueError(f"Unknown scope: {scope}")


def maybe_transform(X: torch.Tensor, scaler, force_no_fit: bool = False):
    """
    Apply scaler transformation, fitting if needed for LocalNormScaler.

    For GlobalScaler: assumes already fitted, just transforms.
    For LocalNormScaler: fits on X then transforms (unless force_no_fit).
    """
    if X is None:
        return None
    if isinstance(scaler, LocalNormScaler):
        if not force_no_fit:
            scaler.fit(X)
        return scaler.transform(X)
    elif isinstance(scaler, GlobalScaler):
        return scaler.transform(X)
    else:
        return X


def get_prepared_data(series_data, lookback, horizons, split_idx_1, split_idx_2,
                      scaler_config):
    """
    Prepare train/val/test data with appropriate scalers.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers_dict
    """
    if not isinstance(horizons, (list, tuple)):
        horizons = [horizons]

    method = scaler_config['method']
    scope = scaler_config['scope']

    strategy = create_strategy(method, scaler_config)

    # Slice boundaries
    test_slice_start = max(0, split_idx_2 - lookback)

    # Get windows
    X_train, Y_train = get_context_and_horizons(
        series_data[:, :split_idx_1], lookback, horizons)

    if split_idx_2 > split_idx_1:
        X_val, Y_val = get_context_and_horizons(
            series_data[:, split_idx_1 - lookback:split_idx_2],
            lookback, horizons)
    else:
        X_val, Y_val = None, None

    X_test, Y_test = get_context_and_horizons(
        series_data[:, test_slice_start:], lookback, horizons)

    # Create scalers
    if scope == 'global':
        scaler_train = GlobalScaler(strategy)
        scaler_train.fit(series_data[:, :split_idx_1])
        scaler_val = scaler_test = scaler_train
    elif scope == 'local':
        ratio = scaler_config.get('local_ratio', 1.0)
        last_k = max(1, int(lookback * ratio))
        scaler_train = LocalNormScaler(strategy, lookback, last_k)
        scaler_val = LocalNormScaler(strategy, lookback, last_k)
        scaler_test = LocalNormScaler(strategy, lookback, last_k)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    scalers = {'train': scaler_train, 'val': scaler_val, 'test': scaler_test}
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers

# ==========================================
# 5. Objective Wrappers
# ==========================================
class SingleObjectiveWrapper:
    def __init__(self, data, series_idx, lookbacks, horizon, alphas, device,
                 train_ratio=0.4, test_ratio=0.2, split_starts=None, split_ends=None,
                 n_folds=1, fold_reg_lambda=0.0,
                 scaler_scope="search", scaler_method="search",
                 fixed_local_ratio=None, fixed_noise_type=None, fixed_aug_sigma=None,
                 pool_series=False):
        """
        Args:
            data: (T, S) time series data
            series_idx: index of target series, or None for all series
            lookbacks: list of valid lookback values for Optuna to search over
            horizon: forecast horizon(s)
            alphas: regularization strengths
            device: torch device
            train_ratio: fraction of data for training during search
            test_ratio: fraction of data for test (not used during search)
            split_starts: list of split start indices
            split_ends: list of split end indices
            n_folds: number of folds for expanding window CV (1 = single split)
            fold_reg_lambda: regularization weight for fold variance (0 = no regularization)
            scaler_scope: "global", "local", or "search" (default: "search")
            scaler_method: "mean", "robust", or "search" (default: "search")
            fixed_local_ratio: if set, use this value instead of searching
            fixed_noise_type: if set, use this augmentation type instead of searching
            fixed_aug_sigma: if set, use this augmentation sigma instead of searching
            pool_series: if True, pool training data across series (single model per group)
        """
        if series_idx is None:
            self.data = data.transpose(0, 1)
            self.is_batched = True
            self.n_series = data.shape[1]
        elif isinstance(series_idx, (list, tuple)):
            self.data = data[:, series_idx].transpose(0, 1)
            self.is_batched = len(series_idx) > 1
            self.n_series = len(series_idx)
        else:
            self.data = data[:, series_idx].unsqueeze(0)
            self.is_batched = False
            self.n_series = 1
        # Store lookback bounds for log-scale search
        self.min_lookback = min(lookbacks)
        self.max_lookback = max(lookbacks)
        self.horizon = horizon
        self.alphas = alphas
        self.device = device
        self.solver = RidgeSolver(device)
        self.n_folds = n_folds
        self.fold_reg_lambda = fold_reg_lambda
        self.pool_series = pool_series

        # Ablation controls
        self.scaler_scope = scaler_scope
        self.scaler_method = scaler_method
        self.fixed_local_ratio = fixed_local_ratio
        self.fixed_noise_type = fixed_noise_type
        self.fixed_aug_sigma = fixed_aug_sigma

        T = self.data.shape[1]

        # first train_ratio data is used as training set during search
        self.split_idx_1 = int(T * train_ratio)
        # the final test_ratio is test, which is not used during search
        self.split_idx_2 = int(T * (1 - test_ratio))

        # use default 0.7/0.1/0.2 split for benchmark if the splits is not provided
        self.split_starts = split_starts
        self.split_ends = split_ends

        # Precompute fold boundaries for expanding window k-fold CV
        if n_folds > 1:
            trainval_len = self.split_idx_2
            fold_size = trainval_len // (n_folds + 1)
            # Boundaries: [fold_size, 2*fold_size, ..., (n_folds+1)*fold_size]
            self.fold_boundaries = [fold_size * (i + 1) for i in range(n_folds + 1)]
        else:
            self.fold_boundaries = None

    def _evaluate_fold(self, train_end, val_end, scaler_config, aug_config, lookback):
        """
        Evaluate a single train/val split.

        Args:
            train_end: end index of training data
            val_end: end index of validation data
            scaler_config: scaler configuration dict
            aug_config: augmentation configuration dict
            lookback: context length for this evaluation

        Returns:
            mse_per_alpha: (K,) tensor of MSE for each alpha, or None if fold is invalid
        """
        try:
            X_train_w, Y_train, X_val_w, Y_val, _, _, scalers = \
                get_prepared_data(self.data, lookback, self.horizon,
                                  train_end, val_end, scaler_config)
        except ValueError:
            return None

        if X_train_w is None or X_val_w is None:
            return None

        if self.is_batched and not self.pool_series:
            # Batched mode: keep series separate (S, N_s, L)
            # X_train_w shape: (S, N_windows, L)
            # Y_train shape: (S, N_windows, H)

            # Solve with batched solver (each series gets own model)
            Theta = self.solver.solve_batched(
                X_train_w, Y_train, self.alphas,
                scaler=scalers['train'],
                aug_config=aug_config
            )  # (K, S, F, H)

            # Predict per-series (returns CPU tensor to avoid OOM)
            Y_pred = self.solver.predict_batched(
                X_val_w, Theta, scaler=scalers['val']
            )  # (K, S, N_val, H) on CPU

            if isinstance(scalers['val'], GlobalScaler):
                Y_pred = scalers['val'].inv_transform(Y_pred)

            # Compute MSE on CPU to avoid large GPU allocation
            # Y_val shape: (S, N_val, H)
            # Y_pred shape: (K, S, N_val, H)
            diff = Y_pred - Y_val.cpu().unsqueeze(0)  # (K, S, N_val, H)
            mse_per_series = (diff ** 2).mean(dim=(-2, -1))  # (K, S)
            mse_per_alpha = mse_per_series.mean(dim=1)  # (K,) - average across series
        else:
            # Single series mode: concatenate windows
            X_train = X_train_w.reshape(-1, lookback)
            Y_train = Y_train.reshape(-1, Y_train.shape[-1])
            X_val = X_val_w.reshape(-1, lookback)
            Y_val = Y_val.reshape(-1, Y_val.shape[-1])

            # Solve
            Theta = self.solver.solve(
                X_train, Y_train, self.alphas,
                scaler=scalers['train'],
                aug_config=aug_config
            )

            # Predict
            Y_pred = self.solver.predict(X_val, Theta, scaler=scalers['val'])
            if isinstance(scalers['val'], GlobalScaler):
                Y_pred = scalers['val'].inv_transform(Y_pred)
            Y_pred = Y_pred.cpu()

            # Compute MSE per alpha
            diff = Y_pred - Y_val.unsqueeze(0)  # (K, N_window, H)
            mse_per_alpha = (diff ** 2).mean(dim=(-2, -1))  # (K,)

        return mse_per_alpha

    def __call__(self, trial):
        # Suggest lookback using log-scale search (more efficient for context length)
        lookback = trial.suggest_int("lookback", self.min_lookback, self.max_lookback, log=True)

        # Scaler scope (global vs local)
        if self.scaler_scope == "search":
            scaler_scope = trial.suggest_categorical("scaler_scope", ["global", "local"])
        else:
            scaler_scope = self.scaler_scope

        # Scaler method (mean vs robust)
        if self.scaler_method == "search":
            scaler_method = trial.suggest_categorical("scaler_method", ["mean", "robust"])
        else:
            scaler_method = self.scaler_method

        # Local ratio (only relevant for local normalization)
        if scaler_scope == "local":
            if self.fixed_local_ratio is not None:
                local_ratio = self.fixed_local_ratio
            else:
                local_ratio = trial.suggest_float("local_ratio", 1e-3, 1.0, log=True)
        else:
            local_ratio = 1.0  # Not used for global, but needed for config dict

        scaler_config = {
            "method": scaler_method, "scope": scaler_scope,
            "local_ratio": local_ratio, "q_min": 0.25, "q_max": 0.75
        }

        # Augmentation hyperparameters
        if self.fixed_noise_type is not None:
            noise_type = self.fixed_noise_type
        else:
            noise_type = trial.suggest_categorical("noise_type", ["none", "time", "freq"])

        if noise_type == "none":
            sigma = 0.0
        else:
            if self.fixed_aug_sigma is not None:
                sigma = self.fixed_aug_sigma
            else:
                # Log-scale search for sigma (small values matter more)
                sigma = trial.suggest_float("aug_sigma", 1e-3, 0.5, log=True)
        aug_config = {"noise_type": noise_type, "sigma": sigma}

        if self.n_folds == 1:
            # Single split evaluation
            mse_per_alpha = self._evaluate_fold(
                self.split_idx_1, self.split_idx_2, scaler_config, aug_config, lookback)
            if mse_per_alpha is None:
                raise optuna.TrialPruned()
        else:
            # Expanding window k-fold CV
            fold_mses = []
            for fold_idx in range(self.n_folds):
                train_end = self.fold_boundaries[fold_idx]
                val_end = self.fold_boundaries[fold_idx + 1]
                mse = self._evaluate_fold(
                    train_end, val_end, scaler_config, aug_config, lookback)
                if mse is not None:
                    fold_mses.append(mse)

            if not fold_mses:
                raise optuna.TrialPruned()

            # Stack fold MSEs: (n_folds, K)
            fold_mses_tensor = torch.stack(fold_mses)
            mean_mse = fold_mses_tensor.mean(dim=0)  # (K,)

            # Apply fold variance regularization if lambda > 0
            if self.fold_reg_lambda > 0 and len(fold_mses) > 1:
                std_mse = fold_mses_tensor.std(dim=0)  # (K,)
                mse_per_alpha = mean_mse + self.fold_reg_lambda * std_mse
                # Log fold statistics for analysis
                best_mean_idx = mean_mse.argmin()
                trial.set_user_attr("fold_mean", mean_mse[best_mean_idx].item())
                trial.set_user_attr("fold_std", std_mse[best_mean_idx].item())
            else:
                mse_per_alpha = mean_mse

        # Find best alpha from (averaged) MSE
        mse_per_alpha = mse_per_alpha.squeeze()
        best_val_mse, best_idx = torch.min(mse_per_alpha, dim=0)

        trial.set_user_attr("best_alpha", self.alphas[best_idx].item())
        return best_val_mse.item()

    def refit_test(self, best_params, best_alpha_val, use_train_val=False):
        """
        Returns DICTIONARY containing raw predictions/targets for full test set.
        """
        # Get hyperparameters: use fixed values if set, otherwise from best_params
        lookback = best_params.get("lookback", self.min_lookback)

        # Scaler config: use fixed values if set, otherwise from best_params
        if self.scaler_method == "search":
            scaler_method = best_params.get("scaler_method", "mean")
        else:
            scaler_method = self.scaler_method

        if self.scaler_scope == "search":
            scaler_scope = best_params.get("scaler_scope", "local")
        else:
            scaler_scope = self.scaler_scope

        if self.fixed_local_ratio is not None:
            local_ratio = self.fixed_local_ratio
        else:
            local_ratio = best_params.get("local_ratio", 1.0)

        scaler_config = {
            "method": scaler_method, "scope": scaler_scope,
            "local_ratio": local_ratio, "q_min": 0.25, "q_max": 0.75
        }

        # Augmentation config: use fixed values if set, otherwise from best_params
        if self.fixed_noise_type is not None:
            noise_type = self.fixed_noise_type
        else:
            noise_type = best_params.get("noise_type", "none")

        if noise_type == "none":
            sigma = 0.0
        else:
            if self.fixed_aug_sigma is not None:
                sigma = self.fixed_aug_sigma
            else:
                sigma = best_params.get("aug_sigma", 0.0)
        aug_config = {"noise_type": noise_type, "sigma": sigma}

        # Use Train+Val (split_idx_2)
        start = self.split_ends[1] if use_train_val else self.split_starts[1]
        end = self.split_ends[1]
        X_train_w, Y_train, _, _, X_test_w, Y_test, scalers = \
            get_prepared_data(self.data, lookback, self.horizon,
                              start, end, scaler_config)

        S = self.n_series
        N_test = X_test_w.shape[1] if X_test_w.dim() == 3 else X_test_w.shape[0]

        X_train = X_train_w.reshape(-1, lookback)
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])
        X_test = X_test_w.reshape(-1, lookback)
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        single_alpha = torch.tensor([best_alpha_val], device=self.device)

        # Solve with scaler
        Theta = self.solver.solve(
            X_train, Y_train, single_alpha,
            scaler=scalers['test'],
            aug_config=aug_config,
        )

        # Predict with scaler
        Y_pred = self.solver.predict(X_test, Theta, scaler=scalers['test'])
        Y_pred = Y_pred.reshape_as(Y_test)  # (S*N_test, H) or (N_test, H)

        # For GlobalScaler, apply inv_transform to get back to original scale
        if isinstance(scalers['test'], GlobalScaler):
            Y_pred = scalers['test'].inv_transform(Y_pred)
        Y_pred = Y_pred.to(torch.float32).cpu()

        if self.pool_series and self.is_batched:
            # Reshape to per-series: (S*N_test, H) -> (S, N_test, H)
            H = Y_pred.shape[-1]
            Y_pred = Y_pred.reshape(S, N_test, H)
            Y_test_per = Y_test.reshape(S, N_test, H)
            per_series_mse = ((Y_pred - Y_test_per)**2).mean(dim=(-2, -1))  # (S,)
            return {
                'test_mse': per_series_mse.mean().item(),
                'per_series_mse': per_series_mse,  # (S,)
                'raw_preds': Y_pred,  # (S, N_test, H)
            }

        # Calculate simple mean for logging
        mse = ((Y_pred - Y_test)**2).mean().item()
        return {
            'test_mse': mse,
            'raw_preds': Y_pred,
        }

class MultiOutputGlobalSeriesObjectiveWrapper:
    def __init__(self, data, lookback, horizons, alphas, device, use_local_norm=True):
        """
        Global ridge regression wrapper for multi-output forecasting.

        Args:
            data: (T, S) time series data
            lookback: context length
            horizons: list of forecast horizons
            alphas: regularization strength (scalar)
            device: torch device
            use_local_norm: if True, use LocalNormScaler (default True for reference compatibility)
        """
        self.data = data
        self.lookback = lookback
        self.horizon = horizons
        self.alphas = alphas
        self.device = device
        self.use_local_norm = use_local_norm
        self.solver = RidgeSolver(device)

    def refit_test(self, n_train, n_test):
        """
        Train on training data and predict on test data.

        Returns:
            dict with 'raw_preds': (S, N_test, H) predictions
        """
        T, S = self.data.shape
        L, H = self.lookback, len(self.horizon)
        max_horizon = max(self.horizon)

        # Create scaler based on setting
        if self.use_local_norm:
            scaler = LocalNormScaler(StandardStrategy(), L, L)
        else:
            scaler = None  # No scaler, use intercept

        # Create training windows
        train_wins = self.data[:n_train].transpose(0, 1).unfold(1, L + max_horizon, 1)
        # train_wins shape: (S, N_train_windows, L + max_horizon)

        X_tr = train_wins[:, :, :L].reshape(-1, L)  # (S*N, L)
        Y_tr = train_wins[:, :, L:L+H].reshape(-1, H)  # (S*N, H)

        # Solve ridge regression using unified solver
        single_alpha = torch.tensor([self.alphas], device=self.device)
        Theta = self.solver.solve(X_tr, Y_tr, single_alpha, scaler=scaler)

        # Create test windows: start from test_start - L to include context
        test_start = T - n_test
        test_wins = self.data[test_start - L:].transpose(0, 1).unfold(1, L + max_horizon, 1)
        # test_wins shape: (S, N_test_windows, L + max_horizon)

        X_te = test_wins[:, :, :L]  # (S, N_test, L)
        N_test = test_wins.shape[1]

        # Predict using unified solver
        X_te_flat = X_te.reshape(-1, L)  # (S*N_test, L)
        Y_pred = self.solver.predict(X_te_flat, Theta, scaler=scaler)

        # Reshape to (S, N_test, H)
        Y_pred = Y_pred.squeeze(0).reshape(S, N_test, H)
        Y_pred = Y_pred.detach().to(torch.float32).cpu()

        return {'raw_preds': Y_pred}

# ==========================================
# 6. Main Loop
# ==========================================
def search_local_models(data, series_names, lookbacks, horizons, alphas, n_trials, device,
                        horizon_group_size=1, series_group_size=1,
                        search_train_ratio=0.4, search_test_ratio=0.2,
                        split_starts=None, split_ends=None, n_folds=1, fold_reg_lambda=0.0,
                        scaler_scope="search", scaler_method="search",
                        fixed_local_ratio=None, fixed_noise_type=None, fixed_aug_sigma=None,
                        pool_series=False, seed=None):
    best_results = []
    raw_results = {}  # Storage for alignment
    T, S = data.shape

    # Handle special value: -1 means all series
    if series_group_size <= 0:
        series_group_size = S

    # Valid lookback calculation
    valid_lookbacks = [int(lb) for lb in lookbacks if lb < int(T * 0.3) - max(horizons)]
    print(f"Valid Lookbacks: {valid_lookbacks}")

    # Timing statistics
    search_times = []
    total_start_time = time.time()

    # Iterate over series groups
    for sg_idx in range(0, S, series_group_size):
        series_group = list(range(sg_idx, min(sg_idx + series_group_size, S)))
        series_names_in_group = [series_names[i] for i in series_group]
        group_display = (series_names_in_group[0] if len(series_group) == 1
                         else f"[{series_names_in_group[0]}..{series_names_in_group[-1]}]")
        print(f"\n[Series Group: {group_display}]")

        for h_idx in range(0, len(horizons), horizon_group_size):
            horizon_group = horizons[h_idx:h_idx + horizon_group_size]

            # HP Search: use series group (grouped objective for robust HP selection)
            sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
            study = optuna.create_study(direction="minimize", sampler=sampler)
            wrap = SingleObjectiveWrapper(
                data, series_group if len(series_group) > 1 else series_group[0],
                valid_lookbacks, horizon_group, alphas, device,
                train_ratio=search_train_ratio, test_ratio=search_test_ratio,
                split_starts=split_starts, split_ends=split_ends,
                n_folds=n_folds, fold_reg_lambda=fold_reg_lambda,
                scaler_scope=scaler_scope, scaler_method=scaler_method,
                fixed_local_ratio=fixed_local_ratio, fixed_noise_type=fixed_noise_type,
                fixed_aug_sigma=fixed_aug_sigma,
                pool_series=pool_series)

            # Time the optimization
            search_start = time.time()
            study.optimize(wrap, n_trials=n_trials)
            search_elapsed = time.time() - search_start
            search_times.append(search_elapsed)

            # Extract best results from study
            params = study.best_trial.params
            noise_type = params.get('noise_type', 'none')
            aug_sigma = params.get('aug_sigma', 0.0) if noise_type != 'none' else 0.0
            local_ratio = params.get('local_ratio', fixed_local_ratio if fixed_local_ratio else 1.0)
            best = {
                'lookback': params['lookback'],
                'val_mse': study.best_value,
                'params': params,
                'best_alpha': study.best_trial.user_attrs['best_alpha'],
                'scaler_method': params.get('scaler_method', 'mean'),
                'scaler_scope': params.get('scaler_scope', 'local'),
                'noise_type': noise_type,
                'aug_sigma': aug_sigma,
                'local_ratio': local_ratio
            }

            if pool_series and len(series_group) > 1:
                # REFIT: single pooled model for entire series group
                metrics = wrap.refit_test(best['params'], best['best_alpha'])
                # metrics['raw_preds'] shape: (S_group, N_test, H)
                for i, s_idx in enumerate(series_group):
                    raw_results[(s_idx, horizon_group[0])] = metrics['raw_preds'][i]
                    rec = best.copy()
                    rec.update({
                        'series': series_names[s_idx],
                        'horizon': horizon_group,
                        'test_mse': metrics['per_series_mse'][i].item()
                    })
                    best_results.append(rec)
            else:
                # REFIT: train individual model per series with shared HPs
                for s_idx in series_group:
                    refitter = SingleObjectiveWrapper(
                        data, s_idx, valid_lookbacks, horizon_group, alphas, device,
                        train_ratio=search_train_ratio, test_ratio=search_test_ratio,
                        split_starts=split_starts, split_ends=split_ends,
                        scaler_scope=scaler_scope, scaler_method=scaler_method,
                        fixed_local_ratio=fixed_local_ratio, fixed_noise_type=fixed_noise_type,
                        fixed_aug_sigma=fixed_aug_sigma)
                    metrics = refitter.refit_test(best['params'], best['best_alpha'])

                    # Save Raw Vectors
                    raw_results[(s_idx, horizon_group[0])] = metrics['raw_preds']

                    # Save Summary
                    rec = best.copy()
                    rec.update({
                        'series': series_names[s_idx],
                        'horizon': horizon_group,
                        'test_mse': metrics['test_mse']
                    })
                    best_results.append(rec)

            display_horizon = (horizon_group if len(horizon_group) == 1
                               else f"[{horizon_group[0]}-{horizon_group[-1]}]")
            aug_str = f"aug={noise_type}" + (f"({aug_sigma:.3f})" if noise_type != 'none' else "")
            print(f"  > H={display_horizon}: L={best['lookback']}, "
                  f"α={best['best_alpha']:.2e}, scaler={best['scaler_scope']}, "
                  f"norm={best['scaler_method']}, {aug_str}. "
                  f"Val={best['val_mse']:.4f}, Time={search_elapsed:.2f}s")

    # Print timing summary
    total_elapsed = time.time() - total_start_time
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    print(f"\n[Timing Summary]")
    print(f"  Total searches: {len(search_times)}")
    print(f"  Average search time: {avg_search_time:.2f}s")
    print(f"  Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f}min)")

    return best_results, raw_results

def search_global_baseline(data, lookback, alphas, device, split_starts=None, split_ends=None,
                           use_local_norm=True):
    """
    Run global ridge regression baseline for multiple forecast horizons.

    Args:
        data: (T, S) time series data
        lookback: context length
        alphas: regularization strength
        device: torch device
        split_starts: list of split start indices [train_start, val_start, test_start]
        split_ends: list of split end indices [train_end, val_end, test_end]
        use_local_norm: if True, use LocalNormScaler (default True for reference compatibility)

    Returns:
        dict mapping cutoff -> (S, N_test, cutoff) predictions
    """
    raw_results = {}
    horizons_of_interest = [96, 192, 336, 720]
    if split_starts is not None and split_ends is not None:
        n_train = split_starts[1]
        n_val = split_ends[1] - n_train
        n_test = split_ends[2] - split_starts[2]
    else:
        n_train = int(data.shape[0] * 0.7)
        n_test = int(data.shape[0] * 0.2)
        n_val = data.shape[0] - n_train - n_test

    for cutoff in horizons_of_interest:
        horizons_list = list(range(1, cutoff + 1))
        wrapper = MultiOutputGlobalSeriesObjectiveWrapper(
            data, lookback, horizons_list, alphas, device, use_local_norm=use_local_norm
        )
        results = wrapper.refit_test(n_train=n_train, n_test=n_test)
        raw_results[cutoff] = results['raw_preds']

    return raw_results

# ==========================================
# 7. Alignment & Comparison Logic
# ==========================================
def align_and_compare(data: torch.Tensor, local_raw: torch.Tensor, global_raw: torch.Tensor, L: int, H: int, horizon_group_size):
    """
    local_raw: list of tensors [(S, N_test[1]), (S, N_test[2]), ..., (S, N_test[720])] if we assume the max horizon is H
    global_raw: a stensor of shape (S, N_test[H], H)
    """
    print("\n[Final Alignment & Benchmarking]")
    assert global_raw.shape[-1] == H

    N_test_at_cutoff = global_raw.shape[1]
    global_diff = global_raw - data
    global_mse = (global_diff**2).mean().item()
    global_mae = global_diff.abs().mean().item()

    local_raw = torch.cat([pred[:, :N_test_at_cutoff] for pred in local_raw[:math.ceil(H/horizon_group_size)]], dim=-1)
    local_diff = local_raw - data
    local_mse = (local_diff**2).mean().item()
    local_mae = local_diff.abs().mean().item()
    
    print(f"  > H={H}: Local {local_mse:.4f} vs Global {global_mse:.4f}")
    return dict(mse=local_mse, mae=local_mae), dict(mse=global_mse, mae=global_mae)


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_csv", type=str, default="data/weather.csv")
        parser.add_argument("--output_dir", type=str, default="results")
        parser.add_argument("--local_horizon_group_size", type=int, default=24)
        parser.add_argument("--local_series_group_size", type=int, default=1,
                            help="Number of series to group for HP search (1 = per-series, -1 = all)")
        parser.add_argument("--instance_norm", action="store_true", default=False,
                            help="Use instance normalization for global baseline (default: False)")
        parser.add_argument("--no_instance_norm", action="store_false", dest="instance_norm",
                            help="Disable instance normalization for global baseline")
        parser.add_argument("--n_folds", type=int, default=1,
                            help="Number of folds for expanding window CV (1 = single split, no CV)")
        parser.add_argument("--fold_reg_lambda", type=float, default=0.0,
                            help="Regularization weight for fold variance (0 = no regularization)")
        parser.add_argument("--n_trials", type=int, default=30)

        # Ablation controls
        parser.add_argument("--scaler_scope", type=str, default="search",
                            choices=["global", "local", "search"],
                            help="Normalization scope: global (dataset-level), local (per-sample), or search both")
        parser.add_argument("--scaler_method", type=str, default="search",
                            choices=["mean", "robust", "search"],
                            help="Normalization method: mean (standard), robust (median/IQR), or search both")
        parser.add_argument("--fixed_local_ratio", type=float, default=None,
                            help="Fix local_ratio instead of searching (only relevant for local scope)")
        parser.add_argument("--fixed_noise_type", type=str, default=None,
                            choices=["none", "time", "freq"],
                            help="Fix augmentation type instead of searching")
        parser.add_argument("--fixed_aug_sigma", type=float, default=None,
                            help="Fix augmentation sigma instead of searching")
        parser.add_argument("--pool_series", action="store_true", default=False,
                            help="Pool training data across series in a group (single model per group)")
        parser.add_argument("--fixed_lookback", type=int, default=None,
                            help="Fix lookback to a single value instead of searching")
        parser.add_argument("--seed", type=int, default=None,
                            help="Seed for numpy/torch and Optuna TPE sampler (for multi-seed runs)")
        args = parser.parse_args()

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        os.makedirs(args.output_dir, exist_ok=True)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Config
        if args.fixed_lookback is not None:
            LOOKBACKS = np.array([args.fixed_lookback], dtype=int)
        else:
            LOOKBACKS = np.logspace(5, 11, 19, base=2, dtype=int)
        HORIZONS = list(range(1, 721))
        HORIZON_GROUP_SIZE = args.local_horizon_group_size
        SERIES_GROUP_SIZE = args.local_series_group_size
        ALPHAS = torch.logspace(-6, 4, 21, device=DEVICE)
        N_TRIALS = args.n_trials

        # Load
        df = pd.read_csv(args.input_csv)
        scaler = StandardScaler()
        if 'ETT' not in args.input_csv:
            n_train = int(len(df) * 0.7)
            n_test = int(len(df) * 0.2)
            n_val = len(df) - n_train - n_test
            total_len = len(df)
            search_train_ratio = 0.4
            search_test_ratio = 0.2
        else:
            # Match reference: ETT datasets are truncated to exactly 20 months
            # Train: 12 months, Val: 4 months, Test: 4 months
            if 'h' in args.input_csv:
                n_train = 12 * 30 * 24          # 8640
                n_val = 4 * 30 * 24             # 2880
                n_test = 4 * 30 * 24            # 2880
                total_len = 12 * 30 * 24 + 8 * 30 * 24  # 14400 (20 months)
            else:
                n_train = 12 * 30 * 24 * 4      # 34560
                n_val = 4 * 30 * 24 * 4         # 11520
                n_test = 4 * 30 * 24 * 4        # 11520
                total_len = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4  # 57600 (20 months)
            search_train_ratio = (n_train + n_val) / total_len * 0.5   # use half of the data as validation
            search_test_ratio = n_test / total_len
        split_starts = [0, n_train, n_train + n_val]
        split_ends = [n_train, n_train + n_val, total_len]
        # Truncate data to match reference implementation
        df_values = df.drop(columns=['date']).values[:total_len]
        scaler.fit(df_values[:n_train])
        data = torch.tensor(scaler.transform(df_values), dtype=torch.float32)
        print("Split start: ", split_starts)
        print("Split end: ", split_ends)

        # 1. Local Search (Returns Raw Preds)
        best_df, local_raw_ind = search_local_models(
            data, df.columns[1:], LOOKBACKS, HORIZONS, ALPHAS, n_trials=N_TRIALS, device=DEVICE,
            horizon_group_size=HORIZON_GROUP_SIZE, series_group_size=SERIES_GROUP_SIZE,
            search_train_ratio=search_train_ratio, search_test_ratio=search_test_ratio,
            split_starts=split_starts, split_ends=split_ends,
            n_folds=args.n_folds, fold_reg_lambda=args.fold_reg_lambda,
            scaler_scope=args.scaler_scope, scaler_method=args.scaler_method,
            fixed_local_ratio=args.fixed_local_ratio, fixed_noise_type=args.fixed_noise_type,
            fixed_aug_sigma=args.fixed_aug_sigma,
            pool_series=args.pool_series, seed=args.seed
        )
        pd.DataFrame(best_df).to_csv(f"{args.output_dir}/local_results.csv", index=False)
        local_raw = []
        for h_idx in range(0, len(HORIZONS), HORIZON_GROUP_SIZE):
            horizon_group = HORIZONS[h_idx:h_idx + HORIZON_GROUP_SIZE]
            local_raw.append(
                torch.stack(
                    [local_raw_ind[(s_idx, horizon_group[0])] for s_idx in range(data.shape[1])], 
                    dim=0))

        # 2. Global Baseline (Returns Raw Preds)
        global_lookback = 720
        global_raw = search_global_baseline(
            data, global_lookback, 1e-5, DEVICE,
            split_starts=split_starts, split_ends=split_ends,
            use_local_norm=args.instance_norm
        )

        # 3. Align and Compare
        cutoffs = [96, 192, 336, 720]
        final_bench = {}
        test_start = split_starts[2]
        assert test_start > global_lookback
        test_data = data[test_start - global_lookback:]

        for cutoff in [96, 192, 336, 720]:
            test_wins = test_data.transpose(0, 1).unfold(1, global_lookback + cutoff, 1)
            Y_te = test_wins[:, :, global_lookback:global_lookback+cutoff] #.reshape(-1, cutoff)
            local_metrics, global_metrics = align_and_compare(Y_te, local_raw, global_raw[cutoff], global_lookback, cutoff, HORIZON_GROUP_SIZE)
            final_bench[cutoff] = [local_metrics['mse'], global_metrics['mse'], local_metrics['mae'], global_metrics['mae']]
            print(f"Cutoff = {cutoff}: ", [local_metrics['mse'], global_metrics['mse'], local_metrics['mae'], global_metrics['mae']])
        bench_df = pd.DataFrame(final_bench).transpose()
        bench_df.to_csv(f"{args.output_dir}/benchmark_comparison.csv", header=['Local MSE', 'Global MSE', 'Local MAE', 'Global MAE'])

        # save the first predictions for visualization
        MAX_HORIZONS = max(HORIZONS)
        VIS_IDX = 336
        local_pred = torch.cat([local_p[:, VIS_IDX, :] for local_p in local_raw], dim=-1).numpy()   # (S, H)
        global_pred = global_raw[MAX_HORIZONS][:, VIS_IDX].numpy()   # (S, H)
        gt = data[test_start+VIS_IDX:test_start+MAX_HORIZONS+VIS_IDX, :].transpose(0, 1)
        np.save(os.path.join(args.output_dir, "predictions.npy"), dict(gt=gt, local_pred=local_pred, global_pred=global_pred))

        # Plot
        S, H = gt.shape
        fig, axes = plt.subplots(S, 1, figsize=(12, 3*S), sharex=True)
        if S == 1: axes = [axes]

        for i in range(S):
            ax = axes[i]
            ax.plot(gt[i], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
            ax.plot(local_pred[i], label='Local (Ours)', color='blue', linestyle='--', alpha=0.8)
            ax.plot(global_pred[i], label='Global (Baseline)', color='red', linestyle=':', alpha=0.8)

            ax.set_ylabel(f'Series {i}')
            ax.set_title(f'Forecast Comparison: Series {i}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right')

        plt.xlabel('Horizon Step (1-720)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "forecast_comparison_all_horizons.png"))


        # the predition at H=336
        VIS_HORIZON = 336
        gt = data[test_start + VIS_HORIZON-1:, :].transpose(0, 1) # (S, N)
        n_test = gt.shape[1]
        global_pred = global_raw[VIS_HORIZON][:, :, VIS_HORIZON - 1].numpy()   # (S, N)
        assert global_pred.shape[1] == n_test
        local_pred = torch.cat([local_p[:, :n_test, :] for local_p in local_raw[:math.ceil(VIS_HORIZON/HORIZON_GROUP_SIZE)]], dim=-1)
        local_pred = local_pred[:, :, VIS_HORIZON - 1].numpy()   # (S, H)
        # Plot
        S, H = gt.shape
        fig, axes = plt.subplots(S, 1, figsize=(12, 3*S), sharex=True)
        if S == 1: axes = [axes]

        for i in range(S):
            ax = axes[i]
            ax.plot(gt[i], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
            ax.plot(local_pred[i], label='Local (Ours)', color='blue', linestyle='--', alpha=0.8)
            ax.plot(global_pred[i], label='Global (Baseline)', color='red', linestyle=':', alpha=0.8)

            ax.set_ylabel(f'Series {i}')
            ax.set_title(f'Forecast Comparison: Series {i}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right')

        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"forecast_comparison_horizon{VIS_HORIZON}.png"))


if __name__ == "__main__":
    main()
