"""Prepare data assets for the project website.

Produces three artefacts:
  1. app/data/ETTh1.csv         — verbatim copy of data/ETTh1.csv (the only bundled dataset).
  2. app/data/manifest.json     — per-(series, horizon) searched-optimal HPs for ETTh1.
  3. assets/widget_a_data.json  — per-dataset snippets + L*(H) tables for the inline Widget A.

Run from the repo root:
    python website/build_data.py
"""

from __future__ import annotations

import ast
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Make the repo root importable so we can reuse the optuna_ridge Ridge solver
# and scaler classes for per-slice inference at the widget's anchor.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optuna_ridge import LocalNormScaler, RidgeSolver, RobustStrategy, StandardStrategy
from refit_and_save_weights import compute_splits


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_ROOT = REPO_ROOT / "website"

ETTH1_SRC = REPO_ROOT / "data" / "ETTh1.csv"
ETTH1_DST = WEBSITE_ROOT / "app" / "data" / "ETTh1.csv"
MANIFEST_DST = WEBSITE_ROOT / "app" / "data" / "manifest.json"
WIDGET_A_DST = WEBSITE_ROOT / "assets" / "widget_a_data.json"
WIDGET_A_JS = WEBSITE_ROOT / "assets" / "widget_a_data.js"
FIGURES_DST = WEBSITE_ROOT / "assets" / "figures_data.json"
FIGURES_JS = WEBSITE_ROOT / "assets" / "figures_data.js"

ETTH1_SERIES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
CUTOFFS = [96, 192, 336, 720]

# (dataset_label, csv_path, series_col, test_anchor_offset, exp_key, color)
# test_anchor_offset = position of the animation anchor measured from test_start,
# so the prediction at the anchor is an honest out-of-sample forecast from a
# train+val refit.
WIDGET_A_DATASETS = [
    ("ETTh1",        "data/ETTh1.csv",         "OT",  336,   "etth1",       "#1f77b4"),
    ("ETTh2",        "data/ETTh2.csv",         "OT",  336,   "etth2",       "#ff7f0e"),
    ("ETTm1",        "data/ETTm1.csv",         "OT",  336,   "ettm1",       "#2ca02c"),
    ("ETTm2",        "data/ETTm2.csv",         "OT",  336,   "ettm2",       "#d62728"),
    ("Weather",      "data/weather.csv",       "OT",  336,   "weather",     "#9467bd"),
    ("Electricity",  "data/electricity.csv",   "104", 336,   "electricity", "#8c564b"),
    ("Traffic",      "data/traffic.csv",       "0",   336,   "traffic",     "#e377c2"),
    ("Exchange",     "data/exchange_rate.csv", "5",   336,   "exchange",    "#7f7f7f"),
]

WIDGET_A_HORIZONS = list(range(24, 721, 24))


def _max_horizon(value) -> int:
    """Horizon column rows can be '[96]', '[1, 2, ...]', or actual lists."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(max(value))
    if isinstance(value, str):
        return int(max(ast.literal_eval(value)))
    return int(value)


def _load_results_with_fallback(dataset_key: str, group_sizes: Iterable[int]) -> pd.DataFrame | None:
    """Try several experiment folders; return the first one that exists."""
    candidates = [f"exp3_{dataset_key}_gs{gs}" for gs in group_sizes]
    candidates += [f"exp8_{dataset_key}_sgs1_pooled"]
    for name in candidates:
        path = REPO_ROOT / "exps" / name / "local_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["_max_h"] = df["horizon"].apply(_max_horizon)
            print(f"  loaded {name} ({len(df)} rows)")
            return df
    print(f"  WARNING: no results found for {dataset_key} (tried {candidates})")
    return None


def build_etth1_copy() -> None:
    print(f"[1/4] Copying ETTh1.csv -> {ETTH1_DST.relative_to(REPO_ROOT)}")
    ETTH1_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ETTH1_SRC, ETTH1_DST)
    size_kb = ETTH1_DST.stat().st_size / 1024
    print(f"  done ({size_kb:.0f} KB)")


def _load_per_group_hps_gs48(ds_key: str) -> list[dict]:
    """Return one HP set per horizon group at gs=48, ordered by group_max.

    For ETTh1 at gs=48 + sgs=7 (paper default), HPs are shared across all 7
    series within each horizon group, so we collapse the table to 15 unique
    entries keyed by group_max (48, 96, 144, ..., 720)."""
    path = REPO_ROOT / "exps" / f"exp3_{ds_key}_gs48" / "local_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df["_max_h"] = df["horizon"].apply(_max_horizon)
    df["_min_h"] = df["horizon"].apply(
        lambda x: min(ast.literal_eval(x)) if isinstance(x, str) else min(x)
    )
    out: list[dict] = []
    for max_h in sorted(df["_max_h"].unique()):
        sub = df[df["_max_h"] == max_h].iloc[0]
        out.append({
            "group_min": int(sub["_min_h"]),
            "group_max": int(max_h),
            "lookback": int(sub["lookback"]),
            "alpha": float(sub["best_alpha"]),
            "scaler_method": str(sub["scaler_method"]),
            "scaler_scope": str(sub["scaler_scope"]),
            "noise_type": str(sub["noise_type"]),
            "aug_sigma": float(sub["aug_sigma"]),
            "local_ratio": float(sub["local_ratio"]),
            "val_mse": float(sub["val_mse"]),
            "test_mse": float(sub["test_mse"]),
        })
    return out


def build_etth1_manifest() -> None:
    print(f"[2/4] Building manifest.json")
    df = _load_results_with_fallback("etth1", group_sizes=[1, 2, 4, 6, 8, 12, 24, 48])
    if df is None:
        raise SystemExit("Cannot build manifest without ETTh1 results.")
    per_group = _load_per_group_hps_gs48("etth1")
    print(f"  loaded {len(per_group)} per-group HP entries from exp3_etth1_gs48")

    hp_table: dict[str, dict[str, dict]] = {}
    for series in ETTH1_SERIES:
        sub = df[df["series"].astype(str) == series]
        if sub.empty:
            print(f"  series {series}: no rows, skipping")
            continue
        per_h: dict[str, dict] = {}
        avail = np.sort(sub["_max_h"].unique())
        for H in CUTOFFS:
            i = int(np.argmin(np.abs(avail - H)))
            nearest = int(avail[i])
            if abs(nearest - H) > 48:
                continue
            row = sub[sub["_max_h"] == nearest].iloc[0]
            per_h[str(H)] = {
                "lookback": int(row["lookback"]),
                "alpha": float(row["best_alpha"]),
                "scaler_method": str(row["scaler_method"]),
                "scaler_scope": str(row["scaler_scope"]),
                "noise_type": str(row["noise_type"]),
                "aug_sigma": float(row["aug_sigma"]),
                "local_ratio": float(row["local_ratio"]),
                "val_mse": float(row["val_mse"]),
                "test_mse": float(row["test_mse"]),
            }
        if per_h:
            hp_table[series] = per_h

    # Paper protocol for ETTh1 (hourly, 20 months):
    #   total = 12+4+4 months × 30 × 24 = 14400 samples
    #   n_train = 12 × 30 × 24 = 8640, n_val = 2880, n_test = 2880
    # Search ratios (optuna_ridge.py:1358-1359):
    #   search_train_ratio = (n_train + n_val) / total * 0.5 = 0.4
    #   search_test_ratio  = n_test / total = 0.2
    ett_total = 14400
    ett_n_train, ett_n_val, ett_n_test = 8640, 2880, 2880
    manifest = {
        "dataset": "ETTh1",
        "csv_path": "data/ETTh1.csv",
        "series": ETTH1_SERIES,
        "cutoffs": CUTOFFS,
        "protocol": {
            "total_samples": ett_total,
            "n_train": ett_n_train,
            "n_val": ett_n_val,
            "n_test": ett_n_test,
            "search_train_ratio": 0.4,
            "search_test_ratio": 0.2,
            # Matches refit_test's default use_train_val=False
            # (optuna_ridge.py:922) — final refit trains on [0, n_train) only.
            "refit_on_trainval": False,
        },
        # Global baseline matches search_global_baseline in optuna_ridge.py
        # (lookback=720, alpha=1e-5, scaler_scope=global, no augmentation).
        "global_baseline": {
            "lookback": 720,
            "alpha": 1e-5,
            "scaler_method": "mean",
            "scaler_scope": "global",
            "noise_type": "none",
            "aug_sigma": 0.0,
            "local_ratio": 1.0,
        },
        # Per-horizon-group searched HPs (gs=48). 15 entries covering 1..720.
        "searched_per_group": per_group,
        "searched_optimum": hp_table,
        "search_space": {
            "lookback":     {"min": 32,    "max": 4096, "scale": "log", "type": "int"},
            "local_ratio":  {"min": 1e-3,  "max": 1.0,  "scale": "log", "type": "float"},
            "alpha":        {"min": 1e-3,  "max": 1e6,  "scale": "log", "type": "float"},
            "aug_sigma":    {"min": 1e-3,  "max": 0.5,  "scale": "log", "type": "float"},
            "scaler_method": ["mean", "robust"],
            "scaler_scope":  ["global", "local"],
            "noise_type":    ["none", "time", "freq"],
        },
    }

    MANIFEST_DST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_DST.write_text(json.dumps(manifest, indent=2))
    n_series = len(hp_table)
    n_entries = sum(len(v) for v in hp_table.values())
    print(f"  wrote manifest with {n_series} series, {n_entries} (series, horizon) entries")


PLAYGROUND_DATASETS = [
    {"key": "etth1",    "label": "ETTh1",    "csv": "data/ETTh1.csv",
     "exp_dir": "exp8_etth1_sgs1_pooled",    "csv_dst": "ETTh1.csv",
     "default_series": "OT"},
    {"key": "etth2",    "label": "ETTh2",    "csv": "data/ETTh2.csv",
     "exp_dir": "exp8_etth2_sgs1_pooled",    "csv_dst": "ETTh2.csv",
     "default_series": "OT"},
    {"key": "exchange", "label": "Exchange", "csv": "data/exchange_rate.csv",
     "exp_dir": "exp8_exchange_sgs1_pooled", "csv_dst": "exchange_rate.csv",
     "default_series": "OT"},
]


def _splits_for(csv_name: str, total_rows: int) -> tuple[int, int, int, int]:
    """Mirror optuna_ridge.py / refit_and_save_weights.py split logic."""
    if "ETT" not in csv_name:
        n_train = int(total_rows * 0.7)
        n_test  = int(total_rows * 0.2)
        n_val   = total_rows - n_train - n_test
        return n_train, n_val, n_test, total_rows
    if "h" in csv_name:
        return 8640, 2880, 2880, 14400
    return 12*30*24*4, 4*30*24*4, 4*30*24*4, 14400 * 4


def _parse_sgs1_hps(local_results_path: Path) -> dict[str, list[dict]]:
    """Read per-series per-horizon-group HPs from an sgs=1 local_results.csv.

    Returns {series_name: [{group_min, group_max, lookback, alpha, ...}, ...]}
    sorted by group_min. Each row covers one horizon group (e.g. 1..48)."""
    df = pd.read_csv(local_results_path)
    by_series: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        h_tuple = ast.literal_eval(row["horizon"])
        gmin = int(min(h_tuple)); gmax = int(max(h_tuple))
        # local_ratio is sometimes nested in params (older runs) — fall back.
        local_ratio = row.get("local_ratio")
        if not isinstance(local_ratio, (int, float)) or pd.isna(local_ratio):
            params = row.get("params")
            if isinstance(params, str):
                try:
                    local_ratio = ast.literal_eval(params).get("local_ratio", 1.0)
                except Exception:
                    local_ratio = 1.0
            else:
                local_ratio = 1.0
        by_series.setdefault(str(row["series"]), []).append({
            "group_min": gmin,
            "group_max": gmax,
            "lookback":  int(row["lookback"]),
            "alpha":     float(row["best_alpha"]),
            "local_ratio": float(local_ratio),
            "scaler_scope": str(row["scaler_scope"]),
            "scaler_method": str(row["scaler_method"]),
            "noise_type": str(row["noise_type"]),
            "aug_sigma": float(row["aug_sigma"]),
        })
    for s in by_series:
        by_series[s].sort(key=lambda g: g["group_min"])
    return by_series


def _bake_dataset(ds: dict) -> dict | None:
    """Build manifest entry for one playground dataset under the sgs=1 +
    refit-on-the-fly architecture.

    No weights are baked — the worker fits Ridge live for whichever series is
    active, using the per-(series, horizon-group) HPs from
    `exp8_*_sgs1_pooled/local_results.csv`. We still need: train mean/std for
    z-scoring on the client, the split protocol so test boundaries line up
    with the paper, and the per-series HP table the worker indexes into.
    """
    src_dir = REPO_ROOT / "exps" / ds["exp_dir"]
    local_results = src_dir / "local_results.csv"
    if not local_results.exists():
        return None

    csv_path = REPO_ROOT / ds["csv"]
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c != "date"]
    n_train, n_val, n_test, total_len = _splits_for(csv_path.name, len(df))
    truncated = df[cols].to_numpy(dtype=np.float64)[:total_len]
    train = truncated[:n_train]
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    max_h = 720

    # sgs=1: each series is its own series-group. sg_groups parallels series order.
    by_series = _parse_sgs1_hps(local_results)
    missing = [c for c in cols if c not in by_series]
    if missing:
        print(f"  {ds['key']}: no HP rows for {missing}, skipping")
        return None
    sg_groups_meta = [by_series[c] for c in cols]
    series_to_sg = list(range(len(cols)))

    # Test-anchor range: every anchor must allow max-L history AND max_h future
    # from the test segment. baseline_L=720 dominates max(lookback) here.
    baseline_L = 720
    anchor_max_L = max(g["lookback"] for groups in sg_groups_meta for g in groups)
    max_L = max(anchor_max_L, baseline_L)
    trainval_end = n_train + n_val
    anchor_start = max(trainval_end, max_L)
    anchor_end = total_len - max_h
    n_anchors = anchor_end - anchor_start + 1

    csv_dst = WEBSITE_ROOT / "app" / "data" / ds["csv_dst"]
    csv_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv_path, csv_dst)

    default_series = ds.get("default_series")
    if default_series not in cols:
        default_series = cols[0]

    entry = {
        "label": ds["label"],
        "csv_path": ds["csv_dst"],
        "series_names": cols,
        "default_series": default_series,
        "train_mean": [float(v) for v in train_mean],
        "train_std":  [float(v) for v in train_std],
        "test_anchor": int(anchor_end),
        "max_h": max_h,
        "n_test_anchors": int(n_anchors),
        "sg_groups": sg_groups_meta,
        "series_to_sg": series_to_sg,
        "protocol": {
            "total_samples": int(total_len),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "search_train_ratio": 0.4,
            "search_test_ratio": 0.2,
            "refit_on_trainval": False,
        },
    }
    print(f"  {ds['key']}: {len(cols)} series × {len(sg_groups_meta[0])} groups, "
          f"{n_anchors} test anchors (refit-on-the-fly; no weights baked)")
    return entry


def build_playground_weights() -> None:
    """Write per-dataset HP tables + protocol into manifest.json so the worker
    can refit Ridge live for whichever (dataset, series) is active. No β
    weights are baked under the current architecture; everything refits on
    the fly using sgs=1 (per-series) HPs."""
    print(f"[2b/4] Building playground HP tables for {len(PLAYGROUND_DATASETS)} datasets")
    datasets_meta: dict[str, dict] = {}
    for ds in PLAYGROUND_DATASETS:
        entry = _bake_dataset(ds)
        if entry is None:
            print(f"  {ds['key']}: missing local_results.csv, skipping")
            continue
        datasets_meta[ds["key"]] = entry

    # weights.bin (the binary β blob) is written by website/bake_weights.mjs
    # after this script runs. That script also patches manifest.json with
    # beta_offset / F / H fields per H-group and per_series_se for each
    # dataset's default series. We deliberately leave any existing
    # weights.bin in place — it'll be overwritten by the JS bake.
    manifest = json.loads(MANIFEST_DST.read_text())
    manifest["precomputed"] = {
        "max_h": 720,
        "default": "etth1",
        "datasets": datasets_meta,
    }
    MANIFEST_DST.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote HP tables for {len(datasets_meta)} datasets to manifest")
    print(f"  next: `node website/bake_weights.mjs` to bake β into weights.bin")


def _zscore(arr: np.ndarray) -> list[float]:
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if sigma < 1e-8:
        return [0.0] * len(arr)
    return [(x - mu) / sigma for x in arr]


def _compute_anchor_idx(snippet_len: int, lookbacks: dict[int, int]) -> int:
    """Mirror widget_a.js's geometry so Python and JS agree on the anchor position.

    The widget computes pastSpan/futureSpan from the L*(H) table and clamps
    displayStart against the snippet length. Anchor index in the snippet array
    is displayStart + pastSpan.
    """
    max_l = max(lookbacks.values()) if lookbacks else 0
    max_h = max(720, max(lookbacks.keys()) if lookbacks else 720)
    past_span = max(max_l + 20, 440)
    future_span = max_h + 20
    visible_span = past_span + future_span
    display_start = max(0, snippet_len - visible_span - 10)
    display_end = min(snippet_len, display_start + visible_span)
    display_start = max(0, display_end - visible_span)
    return display_start + past_span


def _find_weights_dir(exp_key: str) -> Path | None:
    """Return the exp8_{exp_key}_sgs{N}_pooled folder that has weights.npz."""
    base = REPO_ROOT / "exps"
    if not base.is_dir():
        return None
    for path in sorted(base.glob(f"exp8_{exp_key}_sgs*_pooled/weights.npz")):
        return path.parent
    return None


def _slice_lookbacks(weights_dir: Path, series_idx: int) -> dict[int, int]:
    """{slice_end_H: L} from weights.npz for the slices that cover series_idx.

    These are the L's the widget will actually render, so the snippet must be
    sized against them (not against an unrelated L*(H) table from exp3)."""
    npz = np.load(weights_dir / "weights.npz", allow_pickle=False)
    meta = json.loads(str(npz["__meta__"]))
    out: dict[int, int] = {}
    for entry in meta:
        if series_idx not in entry["group_series_idxs"]:
            continue
        end_h = int(max(entry["horizon_offsets"]))
        out[end_h] = int(entry["lookback"])
    return out


def _load_dataset_for_inference(csv_path: Path):
    """Load + StandardScaler-fit on train, return (data_st, scaler, series_names,
    split_starts, split_ends). data_st is (S, T) in training-fit z-score."""
    df = pd.read_csv(csv_path)
    n_train, n_val, n_test, total_len = compute_splits(csv_path, df)
    if "date" in df.columns:
        df_values = df.drop(columns=["date"]).values[:total_len].astype(np.float64)
        series_names = list(df.columns[1:])
    else:
        df_values = df.values[:total_len].astype(np.float64)
        series_names = list(df.columns)
    scaler = StandardScaler()
    scaler.fit(df_values[:n_train])
    data = torch.tensor(scaler.transform(df_values), dtype=torch.float32)
    data_st = data.transpose(0, 1).contiguous()
    split_starts = [0, n_train, n_train + n_val]
    split_ends = [n_train, n_train + n_val, total_len]
    return data_st, scaler, series_names, split_starts, split_ends


def _predict_slices(
    *,
    weights_dir: Path,
    data_st: torch.Tensor,
    scaler: StandardScaler,
    series_names: list[str],
    series_col: str,
    anchor_in_raw: int,
    snippet_mean: float,
    snippet_std: float,
    solver: RidgeSolver,
) -> list[dict]:
    """Run inference at `anchor_in_raw` for each (series-group, horizon-group)
    slice in weights.npz, return list of {start, end, L, pred} in snippet-local
    z-score units (so predictions overlay cleanly on the snippet line)."""
    npz = np.load(weights_dir / "weights.npz", allow_pickle=False)
    meta = json.loads(str(npz["__meta__"]))

    name_to_idx = {n: i for i, n in enumerate(series_names)}
    if series_col in name_to_idx:
        series_idx = name_to_idx[series_col]
    else:
        series_idx = len(series_names) - 1
        print(f"    series_col '{series_col}' not found, using last column '{series_names[series_idx]}'")

    s_mean_raw = float(scaler.mean_[series_idx])
    s_std_raw = float(np.sqrt(scaler.var_[series_idx]))

    slices: list[dict] = []
    for entry in meta:
        if series_idx not in entry["group_series_idxs"]:
            continue
        L = int(entry["lookback"])
        h_offsets = list(entry["horizon_offsets"])
        h_group = len(h_offsets)
        scaler_method = str(entry["scaler_method"])
        local_ratio = float(entry["local_ratio"])

        if anchor_in_raw - L < 0 or anchor_in_raw > data_st.shape[1]:
            print(f"    slice {entry['key']}: anchor {anchor_in_raw} with L={L} out of bounds, skipping")
            continue

        X = data_st[series_idx:series_idx + 1, anchor_in_raw - L:anchor_in_raw].to(torch.float64)
        strat = StandardStrategy() if scaler_method == "mean" else RobustStrategy()
        last_k = max(1, int(L * local_ratio))
        lns = LocalNormScaler(strategy=strat, lookback=L, last_k=last_k)

        Theta = torch.tensor(npz[entry["key"]], dtype=torch.float64).unsqueeze(0)
        Y_pred_zs = solver.predict(X, Theta, scaler=lns).squeeze(0).squeeze(0).cpu().numpy()

        Y_pred_raw = Y_pred_zs * s_std_raw + s_mean_raw
        Y_pred_snippet = (Y_pred_raw - snippet_mean) / snippet_std

        h_min = min(h_offsets)
        h_max = max(h_offsets)
        # Reorder pred to match offset order (defensive — meta order should already be sorted)
        order = sorted(range(h_group), key=lambda j: h_offsets[j])
        pred_sorted = Y_pred_snippet[order].tolist()

        slices.append({
            "start": int(h_min),
            "end": int(h_max),
            "L": L,
            "pred": [round(float(v), 4) for v in pred_sorted],
        })

    slices.sort(key=lambda s: s["start"])
    return slices


def _snippet_lookbacks(df: pd.DataFrame, series_col: str) -> dict[int, int]:
    sub = df[df["series"].astype(str) == series_col]
    if sub.empty:
        # Fall back to the first available series (some datasets use numeric series names)
        sub = df
    out: dict[int, int] = {}
    avail = np.sort(sub["_max_h"].unique())
    if len(avail) == 0:
        return out
    for H in WIDGET_A_HORIZONS:
        i = int(np.argmin(np.abs(avail - H)))
        nearest = int(avail[i])
        if abs(nearest - H) > 24:
            continue
        med = sub.loc[sub["_max_h"] == nearest, "lookback"].median()
        if not np.isnan(med):
            out[H] = int(round(float(med)))
    return out


def build_widget_a_data() -> None:
    print(f"[3/4] Building widget_a_data.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = RidgeSolver(device)

    payload = {"horizons": WIDGET_A_HORIZONS, "datasets": []}
    for label, csv_rel, series_col, test_anchor_offset, exp_key, color in WIDGET_A_DATASETS:
        csv_path = REPO_ROOT / csv_rel
        if not csv_path.exists():
            print(f"  {label}: missing {csv_rel}, skipping")
            continue

        # Load full dataset (training-fit StandardScaler) so we can run
        # inference at the same anchor we display, and inverse-transform back.
        try:
            data_st, scaler, series_names, split_starts, split_ends = _load_dataset_for_inference(csv_path)
        except Exception as e:
            print(f"  {label}: failed to load with splits ({e}); falling back to snippet-only")
            data_st = None
            scaler = None
            series_names = None
            split_starts = None
            split_ends = None

        # Sizing must use the SAME L's the widget will render. Prefer slice L's
        # from weights.npz; fall back to the exp3 L*(H) table only if no weights.
        weights_dir = _find_weights_dir(exp_key) if data_st is not None else None
        lookbacks: dict[int, int] = {}
        if weights_dir is not None and series_names is not None:
            name_to_idx = {n: i for i, n in enumerate(series_names)}
            series_idx_for_l = name_to_idx.get(series_col, len(series_names) - 1)
            lookbacks = _slice_lookbacks(weights_dir, series_idx_for_l)
        if not lookbacks:
            results_df = _load_results_with_fallback(exp_key, group_sizes=[1, 2, 4, 6, 8, 12, 24, 48])
            if results_df is not None:
                lookbacks = _snippet_lookbacks(results_df, series_col)
        max_lb = max(lookbacks.values()) if lookbacks else 0
        snippet_len = int(max(1400, min(4200, max_lb / 0.65 + 400)))
        anchor_idx = _compute_anchor_idx(snippet_len, lookbacks)

        slices: list[dict] = []
        anchor_in_raw = None
        if data_st is not None:
            test_start = split_starts[2]
            T = data_st.shape[1]
            # Anchor must be inside test and leave at least 720 GT points to the right.
            anchor_in_raw = test_start + test_anchor_offset
            anchor_in_raw = max(anchor_in_raw, test_start)
            anchor_in_raw = min(anchor_in_raw, T - 720)
            # Snippet starts so that the widget's anchor position == anchor_in_raw.
            snippet_start = anchor_in_raw - anchor_idx
            if snippet_start < 0:
                # Pathological — slide snippet right and adjust anchor_idx.
                shift = -snippet_start
                snippet_start = 0
                anchor_in_raw += shift
        else:
            snippet_start = 0

        # Read the snippet column.
        try:
            snippet_df = pd.read_csv(csv_path, usecols=[series_col],
                                     skiprows=range(1, snippet_start + 1), nrows=snippet_len)
        except ValueError:
            snippet_df = pd.read_csv(csv_path, header=0,
                                     skiprows=range(1, snippet_start + 1), nrows=snippet_len)
            if series_col not in snippet_df.columns:
                series_col_real = snippet_df.columns[-1]
                print(f"  {label}: column {series_col} missing, using {series_col_real}")
                snippet_df = snippet_df[[series_col_real]]

        series_arr = snippet_df.iloc[:, 0].to_numpy(dtype=float)
        snippet_mean = float(np.mean(series_arr))
        snippet_std = float(np.std(series_arr))
        if snippet_std < 1e-8:
            snippet_std = 1.0
        normed = (series_arr - snippet_mean) / snippet_std

        if weights_dir is not None and anchor_in_raw is not None:
            try:
                slices = _predict_slices(
                    weights_dir=weights_dir,
                    data_st=data_st,
                    scaler=scaler,
                    series_names=series_names,
                    series_col=series_col,
                    anchor_in_raw=anchor_in_raw,
                    snippet_mean=snippet_mean,
                    snippet_std=snippet_std,
                    solver=solver,
                )
                print(f"    {label}: {len(slices)} slices from {weights_dir.name}")
            except Exception as e:
                print(f"    {label}: inference failed ({e}); skipping predictions")
                slices = []
        elif data_st is not None:
            print(f"    {label}: no weights.npz found for exp_key '{exp_key}'")

        payload["datasets"].append({
            "label": label,
            "exp_key": exp_key,
            "series_col": series_col,
            "color": color,
            "snippet": [round(float(v), 4) for v in normed],
            "snippet_start": int(snippet_start),
            "snippet_len": int(len(normed)),
            "anchor_idx": int(anchor_idx),
            "lookback_at_h": {str(h): v for h, v in sorted(lookbacks.items())},
            "slices": slices,
        })
        print(f"  {label}: snippet={len(normed)} (max L*={max_lb}), {len(lookbacks)} L*(H) entries, anchor_idx={anchor_idx}")

    WIDGET_A_DST.parent.mkdir(parents=True, exist_ok=True)
    js_blob = json.dumps(payload, separators=(",", ":"))
    WIDGET_A_DST.write_text(js_blob)
    # Also emit a JS file that assigns to a global, so the article can load it
    # via <script src=...> instead of fetch() — works on file:// and avoids
    # async-race / CSP issues.
    WIDGET_A_JS.write_text("window.__WIDGET_A_DATA = " + js_blob + ";\n")
    size_kb = WIDGET_A_DST.stat().st_size / 1024
    print(f"  wrote {WIDGET_A_DST.relative_to(REPO_ROOT)} + .js ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Figures data — aggregated CSV → JSON for native Plotly redraws.
# ---------------------------------------------------------------------------

DATASET_KEYS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic", "exchange"]
DATASET_LABELS = {
    "etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2",
    "weather": "Weather", "electricity": "Electricity",
    "traffic": "Traffic", "exchange": "Exchange",
}
DATASET_COLORS = {
    "etth1": "#1f77b4", "etth2": "#ff7f0e", "ettm1": "#2ca02c", "ettm2": "#d62728",
    "weather": "#9467bd", "electricity": "#8c564b", "traffic": "#e377c2", "exchange": "#7f7f7f",
}


def _fit_power_law(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Returns (a, b) such that y ≈ a * x^b, fit in log-log space."""
    arr_x = np.array(xs, dtype=float)
    arr_y = np.array(ys, dtype=float)
    mask = (arr_x > 0) & (arr_y > 0) & np.isfinite(arr_x) & np.isfinite(arr_y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    log_x = np.log(arr_x[mask])
    log_y = np.log(arr_y[mask])
    b, log_a = np.polyfit(log_x, log_y, 1)
    return float(np.exp(log_a)), float(b)


def _aggregate_lookback_curve(ds_key: str) -> tuple[list[int], list[float], float, float]:
    """Median L* at each horizon, plus power-law (a, b) fit."""
    df = _load_results_with_fallback(ds_key, group_sizes=[1, 2, 4, 6, 8, 12, 24, 48])
    if df is None:
        return [], [], float("nan"), float("nan")
    horizons = sorted(int(h) for h in df["_max_h"].unique() if 24 <= h <= 720)
    medians: list[float] = []
    valid_h: list[int] = []
    for h in horizons:
        sub = df.loc[df["_max_h"] == h, "lookback"]
        if not sub.empty:
            valid_h.append(h)
            medians.append(float(sub.median()))
    a, b = _fit_power_law(valid_h, medians)
    return valid_h, medians, a, b


def _extract_hp_value(row: pd.Series, hp_col: str):
    """HP can be either a direct column or nested inside the params dict."""
    if hp_col in row.index and pd.notna(row[hp_col]):
        return row[hp_col]
    p = row.get("params")
    if isinstance(p, str):
        try:
            d = ast.literal_eval(p)
            return d.get(hp_col)
        except Exception:
            return None
    return None


def _aggregate_hp_heatmap(ds_key: str, hp_col: str) -> dict:
    """For per-series datasets, build a (series × horizon) heatmap of an HP.
    Uses exp2_<ds>_perseries when available, falls back to exp3_<ds>_gs1."""
    candidates = [f"exp2_{ds_key}_perseries", f"exp3_{ds_key}_gs1", f"exp8_{ds_key}_sgs1_pooled"]
    path = None
    for name in candidates:
        p = REPO_ROOT / "exps" / name / "local_results.csv"
        if p.exists():
            path = p
            print(f"  heatmap {ds_key} {hp_col}: loading {name}")
            break
    if path is None:
        return {}
    df = pd.read_csv(path)
    df["_max_h"] = df["horizon"].apply(_max_horizon)
    series_list = sorted(df["series"].astype(str).unique())
    horizons = sorted(int(h) for h in df["_max_h"].unique() if 24 <= h <= 720)
    matrix = []
    for s in series_list:
        row = []
        for h in horizons:
            sub = df[(df["series"].astype(str) == s) & (df["_max_h"] == h)]
            if sub.empty:
                row.append(None)
            else:
                v = _extract_hp_value(sub.iloc[0], hp_col)
                try:
                    fv = float(v)
                    row.append(fv if math.isfinite(fv) else None)
                except (TypeError, ValueError):
                    row.append(None)
        matrix.append(row)
    return {"series": series_list, "horizons": horizons, "values": matrix}




def _aggregate_horizon_pareto(ds_key: str) -> dict:
    """MSE degradation vs horizon group size, from exp3_<ds>_gs{1,2,...}."""
    group_sizes = [1, 2, 4, 6, 8, 12, 24, 48]
    # Column name in benchmark_comparison.csv is "Local MSE" (with a space) on
    # newer experiment runs; old runs use "Local_MSE". Accept either.
    candidates = ("Local MSE", "Local_MSE", "local_mse")
    points: list[tuple[int, float]] = []
    for gs in group_sizes:
        path = REPO_ROOT / "exps" / f"exp3_{ds_key}_gs{gs}" / "benchmark_comparison.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            continue
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue
        mean_mse = float(df[col].mean())
        points.append((gs, mean_mse))
    if not points:
        return {}
    best_mse = min(p[1] for p in points)
    return {
        "group_sizes": [p[0] for p in points],
        "mses": [p[1] for p in points],
        "degradation_pct": [(p[1] - best_mse) / best_mse * 100 for p in points],
        "best_mse": best_mse,
    }


def _aggregate_aug_selection(ds_key: str) -> dict:
    """Fraction of trials selecting each noise_type, and median σ when selected."""
    df = _load_results_with_fallback(ds_key, group_sizes=[1, 2, 4, 6, 8, 12, 24, 48])
    if df is None or "noise_type" not in df.columns:
        return {}
    total = len(df)
    counts = df["noise_type"].value_counts().to_dict()
    sel = {nt: counts.get(nt, 0) / total for nt in ("none", "time", "freq")}
    sigmas = {}
    for nt in ("time", "freq"):
        sub = df[df["noise_type"] == nt]
        if not sub.empty:
            sigmas[nt] = {
                "median": float(sub["aug_sigma"].median()),
                "q25": float(sub["aug_sigma"].quantile(0.25)),
                "q75": float(sub["aug_sigma"].quantile(0.75)),
            }
    return {"selection": sel, "sigmas": sigmas}


MAIN_RESULTS_TABLE = {
    "linear_models": ["Ours", "OLS", "FITS", "DLinear"],
    "nonlinear_models": ["PatchTST", "iTransformer", "TimeMixer", "TimesNet", "Autoformer"],
    "datasets": [
        {
            "name": "ETTm1",
            "rows": [
                {"H": 96,  "values": [0.297, 0.307, 0.309, 0.312, 0.329, 0.334, 0.320, 0.338, 0.505]},
                {"H": 192, "values": [0.332, 0.336, 0.338, 0.341, 0.367, 0.377, 0.361, 0.374, 0.553]},
                {"H": 336, "values": [0.357, 0.365, 0.367, 0.372, 0.401, 0.426, 0.408, 0.410, 0.621]},
                {"H": 720, "values": [0.396, 0.415, 0.417, 0.422, 0.456, 0.491, 0.469, 0.478, 0.671]},
                {"H": "Avg","values": [0.346, 0.356, 0.358, 0.362, 0.388, 0.407, 0.389, 0.400, 0.588]},
            ],
        },
        {
            "name": "ETTm2",
            "rows": [
                {"H": 96,  "values": [0.160, 0.162, 0.162, 0.163, 0.175, 0.180, 0.175, 0.187, 0.255]},
                {"H": 192, "values": [0.212, 0.216, 0.217, 0.217, 0.241, 0.250, 0.241, 0.249, 0.281]},
                {"H": 336, "values": [0.256, 0.268, 0.269, 0.269, 0.305, 0.311, 0.305, 0.321, 0.339]},
                {"H": 720, "values": [0.323, 0.349, 0.350, 0.354, 0.402, 0.412, 0.378, 0.408, 0.433]},
                {"H": "Avg","values": [0.237, 0.249, 0.250, 0.251, 0.281, 0.288, 0.275, 0.291, 0.327]},
            ],
        },
        {
            "name": "ETTh1",
            "rows": [
                {"H": 96,  "values": [0.369, 0.375, 0.377, 0.379, 0.414, 0.386, 0.375, 0.384, 0.449]},
                {"H": 192, "values": [0.400, 0.413, 0.413, 0.419, 0.460, 0.441, 0.405, 0.436, 0.500]},
                {"H": 336, "values": [0.423, 0.445, 0.432, 0.451, 0.501, 0.487, 0.439, 0.491, 0.521]},
                {"H": 720, "values": [0.430, 0.460, 0.428, 0.470, 0.507, 0.503, 0.469, 0.521, 0.514]},
                {"H": "Avg","values": [0.405, 0.423, 0.412, 0.430, 0.471, 0.454, 0.422, 0.458, 0.496]},
            ],
        },
        {
            "name": "ETTh2",
            "rows": [
                {"H": 96,  "values": [0.268, 0.270, 0.270, 0.275, 0.302, 0.297, 0.289, 0.340, 0.346]},
                {"H": 192, "values": [0.330, 0.331, 0.331, 0.342, 0.388, 0.380, 0.333, 0.402, 0.456]},
                {"H": 336, "values": [0.354, 0.354, 0.354, 0.359, 0.426, 0.428, 0.374, 0.452, 0.482]},
                {"H": 720, "values": [0.383, 0.380, 0.377, 0.384, 0.431, 0.427, 0.416, 0.462, 0.515]},
                {"H": "Avg","values": [0.334, 0.334, 0.333, 0.340, 0.387, 0.383, 0.353, 0.414, 0.450]},
            ],
        },
        {
            "name": "Electricity",
            "rows": [
                {"H": 96,  "values": [0.130, 0.133, 0.133, 0.134, 0.129, 0.148, 0.150, 0.168, 0.201]},
                {"H": 192, "values": [0.145, 0.148, 0.148, 0.149, 0.147, 0.162, 0.163, 0.184, 0.222]},
                {"H": 336, "values": [0.161, 0.164, 0.164, 0.165, 0.163, 0.178, 0.178, 0.198, 0.231]},
                {"H": 720, "values": [0.199, 0.203, 0.203, 0.205, 0.197, 0.225, 0.220, 0.220, 0.254]},
                {"H": "Avg","values": [0.159, 0.162, 0.162, 0.163, 0.159, 0.178, 0.178, 0.192, 0.227]},
            ],
        },
        {
            "name": "Traffic",
            "rows": [
                {"H": 96,  "values": [0.379, 0.385, 0.386, 0.387, 0.360, 0.395, 0.392, 0.593, 0.613]},
                {"H": 192, "values": [0.391, 0.397, 0.398, 0.399, 0.379, 0.417, 0.408, 0.617, 0.616]},
                {"H": 336, "values": [0.404, 0.410, 0.411, 0.412, 0.392, 0.433, 0.422, 0.629, 0.622]},
                {"H": 720, "values": [0.441, 0.448, 0.449, 0.450, 0.432, 0.467, 0.459, 0.640, 0.660]},
                {"H": "Avg","values": [0.404, 0.410, 0.411, 0.412, 0.391, 0.428, 0.420, 0.620, 0.628]},
            ],
        },
        {
            "name": "Weather",
            "rows": [
                {"H": 96,  "values": [0.141, 0.141, 0.142, 0.142, 0.149, 0.174, 0.164, 0.172, 0.266]},
                {"H": 192, "values": [0.184, 0.184, 0.185, 0.185, 0.194, 0.221, 0.215, 0.219, 0.307]},
                {"H": 336, "values": [0.234, 0.234, 0.236, 0.235, 0.245, 0.278, 0.272, 0.280, 0.359]},
                {"H": 720, "values": [0.304, 0.307, 0.307, 0.310, 0.314, 0.358, 0.351, 0.365, 0.419]},
                {"H": "Avg","values": [0.215, 0.217, 0.218, 0.218, 0.225, 0.258, 0.250, 0.259, 0.338]},
            ],
        },
        {
            "name": "Exchange",
            "rows": [
                {"H": 96,  "values": [0.081, 0.086, 0.087, 0.085, 0.088, 0.086, 0.086, 0.107, 0.197]},
                {"H": 192, "values": [0.167, 0.180, 0.183, 0.178, 0.176, 0.177, 0.177, 0.226, 0.300]},
                {"H": 336, "values": [0.305, 0.343, 0.344, 0.335, 0.301, 0.331, 0.338, 0.367, 0.509]},
                {"H": 720, "values": [0.811, 0.992, 0.965, 0.920, 0.901, 0.901, 0.920, 0.964, 1.447]},
                {"H": "Avg","values": [0.341, 0.400, 0.395, 0.380, 0.366, 0.374, 0.380, 0.416, 0.613]},
            ],
        },
    ],
}


def _load_benchmark_table() -> dict:
    """Render Exp10 (norm scope comparison) as a JSON-friendly table."""
    path = REPO_ROOT / "tables" / "exp10_norm_scope_comparison.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "dataset": str(r["dataset"]),
            "cutoff": int(r["cutoff"]),
            "global_mse": float(r["global_mse"]),
            "local_mse": float(r["local_mse"]),
            "search_mse": float(r["search_mse"]),
            "best_config": str(r["best_config"]),
        })
    return {"rows": rows}


def _load_forecast_predictions() -> dict:
    """Pull forecast comparison data (gt vs local vs global) for ETTh1 OT.
    Falls back to the first available exp3_etth1_* folder with predictions.npy."""
    candidates = [
        "exp3_etth1_gs48_sgs1_nt20",
        "exp3_etth1_gs1_sgs1_nt20",
        "exp9_etth1_sgs1_gs48_nt10",
    ]
    for name in candidates:
        path = REPO_ROOT / "exps" / name / "predictions.npy"
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=True).item()
        if not isinstance(d, dict):
            continue
        try:
            gt = np.asarray(d["gt"])
            lp = np.asarray(d["local_pred"])
            gp = np.asarray(d["global_pred"])
        except KeyError:
            continue
        # Pick OT series (index 6 for ETT). Subsample horizons to keep payload small.
        s = min(6, gt.shape[0] - 1)
        return {
            "dataset": "ETTh1",
            "series": "OT",
            "source": name,
            "gt": [round(float(v), 4) for v in gt[s]],
            "local_pred": [round(float(v), 4) for v in lp[s]],
            "global_pred": [round(float(v), 4) for v in gp[s]],
        }
    return {}


def build_figures_data() -> None:
    print(f"[4/4] Building figures_data.json")
    out: dict = {
        "dataset_keys": DATASET_KEYS,
        "dataset_labels": DATASET_LABELS,
        "dataset_colors": DATASET_COLORS,
        "lookback_curves": {},
        "hp_heatmaps": {},
        "horizon_pareto": {},
        "aug_selection": {},
        "forecast": _load_forecast_predictions(),
        "benchmark_table": _load_benchmark_table(),
        "main_results": MAIN_RESULTS_TABLE,
    }

    for ds in DATASET_KEYS:
        h, med, a, b = _aggregate_lookback_curve(ds)
        if h:
            out["lookback_curves"][ds] = {"horizons": h, "median_lookback": med, "a": a, "b": b}

    # HP heatmaps: ETTh1 + Weather (the dataset contrast featured in the paper)
    for ds in ("etth1", "weather"):
        for hp_col, key in (("local_ratio", "r"), ("best_alpha", "alpha")):
            data = _aggregate_hp_heatmap(ds, hp_col)
            if data:
                out["hp_heatmaps"][f"{ds}_{key}"] = data

    for ds in DATASET_KEYS:
        p = _aggregate_horizon_pareto(ds)
        if p:
            out["horizon_pareto"][ds] = p

    for ds in DATASET_KEYS:
        a = _aggregate_aug_selection(ds)
        if a:
            out["aug_selection"][ds] = a

    FIGURES_DST.parent.mkdir(parents=True, exist_ok=True)
    js_blob = json.dumps(out, separators=(",", ":"))
    FIGURES_DST.write_text(js_blob)
    FIGURES_JS.write_text("window.__FIGURES_DATA = " + js_blob + ";\n")
    size_kb = FIGURES_DST.stat().st_size / 1024
    print(f"  wrote {FIGURES_DST.relative_to(REPO_ROOT)} + .js ({size_kb:.0f} KB)")


def main() -> None:
    build_etth1_copy()
    build_etth1_manifest()
    build_playground_weights()
    build_widget_a_data()
    build_figures_data()
    print("done.")


if __name__ == "__main__":
    main()
