/* =====================================================================
 * Ridge solver + scalers + augmentation, ported from optuna_ridge.py
 * for browser use. Designed to run inside a Web Worker.
 *
 * Architecture
 * ------------
 *   - CPU low-level kernels (Float64): gemmAtA, gemmAtB, cholesky, choleskySolve,
 *     ridgePredict — written for correctness and used as the fallback.
 *   - A `Backend` object exposes `buildGrams(X, Y, N, L, H)` and
 *     `predict(X, beta, N, F, H)` and is swapped in/out by the worker at
 *     startup. The GPU backend uses TF.js (WebGPU first, WebGL fallback) and
 *     pays a one-time tensor-upload cost for huge speed-ups on the matmul-heavy
 *     paths. The CPU backend is the fallback.
 *   - `fitAndEvaluate(series, hp)` is async and uses whichever backend is
 *     active. Cholesky stays on CPU since it's small and benefits from Float64.
 *
 * Loaded as a classic script via importScripts() in the worker, so functions
 * are attached to the worker global scope.
 * ===================================================================== */

// ---------- generic numeric helpers ---------------------------------------

function _mean(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
}

function _std(x, mu) {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - mu;
    s += d * d;
  }
  return Math.sqrt(s / x.length + 1e-5);
}

function _randn() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ---------- train / val / test split --------------------------------------
//
// Mirrors optuna_ridge.py's protocol (around line 1339):
//   - If `protocol` is provided in `hp` it sets total_samples / n_train /
//     n_val / n_test explicitly (paper truncation for ETT, etc.).
//   - Otherwise a proportional 70/10/20 split is used.

function resolveProtocol(seriesLen, protocol) {
  if (protocol && protocol.total_samples) {
    const total = Math.min(protocol.total_samples, seriesLen);
    const nTrain = protocol.n_train | 0;
    const nVal   = protocol.n_val   | 0;
    const nTest  = protocol.n_test  | 0;
    return { total, nTrain, nVal, nTest,
             searchTrainRatio: protocol.search_train_ratio ?? 0.4,
             searchTestRatio:  protocol.search_test_ratio  ?? 0.2 };
  }
  const total = seriesLen;
  return {
    total,
    nTrain: Math.floor(total * 0.7),
    nVal:   Math.floor(total * 0.1),
    nTest:  total - Math.floor(total * 0.7) - Math.floor(total * 0.1),
    searchTrainRatio: 0.4,
    searchTestRatio:  0.2,
  };
}

// ---------- sliding-window construction -----------------------------------
//
// Returns (X, Y) for a horizon GROUP [groupMin..groupMax] (paper protocol at
// gs = groupMax - groupMin + 1). X is (N, lookback); Y is (N, groupSize).
//
// Window i covers samples [s, s + lookback) for X and points to
// series[s + lookback + groupMin - 1 .. s + lookback + groupMax - 1] for Y.

function buildWindowsGroup(series, lookback, groupMin, groupMax, segStart, segEnd, stride) {
  const groupSize = groupMax - groupMin + 1;
  const N = Math.max(0, Math.floor((segEnd - segStart - lookback - groupMax) / stride) + 1);
  const X = new Float64Array(N * lookback);
  const Y = new Float64Array(N * groupSize);
  let wi = 0;
  for (let s = segStart; s + lookback + groupMax <= segEnd; s += stride) {
    const xOff = wi * lookback;
    const yOff = wi * groupSize;
    for (let j = 0; j < lookback; j++) X[xOff + j] = series[s + j];
    for (let j = 0; j < groupSize; j++) Y[yOff + j] = series[s + lookback + groupMin - 1 + j];
    wi++;
  }
  return { X, Y, N, lookback, groupMin, groupMax, H: groupSize };
}

// Backwards compatible wrapper: predict the single value at step `horizon`.
function buildWindows(series, lookback, horizon, segStart, segEnd, stride) {
  const g = buildWindowsGroup(series, lookback, horizon, horizon, segStart, segEnd, stride);
  return { X: g.X, Y: g.Y, N: g.N, lookback, horizon, H: 1 };
}

// Build prediction-time X windows at specific anchors. anchors[i] is the
// position immediately AFTER the lookback context (the prediction-start),
// i.e. X[i] = series[anchors[i] - lookback .. anchors[i]).
function buildPredictWindows(series, lookback, anchors) {
  const N = anchors.length;
  const X = new Float64Array(N * lookback);
  for (let i = 0; i < N; i++) {
    const s = anchors[i] - lookback;
    const xOff = i * lookback;
    for (let j = 0; j < lookback; j++) X[xOff + j] = series[s + j];
  }
  return { X, N, lookback };
}

// ---------- scalers --------------------------------------------------------

function makeLocalScaler(method, lookback, lastK) {
  return {
    state: { centers: null, scales: null },
    fit(X, N) {
      const L = lookback;
      const k = lastK;
      const c = new Float64Array(N);
      const s = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const off = i * L + (L - k);
        let sum = 0;
        for (let j = 0; j < k; j++) sum += X[off + j];
        const mu = sum / k;
        c[i] = mu;
        let v = 0;
        for (let j = 0; j < k; j++) {
          const d = X[off + j] - mu;
          v += d * d;
        }
        s[i] = Math.sqrt(v / k + 1e-5);
      }
      this.state.centers = c;
      this.state.scales = s;
    },
    transformToFeatures(X, N) {
      const L = lookback;
      const F = L + 1;
      const out = new Float64Array(N * F);
      const c = this.state.centers, s = this.state.scales;
      for (let i = 0; i < N; i++) {
        const ci = c[i];
        for (let j = 0; j < L; j++) out[i * F + j] = X[i * L + j] - ci;
        out[i * F + L] = s[i];
      }
      return out;
    },
    transformTarget(Y, N, H) {
      const out = new Float64Array(N * H);
      const c = this.state.centers;
      for (let i = 0; i < N; i++) {
        const ci = c[i];
        for (let j = 0; j < H; j++) out[i * H + j] = Y[i * H + j] - ci;
      }
      return out;
    },
    invPredict(Ypred, N, H) {
      const c = this.state.centers;
      for (let i = 0; i < N; i++) {
        const ci = c[i];
        for (let j = 0; j < H; j++) Ypred[i * H + j] += ci;
      }
    },
  };
}

// ---------- CPU linear algebra --------------------------------------------

function cpu_gemmAtA(X, N, L, out) {
  // out (L×L) = X.T @ X
  out.fill(0);
  for (let n = 0; n < N; n++) {
    const off = n * L;
    for (let i = 0; i < L; i++) {
      const xi = X[off + i];
      if (xi === 0) continue;
      for (let j = i; j < L; j++) out[i * L + j] += xi * X[off + j];
    }
  }
  for (let i = 0; i < L; i++)
    for (let j = i + 1; j < L; j++) out[j * L + i] = out[i * L + j];
}

function cpu_gemmAtB(X, N, L, Y, H, out) {
  // out (L×H) = X.T @ Y
  out.fill(0);
  for (let n = 0; n < N; n++) {
    const xo = n * L;
    const yo = n * H;
    for (let i = 0; i < L; i++) {
      const xi = X[xo + i];
      if (xi === 0) continue;
      const baseOut = i * H;
      for (let j = 0; j < H; j++) out[baseOut + j] += xi * Y[yo + j];
    }
  }
}

function cpu_matMul(A, M, K, B, N, out) {
  // out (M×N) = A (M×K) @ B (K×N)
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const aik = A[i * K + k];
      if (aik === 0) continue;
      const bo = k * N;
      const oo = i * N;
      for (let j = 0; j < N; j++) out[oo + j] += aik * B[bo + j];
    }
  }
}

function cholesky(A, n) {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * n + j];
      for (let k = 0; k < j; k++) sum -= A[i * n + k] * A[j * n + k];
      if (i === j) {
        if (sum <= 0) return false;
        A[i * n + i] = Math.sqrt(sum);
      } else {
        A[i * n + j] = sum / A[j * n + j];
      }
    }
    for (let j = i + 1; j < n; j++) A[i * n + j] = 0;
  }
  return true;
}

function choleskySolve(L, n, B, m) {
  // Solve L L^T X = B in place on B. B is row-major (n × m).
  //
  // Vectorized over the m RHS columns: the outer loop is i (row), the inner
  // updates all m columns at once. This makes B accesses unit-stride
  // (B[k*m + col] for col=0..m-1) instead of stride-m as in a per-column
  // implementation. For the playground's m=720, this typically gives 3× on
  // F~1500 because the strided access in the naive version causes a cache
  // miss per inner iteration.
  //
  // L access is row-major in the forward solve (good). In the back-sub we
  // read column-i of L (stride n, cache-unfriendly), but it's a single
  // scalar load per inner k-step amortized over m B updates, so the cost is
  // dominated by the B traversal.

  // Forward: L y = B (in place).
  for (let i = 0; i < n; i++) {
    const iN = i * n;
    const iM = i * m;
    for (let k = 0; k < i; k++) {
      const Lik = L[iN + k];
      if (Lik === 0) continue;
      const kM = k * m;
      for (let col = 0; col < m; col++) B[iM + col] -= Lik * B[kM + col];
    }
    const inv = 1 / L[iN + i];
    for (let col = 0; col < m; col++) B[iM + col] *= inv;
  }
  // Back: L^T x = y (in place). L^T[i,k] = L[k,i] for k > i.
  for (let i = n - 1; i >= 0; i--) {
    const iM = i * m;
    for (let k = i + 1; k < n; k++) {
      const Lki = L[k * n + i];
      if (Lki === 0) continue;
      const kM = k * m;
      for (let col = 0; col < m; col++) B[iM + col] -= Lki * B[kM + col];
    }
    const inv = 1 / L[i * n + i];
    for (let col = 0; col < m; col++) B[iM + col] *= inv;
  }
}

// ---------- Backend abstraction -------------------------------------------
// The worker may swap in a GPU-accelerated backend at startup. Each backend
// is async so GPU upload/download is awaitable; CPU returns resolved promises.

const CpuBackend = {
  name: "cpu",
  async buildGrams(X, Y, N, L, H) {
    const XTX = new Float64Array(L * L);
    const XTY = new Float64Array(L * H);
    cpu_gemmAtA(X, N, L, XTX);
    cpu_gemmAtB(X, N, L, Y, H, XTY);
    return { XTX, XTY };
  },
  async predict(X, beta, N, F, H) {
    const out = new Float64Array(N * H);
    cpu_matMul(X, N, F, beta, H, out);
    return out;
  },
};

let _activeBackend = CpuBackend;
function setBackend(b) { _activeBackend = b || CpuBackend; }
function getBackend() { return _activeBackend; }

// ---------- top-level fit ------------------------------------------------

async function _solveRidge(featureMatrix, F, N, Y, H, alpha, regBias) {
  // (X^T X + α I) β = X^T Y. Gram build runs on the active backend; the
  // Cholesky and back-sub stay on CPU in Float64 for numerical stability.
  const t0 = (typeof performance !== "undefined") ? performance.now() : Date.now();
  const { XTX, XTY } = await _activeBackend.buildGrams(featureMatrix, Y, N, F, H);
  const tGram = ((typeof performance !== "undefined") ? performance.now() : Date.now()) - t0;
  // Promote XTX/XTY to Float64 if backend returned Float32 (TF.js)
  const A = XTX instanceof Float64Array ? XTX : new Float64Array(XTX);
  const B = XTY instanceof Float64Array ? XTY : new Float64Array(XTY);
  for (let i = 0; i < F; i++) {
    if (i === regBias) continue;
    A[i * F + i] += alpha;
  }
  const tChol0 = (typeof performance !== "undefined") ? performance.now() : Date.now();
  if (!cholesky(A, F)) {
    for (let i = 0; i < F; i++) A[i * F + i] += 1e-3;
    cholesky(A, F);
  }
  choleskySolve(A, F, B, H);
  const tChol = ((typeof performance !== "undefined") ? performance.now() : Date.now()) - tChol0;
  if (typeof console !== "undefined" && console.log)
    console.log(`[ridge] F=${F} N=${N} H=${H} | gram ${tGram.toFixed(0)}ms (${_activeBackend.name}) | chol+solve ${tChol.toFixed(0)}ms (cpu)`);
  return B; // (F×H) β
}

async function _predict(X, beta, N, F, H) {
  const out = await _activeBackend.predict(X, beta, N, F, H);
  return out instanceof Float64Array ? out : new Float64Array(out);
}

async function fitAndEvaluate(series, hp, opts) {
  // `opts.mode` selects the split protocol (matches optuna_ridge.py):
  //   'fit'     — refit on train only ([0, nTrain)), evaluate on test
  //               ([nTrain+nVal, total)). This matches refit_test's default
  //               (use_train_val=False at optuna_ridge.py:966) — the val
  //               segment is unused in the refit; only the search HPs touched it.
  //   'search'  — HP-search mode: train on [0, searchTrain), evaluate on
  //               [searchTrain, searchVal). The held-out test stays untouched.
  const mode = (opts && opts.mode) || "fit";
  const {
    lookback, horizon, alpha,
    scaler_scope, scaler_method, noise_type, aug_sigma, local_ratio,
  } = hp;

  const stride = 1;
  const proto = resolveProtocol(series.length, hp.protocol);
  const total = proto.total;
  const nTrain = proto.nTrain;
  const nVal = proto.nVal;
  const trainvalEnd = nTrain + nVal;
  const searchTrainEnd = Math.floor(total * proto.searchTrainRatio);
  const searchValEnd   = Math.floor(total * (1.0 - proto.searchTestRatio));

  // Pick fit / eval boundaries based on mode.
  let trainStart, trainEnd, evalStart, evalEnd;
  if (mode === "search") {
    trainStart = 0;
    trainEnd   = searchTrainEnd;
    evalStart  = searchTrainEnd;
    evalEnd    = searchValEnd;
  } else {
    // 'fit' mode: train on [0, n_train), test on [n_train+n_val, total).
    // Val is intentionally unused (matches refit_test default).
    trainStart = 0;
    trainEnd   = nTrain;
    evalStart  = trainvalEnd;
    evalEnd    = total;
  }

  // Global Z-score with training stats only (paper protocol fits the global
  // scaler on the n_train portion in both modes — search and fit).
  const trainStatsEnd = Math.min(nTrain, trainEnd);
  const trainSegmentArr = new Float64Array(trainStatsEnd);
  for (let i = 0; i < trainStatsEnd; i++) trainSegmentArr[i] = series[i];
  const trainMean = _mean(trainSegmentArr);
  const trainStd = _std(trainSegmentArr, trainMean);
  const normedSeries = new Float64Array(total);
  for (let i = 0; i < total; i++) normedSeries[i] = (series[i] - trainMean) / trainStd;

  const trainW = buildWindows(normedSeries, lookback, horizon, trainStart, trainEnd, stride);
  // Validation/eval windows. We include the lookback overlap so windows
  // straddling the boundary still get the correct historical context.
  const evalW  = buildWindows(normedSeries, lookback, horizon, Math.max(0, evalStart - lookback), evalEnd, 1);
  // testW is the actual held-out test set ([trainvalEnd, total)) for the
  // "test_mse" metric reported in the playground, independent of mode so the
  // user always sees the same out-of-sample number.
  const testW  = mode === "fit"
    ? evalW
    : buildWindows(normedSeries, lookback, horizon,
                   Math.max(0, trainvalEnd - lookback), total, 1);

  const H = trainW.H;  // = 1 (paper protocol)

  const useLocal = scaler_scope === "local";
  let beta, featureF;
  let localScaler = null;

  if (useLocal) {
    const lastK = Math.max(1, Math.min(lookback, Math.round(lookback * local_ratio)));
    localScaler = makeLocalScaler(scaler_method, lookback, lastK);
    localScaler.fit(trainW.X, trainW.N);
    const Xfeat = localScaler.transformToFeatures(trainW.X, trainW.N);
    const Ytarg = localScaler.transformTarget(trainW.Y, trainW.N, H);
    if (noise_type === "time" && aug_sigma > 0) {
      const F = lookback + 1;
      for (let n = 0; n < trainW.N; n++) {
        const off = n * F;
        for (let j = 0; j < lookback; j++) Xfeat[off + j] += _randn() * aug_sigma;
      }
    }
    featureF = lookback + 1;
    beta = await _solveRidge(Xfeat, featureF, trainW.N, Ytarg, H, alpha, -1);
  } else {
    featureF = lookback + 1;
    const Xfeat = new Float64Array(trainW.N * featureF);
    const noisy = (noise_type === "time" && aug_sigma > 0);
    for (let n = 0; n < trainW.N; n++) {
      Xfeat[n * featureF] = 1.0;
      for (let j = 0; j < lookback; j++) {
        Xfeat[n * featureF + 1 + j] = trainW.X[n * lookback + j] + (noisy ? _randn() * aug_sigma : 0);
      }
    }
    beta = await _solveRidge(Xfeat, featureF, trainW.N, trainW.Y, H, alpha, 0);
  }

  async function predictSet(set) {
    if (useLocal) {
      localScaler.fit(set.X, set.N);
      const Xfeat = localScaler.transformToFeatures(set.X, set.N);
      const Yhat = await _predict(Xfeat, beta, set.N, featureF, H);
      localScaler.invPredict(Yhat, set.N, H);
      return Yhat;
    }
    const F = featureF;
    const Xfeat = new Float64Array(set.N * F);
    for (let n = 0; n < set.N; n++) {
      Xfeat[n * F] = 1.0;
      for (let j = 0; j < lookback; j++) Xfeat[n * F + 1 + j] = set.X[n * lookback + j];
    }
    return await _predict(Xfeat, beta, set.N, F, H);
  }

  // In 'fit' mode evalW === testW, so reuse the prediction.
  const yPredEval = await predictSet(evalW);
  const yPredTest = mode === "fit" ? yPredEval : await predictSet(testW);

  function mse(yhat, ytrue, N) {
    let s = 0;
    for (let i = 0; i < N; i++) {
      const d = yhat[i] - ytrue[i];
      s += d * d;
    }
    return s / Math.max(N, 1);
  }
  function mae(yhat, ytrue, N) {
    let s = 0;
    for (let i = 0; i < N; i++) s += Math.abs(yhat[i] - ytrue[i]);
    return s / Math.max(N, 1);
  }
  const valMse  = mse(yPredEval, evalW.Y, evalW.N);
  const testMse = mse(yPredTest, testW.Y, testW.N);
  const testMae = mae(yPredTest, testW.Y, testW.N);

  // Example trajectory for the playground plot. We show the history right
  // before the test boundary, then the SINGLE prediction / truth at step H.
  const lastIdx = testW.N - 1;
  const histStart = (evalEnd - lookback - horizon);
  const history = Array.from(normedSeries.slice(Math.max(0, histStart), histStart + lookback));
  // Surrounding true values for context (so the user can see the trajectory).
  const truthContext = Array.from(normedSeries.slice(histStart + lookback, histStart + lookback + horizon));
  const truthAtH  = normedSeries[histStart + lookback + horizon - 1];
  const predAtH   = yPredTest[lastIdx];

  return {
    metrics: { val_mse: valMse, test_mse: testMse, test_mae: testMae },
    example: {
      history,
      truth_context: truthContext,
      truth_at_h: truthAtH,
      pred_at_h: predAtH,
      lookback, horizon, anchor: history.length,
    },
    n_train_windows: trainW.N,
    n_test_windows: testW.N,
    stride,
    backend: _activeBackend.name,
    mode,
    train_stats: { mean: trainMean, std: trainStd },
    split: { trainEnd, evalStart, evalEnd, total },
  };
}

// ---------- multi-horizon visualization fit ------------------------------
//
// Trains one Ridge per `perGroupHps` entry (multi-output over the group's
// horizons), plus one global-baseline Ridge with multi-output Y of length
// max_horizon. Predicts on a common anchor range. Returns concatenated
// "ours" predictions (length 720), the baseline trajectory (length 720),
// truth, history, per-cutoff MSEs, and the actively used HPs per group so
// the UI can snap sliders to the searched values.

async function _fitOneRidge(series, lookback, groupMin, groupMax, alpha,
                            scalerScope, scalerMethod, noiseType, augSigma, localRatio,
                            trainStart, trainEnd, anchors) {
  // Fit one (group, lookback) Ridge and predict at the test anchors.
  // Returns { pred, beta, F, H }: `pred` for the SE / trajectory pipeline,
  // `beta` (F×H Float32 row-major) so the offline bake can persist weights
  // that the browser can later apply to a live window without refitting.
  const gsize = groupMax - groupMin + 1;
  const trainW = buildWindowsGroup(series, lookback, groupMin, groupMax, trainStart, trainEnd, 1);
  if (trainW.N <= 0) {
    return {
      pred: new Float64Array(anchors.length * gsize),
      beta: null,
      F: 0,
      H: gsize,
    };
  }
  const useLocal = scalerScope === "local";
  const H = trainW.H;
  let beta;
  let featureF;
  let localScaler = null;

  if (useLocal) {
    const lastK = Math.max(1, Math.min(lookback, Math.round(lookback * localRatio)));
    localScaler = makeLocalScaler(scalerMethod, lookback, lastK);
    localScaler.fit(trainW.X, trainW.N);
    const Xfeat = localScaler.transformToFeatures(trainW.X, trainW.N);
    const Ytarg = localScaler.transformTarget(trainW.Y, trainW.N, H);
    if (noiseType === "time" && augSigma > 0) {
      const F = lookback + 1;
      for (let n = 0; n < trainW.N; n++) {
        const off = n * F;
        for (let j = 0; j < lookback; j++) Xfeat[off + j] += _randn() * augSigma;
      }
    }
    featureF = lookback + 1;
    beta = await _solveRidge(Xfeat, featureF, trainW.N, Ytarg, H, alpha, -1);
  } else {
    featureF = lookback + 1;
    const Xfeat = new Float64Array(trainW.N * featureF);
    const noisy = noiseType === "time" && augSigma > 0;
    for (let n = 0; n < trainW.N; n++) {
      Xfeat[n * featureF] = 1.0;
      for (let j = 0; j < lookback; j++) {
        Xfeat[n * featureF + 1 + j] = trainW.X[n * lookback + j] + (noisy ? _randn() * augSigma : 0);
      }
    }
    beta = await _solveRidge(Xfeat, featureF, trainW.N, trainW.Y, H, alpha, 0);
  }

  // Predict at the requested anchors.
  let pred;
  const predW = buildPredictWindows(series, lookback, anchors);
  if (useLocal) {
    localScaler.fit(predW.X, predW.N);
    const Xfeat = localScaler.transformToFeatures(predW.X, predW.N);
    pred = await _predict(Xfeat, beta, predW.N, featureF, H);
    localScaler.invPredict(pred, predW.N, H);
  } else {
    const F = featureF;
    const Xfeat = new Float64Array(predW.N * F);
    for (let n = 0; n < predW.N; n++) {
      Xfeat[n * F] = 1.0;
      for (let j = 0; j < lookback; j++) Xfeat[n * F + 1 + j] = predW.X[n * lookback + j];
    }
    pred = await _predict(Xfeat, beta, predW.N, F, H);
  }

  // Ensure beta is contiguous Float32 (row-major F×H) so the browser can
  // memmap it from weights.bin without conversion. _solveRidge returns
  // Float64; pack it down for storage. Local copy keeps the runtime path
  // unaffected for the prediction we just computed.
  let betaF32 = null;
  if (beta) {
    betaF32 = new Float32Array(featureF * H);
    for (let i = 0; i < featureF * H; i++) betaF32[i] = beta[i];
  }
  return { pred, beta: betaF32, F: featureF, H };
}


async function fitForVisualization(series, perGroupHps, baselineHp, opts) {
  const proto = resolveProtocol(series.length, (opts && opts.protocol) || null);
  const total = proto.total;
  const nTrain = proto.nTrain;
  const trainvalEnd = nTrain + proto.nVal;
  const skipBaseline = !!(opts && opts.skipBaseline);

  const maxH = Math.max.apply(null, perGroupHps.map((g) => g.group_max));
  // baselineHp may be null when opts.skipBaseline is set — callers in that
  // case supply the baseline from baked weights and we don't need to refit.
  const baselineL = (!skipBaseline && baselineHp) ? baselineHp.lookback : 0;
  const groupMaxL = Math.max.apply(null, perGroupHps.map((g) => g.lookback));
  const maxL = Math.max(baselineL, groupMaxL);

  // Common anchor range: every anchor must allow each group AND the baseline
  // (if requested) to read `maxL` of history and `maxH` of future from the
  // *test* segment.
  const anchorStart = Math.max(trainvalEnd, maxL);
  const anchorEnd = total - maxH;
  const nAnchors = Math.max(1, anchorEnd - anchorStart + 1);
  const anchors = new Int32Array(nAnchors);
  for (let i = 0; i < nAnchors; i++) anchors[i] = anchorStart + i;

  // Global Z-score on the n_train segment.
  const trainSeg = new Float64Array(nTrain);
  for (let i = 0; i < nTrain; i++) trainSeg[i] = series[i];
  const trainMean = _mean(trainSeg);
  const trainStd = _std(trainSeg, trainMean);
  const normed = new Float64Array(total);
  for (let i = 0; i < total; i++) normed[i] = (series[i] - trainMean) / trainStd;

  // Per-group ours predictions.
  const groupPreds = [];
  for (const g of perGroupHps) {
    const { pred } = await _fitOneRidge(
      normed, g.lookback, g.group_min, g.group_max, g.alpha,
      g.scaler_scope || "local", g.scaler_method || "mean",
      g.noise_type || "none", g.aug_sigma || 0.0, g.local_ratio || 1.0,
      0, nTrain, anchors,
    );
    groupPreds.push({ groupMin: g.group_min, groupMax: g.group_max, pred, hp: g });
  }

  // Global baseline: one multi-output Ridge with L=720, α=1e-5, scope=global.
  // Skipped when opts.skipBaseline is set — the caller fills basePred from
  // the baked weights, since the baseline's HPs are fixed and never change.
  let basePred = null;
  if (!skipBaseline && baselineHp) {
    const r = await _fitOneRidge(
      normed, baselineHp.lookback, 1, maxH, baselineHp.alpha,
      baselineHp.scaler_scope || "global", baselineHp.scaler_method || "mean",
      "none", 0.0, baselineHp.local_ratio || 1.0,
      0, nTrain, anchors,
    );
    basePred = r.pred;
  }

  // Concatenate per-anchor "ours" predictions across groups in order.
  const ours = new Float64Array(nAnchors * maxH);
  for (let i = 0; i < nAnchors; i++) {
    let offset = 0;
    for (const gp of groupPreds) {
      const gSize = gp.groupMax - gp.groupMin + 1;
      const baseSrc = i * gSize;
      const baseDst = i * maxH + offset;
      for (let j = 0; j < gSize; j++) ours[baseDst + j] = gp.pred[baseSrc + j];
      offset += gSize;
    }
  }

  // Truth at each anchor.
  const truth = new Float64Array(nAnchors * maxH);
  for (let i = 0; i < nAnchors; i++) {
    const s = anchors[i];
    for (let j = 0; j < maxH; j++) truth[i * maxH + j] = normed[s + j];
  }

  // Per-step squared-error sums (over all test anchors). The caller can
  // compute MSE-up-to-H as cumsum(oursSe[:H]) / (nAnchors * H) for any H,
  // which matches the paper's "average over all test windows" protocol.
  const oursSe = new Float64Array(maxH);
  const baseSe = basePred ? new Float64Array(maxH) : null;
  for (let i = 0; i < nAnchors; i++) {
    const baseRow = i * maxH;
    for (let j = 0; j < maxH; j++) {
      const d1 = ours[baseRow + j] - truth[baseRow + j];
      oursSe[j] += d1 * d1;
      if (basePred) {
        const d2 = basePred[baseRow + j] - truth[baseRow + j];
        baseSe[j] += d2 * d2;
      }
    }
  }
  // Legacy MSE-at-cutoff (kept for back-compat).
  const CUTOFFS = [96, 192, 336, 720];
  const mseAtCutoff = {};
  const baseMseAtCutoff = {};
  for (const H of CUTOFFS) {
    if (H > maxH) continue;
    let sOur = 0, sBase = 0;
    for (let j = 0; j < H; j++) {
      sOur += oursSe[j];
      if (baseSe) sBase += baseSe[j];
    }
    mseAtCutoff[H] = sOur / (nAnchors * H);
    if (baseSe) baseMseAtCutoff[H] = sBase / (nAnchors * H);
  }

  // Take the LAST anchor as the "representative" trajectory for plotting.
  const lastIdx = nAnchors - 1;
  const lastAnchor = anchors[lastIdx];
  return {
    anchor: lastAnchor,
    maxL,
    maxH,
    history: Array.from(normed.slice(Math.max(0, lastAnchor - maxL), lastAnchor)),
    truth:   Array.from(truth.subarray(lastIdx * maxH, (lastIdx + 1) * maxH)),
    ourPred: Array.from(ours.subarray(lastIdx * maxH, (lastIdx + 1) * maxH)),
    // Placeholder zeros when the caller skipped the baseline fit — caller
    // is expected to overwrite this with baked predictions before display.
    basePred: basePred
      ? Array.from(basePred.subarray(lastIdx * maxH, (lastIdx + 1) * maxH))
      : new Array(maxH).fill(0),
    mseAtCutoff,
    baseMseAtCutoff,
    oursSe: Array.from(oursSe),
    baseSe: baseSe ? Array.from(baseSe) : null,
    nAnchorsForSe: nAnchors,
    n_test_anchors: nAnchors,
    perGroup: groupPreds.map((g) => ({ groupMin: g.groupMin, groupMax: g.groupMax, hp: g.hp })),
    backend: _activeBackend.name,
    train_stats: { mean: trainMean, std: trainStd },
  };
}

// ---------- offline bake (Node) ------------------------------------------
//
// Runs the same fit pipeline as fitForVisualization, but instead of building
// a render-ready trajectory, surfaces the Ridge β matrices + per-step SE
// over all test anchors. The bake script (website/bake_weights.mjs) writes
// β into weights.bin and SE into the manifest; the browser then applies β
// to the active window at render time. By going through the same
// _fitOneRidge as the worker, the baked outputs are bit-exact with what a
// live refit would produce.
async function bakeWeights(series, perGroupHps, baselineHp, opts) {
  const proto = resolveProtocol(series.length, (opts && opts.protocol) || null);
  const total = proto.total;
  const nTrain = proto.nTrain;
  const trainvalEnd = nTrain + proto.nVal;

  const maxH = Math.max.apply(null, perGroupHps.map((g) => g.group_max));
  const baselineL = baselineHp ? baselineHp.lookback : 0;
  const groupMaxL = Math.max.apply(null, perGroupHps.map((g) => g.lookback));
  const maxL = Math.max(baselineL, groupMaxL);

  const anchorStart = Math.max(trainvalEnd, maxL);
  const anchorEnd = total - maxH;
  const nAnchors = Math.max(1, anchorEnd - anchorStart + 1);
  const anchors = new Int32Array(nAnchors);
  for (let i = 0; i < nAnchors; i++) anchors[i] = anchorStart + i;

  // Global z-score on the n_train segment (matches fitForVisualization).
  const trainSeg = new Float64Array(nTrain);
  for (let i = 0; i < nTrain; i++) trainSeg[i] = series[i];
  const trainMean = _mean(trainSeg);
  const trainStd = _std(trainSeg, trainMean);
  const normed = new Float64Array(total);
  for (let i = 0; i < total; i++) normed[i] = (series[i] - trainMean) / trainStd;

  // Per-group β + per-anchor predictions.
  const groups = [];
  for (const g of perGroupHps) {
    const r = await _fitOneRidge(
      normed, g.lookback, g.group_min, g.group_max, g.alpha,
      g.scaler_scope || "local", g.scaler_method || "mean",
      g.noise_type || "none", g.aug_sigma || 0.0, g.local_ratio || 1.0,
      0, nTrain, anchors,
    );
    groups.push({ hp: g, beta: r.beta, F: r.F, H: r.H, pred: r.pred });
  }

  // Baseline β + per-anchor predictions.
  let baseline = null;
  if (baselineHp) {
    const r = await _fitOneRidge(
      normed, baselineHp.lookback, 1, maxH, baselineHp.alpha,
      baselineHp.scaler_scope || "global", baselineHp.scaler_method || "mean",
      "none", 0.0, baselineHp.local_ratio || 1.0,
      0, nTrain, anchors,
    );
    baseline = { hp: baselineHp, beta: r.beta, F: r.F, H: r.H, pred: r.pred };
  }

  // Concatenate "ours" per-anchor predictions across H-groups.
  const ours = new Float64Array(nAnchors * maxH);
  for (let i = 0; i < nAnchors; i++) {
    let offset = 0;
    for (const gp of groups) {
      const gSize = gp.hp.group_max - gp.hp.group_min + 1;
      const baseSrc = i * gSize;
      const baseDst = i * maxH + offset;
      for (let j = 0; j < gSize; j++) ours[baseDst + j] = gp.pred[baseSrc + j];
      offset += gSize;
    }
  }
  // Truth at each anchor.
  const truth = new Float64Array(nAnchors * maxH);
  for (let i = 0; i < nAnchors; i++) {
    const s = anchors[i];
    for (let j = 0; j < maxH; j++) truth[i * maxH + j] = normed[s + j];
  }
  // Per-step squared-error sums.
  const oursSe = new Float64Array(maxH);
  const baseSe = baseline ? new Float64Array(maxH) : null;
  for (let i = 0; i < nAnchors; i++) {
    const baseRow = i * maxH;
    for (let j = 0; j < maxH; j++) {
      const d1 = ours[baseRow + j] - truth[baseRow + j];
      oursSe[j] += d1 * d1;
      if (baseline) {
        const d2 = baseline.pred[baseRow + j] - truth[baseRow + j];
        baseSe[j] += d2 * d2;
      }
    }
  }

  return {
    groups: groups.map((g) => ({ hp: g.hp, beta: g.beta, F: g.F, H: g.H })),
    baseline: baseline
      ? { hp: baseline.hp, beta: baseline.beta, F: baseline.F, H: baseline.H }
      : null,
    oursSe: Array.from(oursSe),
    baseSe: baseSe ? Array.from(baseSe) : null,
    n_test_anchors: nAnchors,
    test_anchor: anchorEnd,
    train_stats: { mean: trainMean, std: trainStd },
  };
}

// Expose the public API. In a worker `self` is the worker scope; in Node
// (the offline bake script in website/bake_weights.mjs) we run inside a
// `new Function("self", ...)` wrapper so `self` is the sandbox object the
// bake script holds onto. Same line works in both contexts.
self.RidgeLib = {
  fitAndEvaluate,
  fitForVisualization,
  bakeWeights,
  setBackend,
  getBackend,
  CpuBackend,
};
