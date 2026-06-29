/* Playground orchestrator: loads ETTh1, manages controls, dispatches fits to
 * the worker, and renders the time-series plot with the lookback span bar. */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  manifest: null,
  // Active dataset object: { key, label, series: {colName: Float64Array}, columns: [] }
  dataset: null,
  activeDataset: null,
  activeSeries: null,
  hp: {
    lookback: 96,
    horizon: 192,
    alpha: 1.0,
    scaler_scope: "local",
    scaler_method: "mean",
    noise_type: "none",
    aug_sigma: 0.0,
    local_ratio: 0.1,
  },
  // User overrides for L/α (null = use the searched-optimum value for the
  // current horizon group). Reset whenever series or dataset changes.
  override: { lookback: null, alpha: null },
  programmaticSliderSet: false,
  worker: null,
  jobId: 0,
  baselineCache: new Map(),
  baselineComputing: new Set(),
  lastFitResult: null,
  backend: null,
  // Prefit cache keyed by `${source}|${series}|${L|alpha override sig}`.
  prefitCache: new Map(),
  prefit: null,
  // ArrayBuffer of pre-fit Ridge β for each dataset's baked default series.
  // Loaded from app/data/weights.bin (produced by website/bake_weights.mjs).
  // When present, the JS applies these β to the active window to render the
  // default state instantly; falls back to a live worker refit otherwise.
  weightsBin: null,
  // Autotune ON: refit using the per-horizon-group searched-optimum HPs from
  // the manifest (one ridge per group; matches the paper protocol).
  // Autotune OFF: refit with the user's slider L/α as a single 1..720 ridge
  // (exploration mode).
  autotune: true,
};

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

function newWorker() {
  // Classic worker — importScripts() loads tf.min.js and ridge.js at startup,
  // probes for a GPU backend (WebGPU, then WebGL), and falls back to CPU.
  const w = new Worker("ridge.worker.js");
  w.addEventListener("message", (ev) => {
    if (ev.data && ev.data.type === "backend-ready") {
      state.backend = ev.data.backend;
      updateBackendBadge();
    }
  });
  return w;
}

function updateBackendBadge() {
  const el = document.getElementById("backend-badge");
  if (!el) return;
  const map = {
    "webgpu": { label: "WebGPU", cls: "ok" },
    "webgl":  { label: "WebGL",  cls: "ok-mid" },
    "cpu":    { label: "CPU (no GPU)", cls: "warn" },
  };
  const info = map[state.backend] || { label: state.backend || "…", cls: "" };
  el.textContent = info.label;
  el.className = "backend-badge " + info.cls;
}

function postFit(series, hp, mode) {
  return new Promise((resolve, reject) => {
    const id = ++state.jobId;
    const onMsg = (ev) => {
      if (ev.data.id !== id) return;
      if (ev.data.type === "fit-result") {
        state.worker.removeEventListener("message", onMsg);
        resolve(ev.data);
      } else if (ev.data.type === "error") {
        state.worker.removeEventListener("message", onMsg);
        reject(new Error(ev.data.error));
      }
    };
    state.worker.addEventListener("message", onMsg);
    state.worker.postMessage({
      type: "fit",
      id,
      payload: { series: Array.from(series), hp, mode: mode || "fit" },
    });
  });
}

function postFitVis(series, perGroupHps, baselineHp, protocol, opts) {
  return new Promise((resolve, reject) => {
    const id = ++state.jobId;
    const onMsg = (ev) => {
      if (ev.data.id !== id) return;
      if (ev.data.type === "fitVis-result") {
        state.worker.removeEventListener("message", onMsg);
        resolve(ev.data);
      } else if (ev.data.type === "error") {
        state.worker.removeEventListener("message", onMsg);
        reject(new Error(ev.data.error));
      }
    };
    state.worker.addEventListener("message", onMsg);
    state.worker.postMessage({
      type: "fitVis", id,
      payload: {
        series: Array.from(series), perGroupHps, baselineHp, protocol,
        skipBaseline: !!(opts && opts.skipBaseline),
      },
    });
  });
}

function postAutoSearch(series, baseHp, space, trials, onProgress) {
  return new Promise((resolve, reject) => {
    const id = ++state.jobId;
    const onMsg = (ev) => {
      if (ev.data.id !== id) return;
      if (ev.data.type === "autoSearch-progress") {
        if (onProgress) onProgress(ev.data);
      } else if (ev.data.type === "autoSearch-result") {
        state.worker.removeEventListener("message", onMsg);
        resolve(ev.data);
      } else if (ev.data.type === "error") {
        state.worker.removeEventListener("message", onMsg);
        reject(new Error(ev.data.error));
      }
    };
    state.worker.addEventListener("message", onMsg);
    state.worker.postMessage({
      type: "autoSearch",
      id,
      // Trials run in 'search' mode (search_train + search_val window),
      // matching optuna_ridge.py's HP-search protocol.
      payload: { series: Array.from(series), baseHp, space, trials, mode: "search" },
    });
  });
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function loadManifestAndWeights() {
  const manifest = await fetch("data/manifest.json").then((r) => r.json());
  state.manifest = manifest;
  // Best-effort β fetch. weights.bin is produced by `node website/bake_weights.mjs`
  // and holds β only for each dataset's baked default series. When absent or
  // when the active state has no baked entry, the JS falls back to a live
  // worker refit. Loading is non-blocking so the page can render context
  // immediately; tryInitialRender() runs again once the buffer arrives.
  if (manifest.precomputed && manifest.precomputed.weights_bin) {
    fetch("data/" + manifest.precomputed.weights_bin)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((buf) => {
        if (buf) {
          state.weightsBin = buf;
          tryInitialRender();
        }
      })
      .catch((e) => console.warn("weights fetch failed:", e));
  }
}

async function loadDataset(key) {
  // Load the CSV for the chosen dataset and populate state.dataset. Manifest
  // (with precomputed weights & SE) is shared and stays loaded.
  const pc = state.manifest && state.manifest.precomputed;
  const entry = pc && pc.datasets && pc.datasets[key];
  if (!entry) throw new Error("Unknown dataset: " + key);
  setStatus(`loading ${entry.label}…`);
  const csvText = await fetch("data/" + entry.csv_path).then((r) => r.text());
  const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
  const fields = parsed.meta.fields.filter((f) => f !== "date");
  const series = {};
  for (const col of fields) {
    const arr = new Float64Array(parsed.data.length);
    for (let i = 0; i < parsed.data.length; i++) arr[i] = parseFloat(parsed.data[i][col]);
    series[col] = arr;
  }
  state.activeDataset = key;
  state.dataset = { key, label: entry.label, columns: fields, series, source: "bundled" };
  // Default series: prefer "OT" if present, else first.
  const defaultSeries = fields.indexOf("OT") >= 0 ? "OT" : fields[0];
  state.activeSeries = defaultSeries;
  populateSeriesDropdown(fields, defaultSeries);
  setStatus(`${entry.label} loaded · ${parsed.data.length.toLocaleString()} rows · ${fields.length} series`);
}

function populateDatasetDropdown() {
  const pc = state.manifest && state.manifest.precomputed;
  if (!pc || !pc.datasets) return;
  const sel = document.getElementById("dataset-select");
  sel.innerHTML = "";
  for (const key of Object.keys(pc.datasets)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = pc.datasets[key].label;
    if (key === (state.activeDataset || pc.default)) opt.selected = true;
    sel.appendChild(opt);
  }
}

function setStatus(msg, sticky) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.classList.add("visible");
  if (!sticky) {
    clearTimeout(setStatus._t);
    setStatus._t = setTimeout(() => el.classList.remove("visible"), 1400);
  }
  const ds = document.getElementById("data-status");
  if (ds) ds.textContent = msg;
}

function populateSeriesDropdown(cols, activeCol) {
  const sel = document.getElementById("series-select");
  sel.innerHTML = "";
  for (const c of cols) {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = c;
    if (c === activeCol) o.selected = true;
    sel.appendChild(o);
  }
}

// ---------------------------------------------------------------------------
// Controls / refresh loop
// ---------------------------------------------------------------------------

function readHpFromUI() {
  state.hp.lookback = parseInt(document.getElementById("lookback-slider").value, 10);
  state.hp.horizon = parseInt(document.getElementById("horizon-slider").value, 10);
  state.hp.alpha = Math.pow(10, parseFloat(document.getElementById("alpha-slider").value));
  // Non-tunable HPs follow the searched optimum for the current horizon group
  // (filled in by snapPinnedHps before each fit).
}

function writeHpToUI(hp) {
  state.programmaticSliderSet = true;
  try {
    if (hp.lookback != null)
      document.getElementById("lookback-slider").value = Math.max(32, Math.min(2048, Math.round(hp.lookback)));
    if (hp.horizon != null) document.getElementById("horizon-slider").value = hp.horizon;
    if (hp.alpha != null)
      document.getElementById("alpha-slider").value = Math.log10(Math.max(hp.alpha, 1e-3)).toFixed(2);
  } finally {
    state.programmaticSliderSet = false;
  }
  updateReadouts();
}

function updateReadouts() {
  document.getElementById("lookback-readout").textContent = document.getElementById("lookback-slider").value;
  document.getElementById("horizon-readout").textContent = document.getElementById("horizon-slider").value;
  const alpha = Math.pow(10, parseFloat(document.getElementById("alpha-slider").value));
  document.getElementById("alpha-readout").textContent = `α = ${alpha.toExponential(2)}`;
}

function updateChips() {
  const c = document.getElementById("hp-chips");
  if (!c) return;
  const hp = state.hp;
  c.innerHTML = "";
  const chips = [
    ["L", hp.lookback],
    ["H", hp.horizon],
    ["α", hp.alpha.toExponential(2)],
    ["scope", hp.scaler_scope],
    ["method", hp.scaler_method],
    ["noise", hp.noise_type],
    ["σ", hp.aug_sigma.toFixed(3)],
    ["r", hp.local_ratio.toFixed(3)],
  ];
  for (const [k, v] of chips) {
    const el = document.createElement("span");
    el.className = "hp-chip";
    el.innerHTML = `${k} <strong>${v}</strong>`;
    c.appendChild(el);
  }
}

function snapPinnedHps() {
  // Pull all HPs from the active dataset's per-horizon-group baked weights.
  // The group that contains the current H determines L, α, r, scope, method,
  // noise, σ — so dragging H actually changes the displayed optimum and the
  // orange lookback box. Overrides (user-edited L or α) take precedence.
  const ds = activeDatasetEntry();
  if (!ds || !state.activeSeries) return;
  const seriesIdx = ds.series_names.indexOf(state.activeSeries);
  const sgIdx = seriesIdx >= 0 ? ds.series_to_sg[seriesIdx] : 0;
  const groups = (ds.sg_groups && ds.sg_groups[sgIdx]) || [];
  const H = state.hp.horizon;
  const g = groups.find((g) => H >= g.group_min && H <= g.group_max) || groups[groups.length - 1];
  if (g) {
    state.hp.scaler_scope = g.scaler_scope;
    state.hp.scaler_method = g.scaler_method;
    // The browser ridge worker only implements time-domain augmentation
    // (ridge.js:491). Map freq → none so the displayed HPs match what the
    // worker actually applies.
    state.hp.noise_type = g.noise_type === "freq" ? "none" : g.noise_type;
    state.hp.aug_sigma = state.hp.noise_type === "none" ? 0.0 : g.aug_sigma;
    state.hp.local_ratio = g.local_ratio;
    // Default L/α to the group's stored values unless the user has overridden.
    if (state.override.lookback == null) state.hp.lookback = g.lookback;
    else state.hp.lookback = state.override.lookback;
    if (state.override.alpha == null) state.hp.alpha = g.alpha;
    else state.hp.alpha = state.override.alpha;
    writeHpToUI({ lookback: state.hp.lookback, alpha: state.hp.alpha });
    // Re-read so state.hp matches the slider's stepped value (slider step=8
    // can snap L by a few units; the override-free case must read back the
    // exact value the user sees).
    if (state.override.lookback == null)
      state.hp.lookback = parseInt(document.getElementById("lookback-slider").value, 10);
    if (state.override.alpha == null)
      state.hp.alpha = Math.pow(10, parseFloat(document.getElementById("alpha-slider").value));
  }
}

// ---------------------------------------------------------------------------
// Plot rendering
// ---------------------------------------------------------------------------

function activeDatasetEntry() {
  const pc = state.manifest && state.manifest.precomputed;
  if (!pc || !pc.datasets) return null;
  const key = state.activeDataset || pc.default;
  return pc.datasets[key] || null;
}

// ---- β application (baked default series only) --------------------------
//
// The bake step (website/bake_weights.mjs) saves one Ridge β per H-group
// plus one baseline β, only for each dataset's `baked_series`. The helpers
// below apply those β to a length-L lookback window from the active series
// and produce a length-H prediction in z-scored space.
function predictLocalGroup(window, beta, F, H, lastK) {
  // window: Float64Array of length L = F - 1, in per-column-z-scored space.
  // beta: Float32Array of length F * H (row-major, F rows × H cols).
  // last_k matches optuna's `int(L * ratio)` (truncation).
  const L = F - 1;
  let mu = 0;
  for (let i = L - lastK; i < L; i++) mu += window[i];
  mu /= lastK;
  let v = 0;
  for (let i = L - lastK; i < L; i++) { const d = window[i] - mu; v += d * d; }
  const sigma = Math.sqrt(v / lastK + 1e-5);
  const Yhat = new Float64Array(H);
  for (let i = 0; i < L; i++) {
    const fi = window[i] - mu;
    const off = i * H;
    for (let j = 0; j < H; j++) Yhat[j] += fi * beta[off + j];
  }
  const offSigma = L * H;
  for (let j = 0; j < H; j++) Yhat[j] += sigma * beta[offSigma + j];
  for (let j = 0; j < H; j++) Yhat[j] += mu;
  return Yhat;
}

function predictGlobalGroup(window, beta, F, H) {
  // window: Float64Array of length L = F - 1 (globally z-scored).
  // beta has the bias as feature 0 and the L coefficients afterwards.
  const L = F - 1;
  const Yhat = new Float64Array(H);
  for (let j = 0; j < H; j++) Yhat[j] = beta[j];
  for (let i = 0; i < L; i++) {
    const wi = window[i];
    const off = (i + 1) * H;
    for (let j = 0; j < H; j++) Yhat[j] += wi * beta[off + j];
  }
  return Yhat;
}

function buildPrefitFromWeights() {
  // Apply the dataset's baked β to the active series's window at the
  // bake-time anchor. Returns null when:
  //   - weights.bin hasn't loaded yet,
  //   - the active series isn't this dataset's baked default series, or
  //   - any group is missing β metadata (groups[i].beta_offset / F / H).
  // Callers must then fall back to a worker refit.
  if (!state.weightsBin) return null;
  const ds = activeDatasetEntry();
  if (!ds || !state.dataset || !state.activeSeries) return null;
  if (ds.baked_series && state.activeSeries !== ds.baked_series) return null;
  const seriesIdx = ds.series_names.indexOf(state.activeSeries);
  if (seriesIdx < 0) return null;
  const sgIdx = ds.series_to_sg[seriesIdx];
  const groups = ds.sg_groups && ds.sg_groups[sgIdx];
  if (!groups || !ds.baseline || !groups.every((g) => g.beta_offset != null)) return null;

  const anchor = ds.test_anchor;
  const maxH = ds.max_h;
  const trainMean = ds.train_mean[seriesIdx];
  const trainStd = ds.train_std[seriesIdx];
  const series = state.dataset.series[state.activeSeries];

  const sliderMaxL = parseInt(
    (document.getElementById("lookback-slider") || {}).max, 10) || 2048;
  const groupMaxL = Math.max.apply(null, groups.map((g) => g.lookback));
  const desiredL = Math.max(groupMaxL, ds.baseline.lookback, sliderMaxL);
  const maxL = Math.min(desiredL, anchor);
  const histStart = anchor - maxL;
  const histLen = maxL;
  const history = new Float64Array(histLen);
  for (let i = 0; i < histLen; i++) history[i] = (series[histStart + i] - trainMean) / trainStd;
  const truth = new Float64Array(maxH);
  for (let j = 0; j < maxH; j++) truth[j] = (series[anchor + j] - trainMean) / trainStd;

  // Per-H-group "ours" predictions, concatenated into one 1..maxH trajectory.
  const ours = new Float64Array(maxH);
  for (const g of groups) {
    const L = g.lookback, F = g.F, H = g.H;
    const win = history.subarray(histLen - L, histLen);
    const beta = new Float32Array(state.weightsBin, g.beta_offset, F * H);
    const pred = g.scaler_scope === "local"
      ? predictLocalGroup(win, beta, F, H,
          Math.max(1, Math.min(L, Math.floor(L * g.local_ratio))))
      : predictGlobalGroup(win, beta, F, H);
    for (let j = 0; j < H; j++) ours[g.group_min - 1 + j] = pred[j];
  }

  // Baseline ridge: one global fit at L=720, α=1e-5 covering 1..maxH.
  const bm = ds.baseline;
  const baseWin = history.subarray(histLen - bm.lookback, histLen);
  const baseBeta = new Float32Array(state.weightsBin, bm.beta_offset, bm.F * bm.H);
  const basePred = predictGlobalGroup(baseWin, baseBeta, bm.F, bm.H);

  const seEntry = ds.per_series_se && ds.per_series_se[state.activeSeries];
  return {
    anchor, maxL, maxH,
    history: Array.from(history),
    truth: Array.from(truth),
    ourPred: Array.from(ours),
    basePred: Array.from(basePred),
    n_test_anchors: ds.n_test_anchors || 1,
    oursSe: seEntry ? seEntry.ours_se : null,
    baseSe: seEntry ? seEntry.base_se : null,
    nAnchorsForSe: ds.n_test_anchors,
  };
}

function tryInitialRender() {
  // Default state: try the baked β path first. Falls back to a live worker
  // refit when weights.bin isn't loaded yet, the active series isn't this
  // dataset's baked series, or the user has set L/α overrides.
  if (!state.manifest || !state.dataset) return;
  const hasOverride = state.override.lookback != null || state.override.alpha != null;
  if (!hasOverride) {
    const pf = buildPrefitFromWeights();
    if (pf) {
      state.prefit = pf;
      renderFromPrefit();
      const statusEl = document.getElementById("m-status");
      if (statusEl) statusEl.textContent = "predictions from pre-fit weights";
      setStatus("predictions from pre-fit weights");
      return;
    }
  }
  renderDefaultPlot();
  scheduleFit();
}

function renderFromPrefit() {
  if (!state.prefit) {
    renderDefaultPlot();
    return;
  }
  const pf = state.prefit;
  const H = state.hp.horizon;
  // Always pin the non-tunable HPs and reset L/α displays to the current
  // group's stored values (modulo any active user override).
  snapPinnedHps();
  updateChips();
  renderPlotPrefit(pf, H);
  applyPrefitMetrics(pf, H);
}

function renderDefaultPlot() {
  // Render history + truth + lookback span + horizon marker WITHOUT any
  // predictions. This is the zero-fit default view: tells the viewer where
  // the model would predict, what the lookback covers, and what the stored
  // HPs are, without waiting for the worker.
  if (!state.dataset || !state.activeSeries) return;
  const ds = activeDatasetEntry();
  const proto = ds && ds.protocol;
  if (!proto) return;
  const series = state.dataset.series[state.activeSeries];
  const total = Math.min(series.length, proto.total_samples);
  const nTrain = proto.n_train;
  const H = state.hp.horizon;
  // Anchor at the last possible test point that allows 720 future steps.
  const maxH = 720;
  const anchor = total - maxH;
  const L = state.hp.lookback;
  const maxL = 2048;
  // Global z-score using training-segment mean/std (matches the worker's basis).
  let mu = 0; for (let i = 0; i < nTrain; i++) mu += series[i]; mu /= nTrain;
  let sq = 0; for (let i = 0; i < nTrain; i++) sq += (series[i] - mu) ** 2;
  const sigma = Math.sqrt(sq / nTrain) || 1;
  const histStart = Math.max(0, anchor - maxL);
  const hist = new Array(anchor - histStart);
  for (let i = 0; i < hist.length; i++) hist[i] = (series[histStart + i] - mu) / sigma;
  const truth = new Array(H);
  for (let j = 0; j < H; j++) truth[j] = (series[anchor + j] - mu) / sigma;

  const histLen = hist.length;
  const xHist = Array.from({ length: histLen }, (_, i) => i - histLen);
  const xF = Array.from({ length: H }, (_, i) => i + 1);
  const allVals = hist.concat(truth);
  const yMin = Math.min.apply(null, allVals);
  const yMax = Math.max.apply(null, allVals);
  const yPad = (yMax - yMin) * 0.12 + 0.001;
  const scopeColor = state.hp.scaler_scope === "local" ? "#ff7f0e" : "#1f77b4";

  const traces = [
    { x: xHist, y: hist, type: "scatter", mode: "lines",
      name: "history", line: { color: "#383834", width: 1.4 } },
    { x: [0].concat(xF), y: [hist[histLen - 1]].concat(truth),
      type: "scatter", mode: "lines",
      name: "ground truth (1..H)", line: { color: "#1a1a1a", width: 2 } },
  ];
  const layout = {
    margin: { l: 44, r: 18, t: 16, b: 44 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: -0.18, font: { family: "system-ui", size: 12 } },
    xaxis: { zeroline: false, gridcolor: "#eee",
      title: { text: "samples relative to prediction start", font: { family: "system-ui", size: 11 } } },
    yaxis: { zeroline: false, gridcolor: "#eee" },
    plot_bgcolor: "white", paper_bgcolor: "white",
    shapes: [
      { type: "rect", xref: "x", yref: "y",
        x0: -L, x1: 0, y0: yMin - yPad, y1: yMax + yPad,
        line: { color: scopeColor, width: 1 }, fillcolor: scopeColor,
        opacity: 0.12, layer: "below" },
      { type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1,
        line: { color: "#555", width: 1, dash: "dash" } },
      { type: "line", xref: "x", yref: "paper", x0: H, x1: H, y0: 0, y1: 1,
        line: { color: "#0c6e6e", width: 1, dash: "dot" } },
    ],
    annotations: [
      { xref: "x", yref: "paper", x: -L / 2, y: 1.02,
        text: `L = ${L} (${state.hp.scaler_scope})`, showarrow: false,
        font: { family: "ui-monospace, monospace", size: 12, color: scopeColor } },
      { xref: "x", yref: "paper", x: H, y: 1.02, xanchor: "right",
        text: `H = ${H}`, showarrow: false,
        font: { family: "ui-monospace, monospace", size: 12, color: "#0c6e6e" } },
    ],
  };
  Plotly.react("demo-plot", traces, layout, { displayModeBar: false, responsive: true });
}

function applyPrefitMetrics(pf, H) {
  // Compute MSE aggregated over ALL test anchors (the paper's protocol)
  // using the per-step squared-error sums baked into the manifest and / or
  // returned by the worker. Falls back to single-anchor MSE if SE arrays
  // are unavailable.
  let mseOur, mseBase, nAnchors;
  if (pf.oursSe && pf.baseSe && pf.nAnchorsForSe) {
    const limit = Math.min(H, pf.oursSe.length, pf.baseSe.length);
    let cumO = 0, cumB = 0;
    for (let j = 0; j < limit; j++) { cumO += pf.oursSe[j]; cumB += pf.baseSe[j]; }
    nAnchors = pf.nAnchorsForSe;
    mseOur  = cumO / (nAnchors * limit);
    mseBase = cumB / (nAnchors * limit);
  } else {
    // Fallback: single displayed trajectory.
    let sOur = 0, sBase = 0, n = 0;
    const len = Math.min(H, pf.truth.length, pf.ourPred.length, pf.basePred.length);
    for (let j = 0; j < len; j++) {
      const t = pf.truth[j];
      sOur  += (pf.ourPred[j] - t) * (pf.ourPred[j] - t);
      sBase += (pf.basePred[j] - t) * (pf.basePred[j] - t);
      n++;
    }
    mseOur  = sOur  / Math.max(n, 1);
    mseBase = sBase / Math.max(n, 1);
    nAnchors = pf.n_test_anchors || 1;
  }
  const mae = Math.sqrt(mseOur);  // not exactly MAE; just a coarse rms-style stand-in for the second card
  state.lastFitResult = {
    metrics: { val_mse: mseOur, test_mse: mseOur, test_mae: NaN },
    n_train_windows: pf.n_test_anchors,
    stride: 1,
    _baselineMse: mseBase,
  };
  document.getElementById("m-mse").textContent = fmtNumber(mseOur, 3);
  document.getElementById("m-mae").textContent = fmtNumber(mseBase, 3);
  document.getElementById("m-n").textContent = (nAnchors || 1).toLocaleString();
  const strideEl = document.getElementById("m-stride");
  if (strideEl) strideEl.textContent = nAnchors > 1 ? "test anchors" : "stride 1";
  // Show the delta (ours vs prefit baseline). This is per-trajectory at H.
  const deltaEl = document.getElementById("m-delta");
  if (deltaEl) {
    const pct = ((mseBase - mseOur) / mseBase) * 100;
    deltaEl.textContent = `${pct >= 0 ? "−" : "+"}${Math.abs(pct).toFixed(1)}%`;
    deltaEl.classList.toggle("delta-good", pct >= 0);
    deltaEl.classList.toggle("delta-bad", pct < 0);
  }
}

function renderPlotPrefit(pf, H) {
  const lookback = state.hp.lookback;
  const maxL = pf.maxL;
  const histLen = pf.history.length;
  const xHist = Array.from({ length: histLen }, (_, i) => i - histLen);
  // Subsample the trajectory by ~3 for big H so Plotly stays snappy.
  const stride = H > 360 ? 3 : (H > 180 ? 2 : 1);
  const xF = [];
  const truthF = [];
  const ourF = [];
  const baseF = [];
  for (let j = 0; j < H; j += stride) {
    xF.push(j + 1);
    truthF.push(pf.truth[j]);
    ourF.push(pf.ourPred[j]);
    baseF.push(pf.basePred[j]);
  }
  // Always include the very last point so the curves don't visually undershoot H.
  if (xF[xF.length - 1] !== H) {
    xF.push(H);
    truthF.push(pf.truth[H - 1]);
    ourF.push(pf.ourPred[H - 1]);
    baseF.push(pf.basePred[H - 1]);
  }

  const scopeColor = state.hp.scaler_scope === "local" ? "#ff7f0e" : "#1f77b4";
  const allVals = pf.history.concat(truthF).concat(ourF).concat(baseF);
  const yMin = Math.min.apply(null, allVals);
  const yMax = Math.max.apply(null, allVals);
  const yPad = (yMax - yMin) * 0.12 + 0.001;

  const traces = [
    { x: xHist, y: pf.history, type: "scatter", mode: "lines",
      name: "history (L)", line: { color: "#383834", width: 1.4 } },
    { x: [0].concat(xF), y: [pf.history[histLen - 1]].concat(truthF),
      type: "scatter", mode: "lines",
      name: "ground truth", line: { color: "#1a1a1a", width: 2 } },
    { x: [0].concat(xF), y: [pf.history[histLen - 1]].concat(ourF),
      type: "scatter", mode: "lines",
      name: "ours (per-group searched HPs)",
      line: { color: "#0c6e6e", width: 2.4 } },
    { x: [0].concat(xF), y: [pf.history[histLen - 1]].concat(baseF),
      type: "scatter", mode: "lines",
      name: "global baseline (L=720, α=10⁻⁵)",
      line: { color: "#b53a1a", width: 1.6, dash: "dot" } },
  ];
  const layout = {
    margin: { l: 44, r: 18, t: 16, b: 44 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: -0.18, font: { family: "system-ui", size: 12 } },
    xaxis: { zeroline: false, gridcolor: "#eee",
      title: { text: "samples relative to prediction start", font: { family: "system-ui", size: 11 } } },
    yaxis: { zeroline: false, gridcolor: "#eee" },
    plot_bgcolor: "white", paper_bgcolor: "white",
    shapes: [
      // Lookback span for the CURRENT horizon group's lookback (so users see
      // which slice of history the model used). In autotune ON this matches
      // the searched lookback of the group containing H.
      { type: "rect", xref: "x", yref: "y",
        x0: -lookback, x1: 0,
        y0: yMin - yPad, y1: yMax + yPad,
        line: { color: scopeColor, width: 1 }, fillcolor: scopeColor,
        opacity: 0.12, layer: "below" },
      // Prediction-start vertical dashed line.
      { type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1,
        line: { color: "#555", width: 1, dash: "dash" } },
      // Horizon target vertical line.
      { type: "line", xref: "x", yref: "paper", x0: H, x1: H, y0: 0, y1: 1,
        line: { color: "#0c6e6e", width: 1, dash: "dot" } },
    ],
    annotations: [
      { xref: "x", yref: "paper", x: -lookback / 2, y: 1.02,
        text: `L = ${lookback} (${state.hp.scaler_scope})`, showarrow: false,
        font: { family: "ui-monospace, monospace", size: 12, color: scopeColor } },
      { xref: "x", yref: "paper", x: H, y: 1.02, xanchor: "right",
        text: `H = ${H}`, showarrow: false,
        font: { family: "ui-monospace, monospace", size: 12, color: "#0c6e6e" } },
    ],
  };
  Plotly.react("demo-plot", traces, layout, { displayModeBar: false, responsive: true });
}

function renderPlot(fitResult) {
  const ex = fitResult.example;
  const lookback = ex.lookback;
  const horizon = ex.horizon;
  const histLen = ex.history.length;
  const xHist = Array.from({ length: histLen }, (_, i) => i - histLen);
  // truth_context is the continuous trajectory between the prediction-start
  // line and step H, shown thin as a reference so users see WHERE H sits.
  const ctx = ex.truth_context || [];
  const xCtx = Array.from({ length: ctx.length }, (_, i) => i + 1);
  const scopeColor = state.hp.scaler_scope === "local" ? "#ff7f0e" : "#1f77b4";

  const allVals = ex.history.concat(ctx).concat([ex.truth_at_h, ex.pred_at_h]);
  const yMin = Math.min.apply(null, allVals);
  const yMax = Math.max.apply(null, allVals);
  const yPad = (yMax - yMin) * 0.15 + 0.001;

  const traces = [
    {
      x: xHist, y: ex.history,
      type: "scatter", mode: "lines",
      name: "history (L)",
      line: { color: "#383834", width: 1.4 },
    },
    {
      x: [0].concat(xCtx),
      y: [ex.history[histLen - 1]].concat(ctx),
      type: "scatter", mode: "lines",
      name: `future ground truth (1..${horizon})`,
      line: { color: "#999", width: 1.2, dash: "dot" },
      hoverinfo: "skip",
    },
    {
      x: [horizon], y: [ex.truth_at_h],
      type: "scatter", mode: "markers",
      name: `truth @ H=${horizon}`,
      marker: { color: "#1a1a1a", size: 11, symbol: "circle-open", line: { width: 2 } },
    },
    {
      x: [horizon], y: [ex.pred_at_h],
      type: "scatter", mode: "markers",
      name: `prediction @ H=${horizon}`,
      marker: { color: "#0c6e6e", size: 12, symbol: "x-thin", line: { width: 3 } },
    },
  ];

  const layout = {
    margin: { l: 44, r: 18, t: 16, b: 36 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: -0.16, font: { family: "system-ui", size: 12 } },
    xaxis: {
      zeroline: false,
      gridcolor: "#eee",
      title: { text: "samples relative to prediction start", font: { family: "system-ui", size: 11 } },
    },
    yaxis: {
      zeroline: false,
      gridcolor: "#eee",
    },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    shapes: [
      // Lookback span (cool/warm by scope)
      {
        type: "rect",
        xref: "x",
        yref: "y",
        x0: -lookback,
        x1: 0,
        y0: yMin - yPad,
        y1: yMax + yPad,
        line: { color: scopeColor, width: 1 },
        fillcolor: scopeColor,
        opacity: 0.12,
        layer: "below",
      },
      // Prediction-start vertical dashed line
      {
        type: "line",
        xref: "x",
        yref: "paper",
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 1,
        line: { color: "#555", width: 1, dash: "dash" },
      },
      // Vertical guide at horizon H to highlight the prediction target.
      {
        type: "line",
        xref: "x",
        yref: "paper",
        x0: horizon, x1: horizon, y0: 0, y1: 1,
        line: { color: "#0c6e6e", width: 1, dash: "dot" },
      },
    ],
    annotations: [
      {
        xref: "x",
        yref: "paper",
        x: -lookback / 2,
        y: 1.02,
        text: `L = ${lookback} (${state.hp.scaler_scope})`,
        showarrow: false,
        font: { family: "ui-monospace, monospace", size: 12, color: scopeColor },
      },
      {
        xref: "x", yref: "paper",
        x: horizon, y: 1.02, xanchor: "right",
        text: `H = ${horizon}`,
        showarrow: false,
        font: { family: "ui-monospace, monospace", size: 12, color: "#0c6e6e" },
      },
    ],
  };

  Plotly.react("demo-plot", traces, layout, { displayModeBar: false, responsive: true });
}

function fmtNumber(x, sig = 3) {
  if (!isFinite(x)) return "—";
  if (Math.abs(x) >= 1000 || (Math.abs(x) < 0.01 && x !== 0)) return x.toExponential(sig);
  return x.toPrecision(sig + 1);
}

function baselineKey(horizon) {
  if (!state.dataset || !state.activeSeries) return null;
  return `${state.dataset.source}|${state.activeSeries}|${horizon}`;
}

function refreshDelta() {
  const deltaEl = document.getElementById("m-delta");
  if (!deltaEl) return;
  const fit = state.lastFitResult;
  if (!fit) {
    deltaEl.textContent = "—";
    deltaEl.classList.remove("delta-good", "delta-bad");
    return;
  }
  const key = baselineKey(state.hp.horizon);
  const baselineMSE = key ? state.baselineCache.get(key) : null;
  if (baselineMSE != null) {
    const pct = ((baselineMSE - fit.metrics.test_mse) / baselineMSE) * 100;
    deltaEl.textContent = `${pct >= 0 ? "−" : "+"}${Math.abs(pct).toFixed(1)}%`;
    deltaEl.classList.toggle("delta-good", pct >= 0);
    deltaEl.classList.toggle("delta-bad", pct < 0);
  } else if (key && state.baselineComputing.has(key)) {
    deltaEl.textContent = "computing…";
    deltaEl.classList.remove("delta-good", "delta-bad");
  } else {
    deltaEl.textContent = "—";
    deltaEl.classList.remove("delta-good", "delta-bad");
  }
}

function setMetrics(fitResult, elapsed) {
  const m = fitResult.metrics;
  state.lastFitResult = fitResult;
  document.getElementById("m-mse").textContent = fmtNumber(m.test_mse, 3);
  document.getElementById("m-mae").textContent = fmtNumber(m.test_mae, 3);
  document.getElementById("m-n").textContent = fitResult.n_train_windows.toLocaleString();
  const strideEl = document.getElementById("m-stride");
  if (strideEl) strideEl.textContent = `stride ${fitResult.stride}`;
  if (elapsed != null)
    document.getElementById("m-time").textContent = `${elapsed.toFixed(0)} ms`;
  const statusEl = document.getElementById("m-status");
  if (statusEl) statusEl.textContent = elapsed > 1500 ? "heavy fit" : "ok";
  refreshDelta();
}

// ---------------------------------------------------------------------------
// Fit pipeline
// ---------------------------------------------------------------------------

let pendingDebounce = null;
let busy = false;
let pendingAfterBusy = false;

function withProtocol(hp) {
  // Attach the dataset's exact split protocol (truncation + n_train / n_val /
  // n_test + search ratios) so the worker uses paper-faithful boundaries.
  const proto = state.manifest && state.manifest.protocol;
  return proto ? { ...hp, protocol: proto } : hp;
}

function currentPerGroupHps() {
  // Autotune ON: return the active series's searched-optimum per-horizon-group
  // HPs from the manifest. One ridge is fit per group (matches the paper protocol
  // — sgs=1, gs=48).
  //
  // Autotune OFF (overrides set): collapse to a single 1..720 ridge using the
  // slider's L/α so the override applies uniformly across the forecast.
  const hp = state.hp;
  const ovL = state.override.lookback;
  const ovA = state.override.alpha;
  const hasOverride = ovL != null || ovA != null;
  const ds = activeDatasetEntry();
  if (!hasOverride && ds && state.activeSeries) {
    const seriesIdx = ds.series_names.indexOf(state.activeSeries);
    const sgIdx = seriesIdx >= 0 ? ds.series_to_sg[seriesIdx] : 0;
    const groups = (ds.sg_groups && ds.sg_groups[sgIdx]) || [];
    if (groups.length > 0) {
      return groups.map((g) => ({
        group_min: g.group_min, group_max: g.group_max,
        lookback: g.lookback, alpha: g.alpha,
        scaler_method: g.scaler_method, scaler_scope: g.scaler_scope,
        noise_type: g.noise_type === "freq" ? "none" : g.noise_type,
        aug_sigma: g.noise_type === "freq" ? 0.0 : g.aug_sigma,
        local_ratio: g.local_ratio,
      }));
    }
  }
  return [{
    group_min: 1, group_max: 720,
    lookback: ovL != null ? ovL : hp.lookback,
    alpha:    ovA != null ? ovA : hp.alpha,
    scaler_method: hp.scaler_method, scaler_scope: hp.scaler_scope,
    noise_type: hp.noise_type === "freq" ? "none" : hp.noise_type,
    aug_sigma: hp.noise_type === "freq" ? 0.0 : hp.aug_sigma,
    local_ratio: hp.local_ratio,
  }];
}

function baselineHpFromManifest() {
  return (state.manifest && state.manifest.global_baseline) || {
    lookback: 720, alpha: 1e-5, scaler_method: "mean",
    scaler_scope: "global", noise_type: "none", aug_sigma: 0.0, local_ratio: 1.0,
  };
}

async function runPrefit() {
  if (!state.dataset || !state.activeSeries) return;
  const ovL = state.override.lookback;
  const ovA = state.override.alpha;
  const tag = (ovL == null && ovA == null)
    ? "stored"
    : `ovr|L${ovL != null ? ovL : "-"}|a${ovA != null ? ovA.toExponential(2) : "-"}`;
  const cacheKey = `${state.activeDataset}|${state.activeSeries}|${tag}`;
  if (state.prefitCache.has(cacheKey)) {
    state.prefit = state.prefitCache.get(cacheKey);
    return state.prefit;
  }
  busy = true;
  const msg = (ovL == null && ovA == null)
    ? "refitting from searched HPs…"
    : "refitting with your L/α…";
  setStatus(msg, true);
  const statusEl = document.getElementById("m-status");
  if (statusEl) statusEl.textContent = "refitting…";
  try {
    const t0 = performance.now();
    const series = state.dataset.series[state.activeSeries];
    const ds = activeDatasetEntry();
    const proto = ds && ds.protocol;
    const res = await postFitVis(
      series, currentPerGroupHps(), baselineHpFromManifest(), proto, {},
    );
    const elapsed = performance.now() - t0;
    state.prefit = res.result;
    state.prefitCache.set(cacheKey, state.prefit);
    setStatus(`refit in ${elapsed.toFixed(0)} ms`);
    if (statusEl) statusEl.textContent = `refit in ${elapsed.toFixed(0)} ms`;
    const timeEl = document.getElementById("m-time");
    if (timeEl) timeEl.textContent = `${elapsed.toFixed(0)} ms`;
    return state.prefit;
  } catch (err) {
    console.error(err);
    setStatus("refit error: " + err.message);
    return null;
  } finally {
    busy = false;
  }
}

async function fitNow() {
  if (!state.dataset || !state.activeSeries) return;
  if (busy) { pendingAfterBusy = true; return; }
  updateReadouts();
  await runPrefit();
  renderFromPrefit();
  if (pendingAfterBusy) {
    pendingAfterBusy = false;
    setTimeout(fitNow, 0);
  }
}

function scheduleFit() {
  clearTimeout(pendingDebounce);
  pendingDebounce = setTimeout(fitNow, 120);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function setAutotune(on) {
  state.autotune = !!on;
  const btn = document.getElementById("autotune-btn");
  if (btn) {
    btn.classList.toggle("on", state.autotune);
    btn.classList.toggle("off", !state.autotune);
    btn.setAttribute("aria-pressed", String(state.autotune));
    const label = btn.querySelector(".autotune-state");
    if (label) label.textContent = state.autotune ? "ON" : "OFF";
  }
}

// "input" fires continuously during drag — update the chip + orange box for
// real-time visual feedback, but do NOT trigger a refit. The refit goes on
// "change" instead (fires only on mouse release / Enter key).
function onLookbackSliderInput() {
  if (state.programmaticSliderSet) return;
  if (state.autotune) setAutotune(false);
  state.override.lookback = parseInt(document.getElementById("lookback-slider").value, 10);
  state.hp.lookback = state.override.lookback;
  updateChips();
  updateReadouts();
  if (state.prefit) renderPlotPrefit(state.prefit, state.hp.horizon);
}

function onLookbackSliderChange() {
  if (state.programmaticSliderSet) return;
  scheduleFit();
}

function onAlphaSliderInput() {
  if (state.programmaticSliderSet) return;
  if (state.autotune) setAutotune(false);
  state.override.alpha = Math.pow(10, parseFloat(document.getElementById("alpha-slider").value));
  state.hp.alpha = state.override.alpha;
  updateChips();
  updateReadouts();
}

function onAlphaSliderChange() {
  if (state.programmaticSliderSet) return;
  scheduleFit();
}

function onAutotuneClick() {
  const next = !state.autotune;
  setAutotune(next);
  if (next) {
    // Switching ON: drop overrides, snap sliders back to the baked optimum,
    // and re-render from the pre-fit weights (no worker needed).
    state.override.lookback = null;
    state.override.alpha = null;
    state.prefit = null;
    snapPinnedHps();
    updateChips();
    tryInitialRender();
  } else {
    // Switching OFF manually: keep current slider values and refit.
    state.override.lookback = state.hp.lookback;
    state.override.alpha = state.hp.alpha;
    scheduleFit();
  }
}

function onHorizonSliderEdit() {
  // Horizon slider snaps L/α to the new group's stored optimum (unless the
  // user overrode them) and re-renders. If a prefit is cached, use it;
  // otherwise show the no-prediction default plot.
  state.hp.horizon = parseInt(document.getElementById("horizon-slider").value, 10);
  snapPinnedHps();
  updateChips();
  updateReadouts();
  if (state.prefit) renderFromPrefit();
  else renderDefaultPlot();
}

function bindControls() {
  const lb = document.getElementById("lookback-slider");
  lb.addEventListener("input", onLookbackSliderInput);   // visual only
  lb.addEventListener("change", onLookbackSliderChange); // refit on release
  const al = document.getElementById("alpha-slider");
  al.addEventListener("input", onAlphaSliderInput);
  al.addEventListener("change", onAlphaSliderChange);
  document.getElementById("horizon-slider").addEventListener("input", onHorizonSliderEdit);
  const auto = document.getElementById("autotune-btn");
  if (auto) auto.addEventListener("click", onAutotuneClick);

  document.getElementById("series-select").addEventListener("change", (e) => {
    state.activeSeries = e.target.value;
    state.override.lookback = null;
    state.override.alpha = null;
    state.prefit = null;
    snapPinnedHps();
    updateChips();
    tryInitialRender();
  });
  document.getElementById("dataset-select").addEventListener("change", async (e) => {
    const key = e.target.value;
    if (!key || key === state.activeDataset) return;
    state.override.lookback = null;
    state.override.alpha = null;
    state.prefit = null;
    state.prefitCache.clear();
    await loadDataset(key);
    setAutotune(true);
    snapPinnedHps();
    updateChips();
    updateReadouts();
    tryInitialRender();
  });
}

async function ensureBaseline(horizon) {
  const key = baselineKey(horizon);
  if (!key) return;
  if (state.baselineCache.has(key)) return;
  if (state.baselineComputing.has(key)) return;
  state.baselineComputing.add(key);
  const series = state.dataset.series[state.activeSeries];
  const baselineHp = state.manifest && state.manifest.naive_baseline
    ? { ...state.manifest.naive_baseline, horizon }
    : {
        lookback: 96, horizon,
        alpha: 1.0, scaler_scope: "global", scaler_method: "mean",
        noise_type: "none", aug_sigma: 0.0, local_ratio: 1.0,
      };
  try {
    const r = await postFit(series, withProtocol(baselineHp), "fit");
    state.baselineCache.set(key, r.result.metrics.test_mse);
    if (state.hp.horizon === horizon) refreshDelta();
  } catch (e) {
    /* leave baseline unset */
  } finally {
    state.baselineComputing.delete(key);
  }
}

async function main() {
  state.worker = newWorker();
  bindControls();
  await loadManifestAndWeights();
  populateDatasetDropdown();
  const pc = state.manifest && state.manifest.precomputed;
  const defaultKey = (pc && pc.default) || "etth1";
  const qs = new URLSearchParams(window.location.search);
  const dsKey = qs.get("dataset") && pc.datasets[qs.get("dataset")] ? qs.get("dataset") : defaultKey;
  await loadDataset(dsKey);
  document.getElementById("dataset-select").value = dsKey;
  if (qs.get("h")) document.getElementById("horizon-slider").value = qs.get("h");
  if (qs.get("series")) {
    const s = qs.get("series");
    if (state.dataset.columns.includes(s)) {
      state.activeSeries = s;
      document.getElementById("series-select").value = s;
    }
  }
  state.hp.horizon = parseInt(document.getElementById("horizon-slider").value, 10);
  setAutotune(true);
  snapPinnedHps();
  updateChips();
  updateReadouts();
  // First paint: render predictions from baked weights when ready; otherwise
  // fall back to history-only.
  tryInitialRender();
  const statusEl = document.getElementById("m-status");
  if (statusEl && !state.prefit) statusEl.textContent = "loading weights…";
}

main();
