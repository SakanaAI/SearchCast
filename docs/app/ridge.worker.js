/* Classic Web Worker that runs Ridge fits off the main thread.
 *
 * Loads TF.js + the WebGPU backend via importScripts, probes for an
 * accelerator at startup, and installs a GPU backend on RidgeLib if one is
 * available. The CPU fallback is the default. */

importScripts(
  "../scripts/vendor/tf.min.js",
  "../scripts/vendor/tf-backend-webgpu.min.js",
  "./ridge.js"
);

// ----- GPU backend ---------------------------------------------------------

function makeTfBackend(name) {
  return {
    name,
    async buildGrams(X, Y, N, L, H) {
      // X is Float64Array (N*L); upload as Float32 for GPU and convert back.
      const Xf32 = X instanceof Float32Array ? X : new Float32Array(X);
      const Yf32 = Y instanceof Float32Array ? Y : new Float32Array(Y);
      const [XTX, XTY] = tf.tidy(() => {
        const Xt = tf.tensor2d(Xf32, [N, L]);
        const Yt = tf.tensor2d(Yf32, [N, H]);
        const Xtr = tf.transpose(Xt);
        return [tf.matMul(Xtr, Xt), tf.matMul(Xtr, Yt)];
      });
      const xtxArr = await XTX.data();
      const xtyArr = await XTY.data();
      XTX.dispose();
      XTY.dispose();
      return { XTX: new Float64Array(xtxArr), XTY: new Float64Array(xtyArr) };
    },
    async predict(X, beta, N, F, H) {
      const Xf32   = X    instanceof Float32Array ? X    : new Float32Array(X);
      const betaF  = beta instanceof Float32Array ? beta : new Float32Array(beta);
      const out = tf.tidy(() => {
        const Xt = tf.tensor2d(Xf32, [N, F]);
        const Bt = tf.tensor2d(betaF, [F, H]);
        return tf.matMul(Xt, Bt);
      });
      const arr = await out.data();
      out.dispose();
      return new Float64Array(arr);
    },
  };
}

let backendName = "cpu";

async function initBackend() {
  // Try WebGPU, then WebGL, then keep CPU.
  for (const name of ["webgpu", "webgl"]) {
    try {
      await tf.setBackend(name);
      await tf.ready();
      // Smoke test: run a tiny matmul to make sure the backend actually works
      // (some browsers expose the API but fail on first use).
      const probe = tf.tidy(() => tf.matMul(tf.ones([4, 4]), tf.ones([4, 4])));
      await probe.data();
      probe.dispose();
      RidgeLib.setBackend(makeTfBackend(name));
      backendName = name;
      break;
    } catch (e) {
      // try next
    }
  }
  self.postMessage({ type: "backend-ready", backend: backendName });
}

// ----- HP random-search trial ---------------------------------------------

function randomSearchTrial(hp, space) {
  const draw = { ...hp };
  if (space.lookback) {
    const lo = Math.log(space.lookback.min), hi = Math.log(space.lookback.max);
    draw.lookback = Math.max(8, Math.round(Math.exp(lo + Math.random() * (hi - lo))));
  }
  if (space.local_ratio) {
    const lo = Math.log(space.local_ratio.min), hi = Math.log(space.local_ratio.max);
    draw.local_ratio = Math.exp(lo + Math.random() * (hi - lo));
  }
  if (space.alpha) {
    const lo = Math.log(space.alpha.min), hi = Math.log(space.alpha.max);
    draw.alpha = Math.exp(lo + Math.random() * (hi - lo));
  }
  if (space.aug_sigma) {
    const lo = Math.log(space.aug_sigma.min), hi = Math.log(space.aug_sigma.max);
    draw.aug_sigma = Math.exp(lo + Math.random() * (hi - lo));
  }
  if (space.scaler_method) {
    draw.scaler_method = space.scaler_method[Math.floor(Math.random() * space.scaler_method.length)];
  }
  if (space.scaler_scope) {
    draw.scaler_scope = space.scaler_scope[Math.floor(Math.random() * space.scaler_scope.length)];
  }
  if (space.noise_type) {
    draw.noise_type = space.noise_type[Math.floor(Math.random() * space.noise_type.length)];
  }
  return draw;
}

// ----- Message dispatch ----------------------------------------------------

self.onmessage = async (event) => {
  const { type, id, payload } = event.data;
  try {
    if (type === "fit") {
      const t0 = performance.now();
      const result = await RidgeLib.fitAndEvaluate(payload.series, payload.hp, { mode: payload.mode || "fit" });
      const elapsed = performance.now() - t0;
      self.postMessage({ id, type: "fit-result", result, elapsed });
    } else if (type === "fitVis") {
      const t0 = performance.now();
      const result = await RidgeLib.fitForVisualization(
        payload.series, payload.perGroupHps, payload.baselineHp,
        { protocol: payload.protocol, skipBaseline: !!payload.skipBaseline },
      );
      const elapsed = performance.now() - t0;
      self.postMessage({ id, type: "fitVis-result", result, elapsed });
    } else if (type === "autoSearch") {
      const { series, baseHp, space, trials, mode } = payload;
      let best = null;
      const history = [];
      for (let t = 0; t < trials; t++) {
        const hp = randomSearchTrial(baseHp, space);
        let r;
        try {
          r = await RidgeLib.fitAndEvaluate(series, hp, { mode: mode || "search" });
        } catch (err) {
          continue;
        }
        history.push({ hp, val_mse: r.metrics.val_mse, test_mse: r.metrics.test_mse });
        if (best == null || r.metrics.val_mse < best.metrics.val_mse) {
          best = { hp, metrics: r.metrics, result: r };
        }
        self.postMessage({
          id, type: "autoSearch-progress",
          trial: t, history, best: best ? { hp: best.hp, metrics: best.metrics } : null,
        });
      }
      self.postMessage({ id, type: "autoSearch-result", best, history });
    } else {
      self.postMessage({ id, type: "error", error: "Unknown message type: " + type });
    }
  } catch (err) {
    self.postMessage({ id, type: "error", error: err && err.message ? err.message : String(err) });
  }
};

initBackend();
