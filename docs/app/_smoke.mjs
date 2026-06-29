// Quick sanity check: import ridge.js and run a tiny fit. Run with: node website/app/_smoke.mjs
import { fitAndEvaluate } from "./ridge.js";

// Synthetic AR(1)-ish series: y_t = 0.9 * y_{t-1} + small noise
const n = 4000;
const series = new Float64Array(n);
series[0] = 0;
for (let i = 1; i < n; i++) {
  series[i] = 0.9 * series[i - 1] + (Math.random() - 0.5) * 0.2;
}

const hps = [
  { lookback: 32, horizon: 24, alpha: 1.0, scaler_scope: "local", scaler_method: "mean", noise_type: "none", aug_sigma: 0.0, local_ratio: 0.5 },
  { lookback: 96, horizon: 96, alpha: 1.0, scaler_scope: "global", scaler_method: "mean", noise_type: "none", aug_sigma: 0.0, local_ratio: 0.5 },
  { lookback: 192, horizon: 96, alpha: 10, scaler_scope: "local", scaler_method: "mean", noise_type: "time", aug_sigma: 0.05, local_ratio: 0.1 },
];

for (const hp of hps) {
  const t0 = process.hrtime.bigint();
  const r = fitAndEvaluate(series, hp);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log(`L=${hp.lookback} H=${hp.horizon} scope=${hp.scaler_scope} ${hp.noise_type}: test_mse=${r.metrics.test_mse.toExponential(2)} n_train=${r.n_train_windows} stride=${r.stride} fit=${ms.toFixed(0)}ms`);
}
