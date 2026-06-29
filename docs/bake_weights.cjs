/* Offline bake of Ridge β weights for the default series of each playground
 * dataset. Same JS as the worker (loads app/ridge.js), so baked predictions
 * are bit-exact with a live refit.
 *
 * Run from the repo root, AFTER `python website/build_data.py`:
 *   node website/bake_weights.cjs
 *
 * Side effects:
 *   - writes website/app/data/weights.bin (concatenated Float32 β arrays)
 *   - patches website/app/data/manifest.json with:
 *       precomputed.datasets.<key>.baseline = { beta_offset, F, H, ... }
 *       precomputed.datasets.<key>.sg_groups[default_idx][g].beta_offset/F/H
 *       precomputed.datasets.<key>.per_series_se[default_series] = {ours_se, base_se}
 *       precomputed.datasets.<key>.baked_series = "<default series name>"
 *       precomputed.weights_bin = "weights.bin"
 *
 * Targets CommonJS (Node 10+) since that's what's installed in this repo.
 */

// Node 10 (the version in this repo's runner) doesn't support the `node:`
// scheme; fall back to bare specifiers.
const fs = require("fs");
const path = require("path");

const APP_DATA = path.join(__dirname, "app", "data");
const MANIFEST_PATH = path.join(APP_DATA, "manifest.json");
const WEIGHTS_PATH = path.join(APP_DATA, "weights.bin");
const RIDGE_PATH = path.join(__dirname, "app", "ridge.js");

function loadRidgeLib() {
  // ridge.js attaches its exports to `self.RidgeLib`. Run it inside a fresh
  // Function so we can pass our own sandbox object as `self` (Node 21+ has
  // a built-in `self` global; older Node doesn't, so the function arg is
  // safest).
  let src = fs.readFileSync(RIDGE_PATH, "utf8");
  // Node 10 doesn't parse nullish coalescing. The only `??` uses in ridge.js
  // default `search_*_ratio` to non-zero numbers, so `||` is equivalent here.
  src = src.replace(/\?\?/g, "||");
  const sandboxSelf = {};
  const perf = (typeof performance !== "undefined")
    ? performance
    : { now: () => Date.now() };
  const fn = new Function("self", "performance", "console", src);
  fn(sandboxSelf, perf, console);
  if (!sandboxSelf.RidgeLib) throw new Error("ridge.js did not expose RidgeLib");
  return sandboxSelf.RidgeLib;
}

function parseCsvColumn(csvPath, column) {
  const text = fs.readFileSync(csvPath, "utf8");
  const lines = text.split(/\r?\n/);
  const headers = lines[0].split(",");
  const col = headers.indexOf(column);
  if (col < 0) throw new Error(`column "${column}" not found in ${csvPath}`);
  const out = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    out.push(parseFloat(lines[i].split(",")[col]));
  }
  return new Float64Array(out);
}

async function main() {
  const { bakeWeights } = loadRidgeLib();
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  const baselineHp = manifest.global_baseline;
  if (!baselineHp) throw new Error("manifest.global_baseline missing");
  const pc = manifest.precomputed;
  if (!pc || !pc.datasets) throw new Error("manifest.precomputed.datasets missing");

  const blobParts = [];
  let blobOffset = 0;

  for (const key of Object.keys(pc.datasets)) {
    const ds = pc.datasets[key];
    const defaultSeries = ds.default_series;
    if (!defaultSeries) {
      console.warn(`[${key}] no default_series - skipping`);
      continue;
    }
    const seriesIdx = ds.series_names.indexOf(defaultSeries);
    if (seriesIdx < 0) {
      console.warn(`[${key}] default_series=${defaultSeries} not in series_names - skipping`);
      continue;
    }
    const sgIdx = ds.series_to_sg[seriesIdx];
    const perGroupHps = ds.sg_groups[sgIdx].map(function (g) {
      return {
        group_min: g.group_min, group_max: g.group_max,
        lookback: g.lookback, alpha: g.alpha,
        scaler_method: g.scaler_method, scaler_scope: g.scaler_scope,
        noise_type: g.noise_type === "freq" ? "none" : g.noise_type,
        aug_sigma: g.noise_type === "freq" ? 0.0 : g.aug_sigma,
        local_ratio: g.local_ratio,
      };
    });

    const csvPath = path.join(APP_DATA, ds.csv_path);
    const series = parseCsvColumn(csvPath, defaultSeries);

    const t0 = Date.now();
    console.log(`[${key}] baking ${defaultSeries}: ${perGroupHps.length} H-groups + baseline ...`);
    const baked = await bakeWeights(
      series, perGroupHps, baselineHp, { protocol: ds.protocol }
    );
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`  done in ${elapsed.toFixed(1)}s (n_anchors=${baked.n_test_anchors})`);

    // Per-H-group β: append to blob, attach offset/F/H to the matching
    // sg_groups entry (we're patching only the default series's column).
    const bakedGroups = ds.sg_groups[sgIdx].map(function (g, i) {
      const bg = baked.groups[i];
      if (!bg.beta) return Object.assign({}, g);
      blobParts.push(Buffer.from(bg.beta.buffer, bg.beta.byteOffset, bg.beta.byteLength));
      const out = Object.assign({}, g, { beta_offset: blobOffset, F: bg.F, H: bg.H });
      blobOffset += bg.beta.byteLength;
      return out;
    });
    ds.sg_groups[sgIdx] = bakedGroups;

    // Baseline β (shared across this dataset's series at the JS layer too,
    // since L=720 global only depends on the (already per-series z-scored)
    // window the JS feeds in).
    if (baked.baseline && baked.baseline.beta) {
      const b = baked.baseline;
      blobParts.push(Buffer.from(b.beta.buffer, b.beta.byteOffset, b.beta.byteLength));
      ds.baseline = {
        lookback: b.hp.lookback,
        alpha: b.hp.alpha,
        scaler_scope: b.hp.scaler_scope || "global",
        scaler_method: b.hp.scaler_method || "mean",
        beta_offset: blobOffset,
        F: b.F,
        H: b.H,
      };
      blobOffset += b.beta.byteLength;
    }

    // Per-step SE for the default series (cumulated over all test anchors).
    ds.per_series_se = ds.per_series_se || {};
    ds.per_series_se[defaultSeries] = {
      ours_se: baked.oursSe,
      base_se: baked.baseSe,
    };
    ds.baked_series = defaultSeries;
    ds.n_test_anchors = baked.n_test_anchors;
    ds.test_anchor = baked.test_anchor;
  }

  fs.writeFileSync(WEIGHTS_PATH, Buffer.concat(blobParts));
  pc.weights_bin = "weights.bin";
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));

  const kb = fs.statSync(WEIGHTS_PATH).size / 1024;
  console.log(`wrote ${path.relative(process.cwd(), WEIGHTS_PATH)} (${kb.toFixed(0)} KB)`);
  console.log(`patched ${path.relative(process.cwd(), MANIFEST_PATH)}`);
}

main().catch(function (err) {
  console.error(err);
  process.exit(1);
});
