/**
 * Native Plotly redraws of the paper's key figures from assets/figures_data.json.
 *
 * - #fig-lookback         : Exp1, L*(H) per dataset, log-log + power-law lines.
 * - #fig-hm-etth1-r/-a    : Exp2, per-series log10(r) and log10(α) on ETTh1.
 * - #fig-hm-weather-r/-a  : same on Weather (the cross-series contrast).
 * - #fig-pareto           : Exp3, MSE degradation vs. horizon group size.
 * - #fig-aug              : Exp4, augmentation selection rate per dataset.
 */

(function () {
  "use strict";

  const log10 = (x) => Math.log10(x);

  function lookbackFig(data, host) {
    const traces = [];
    for (const ds of data.dataset_keys) {
      const curve = data.lookback_curves[ds];
      if (!curve) continue;
      const label = data.dataset_labels[ds];
      const color = data.dataset_colors[ds];
      // Power-law line on log-log axes
      const xFit = [];
      const yFit = [];
      const minH = Math.min.apply(null, curve.horizons);
      const maxH = Math.max.apply(null, curve.horizons);
      for (let lx = Math.log(minH); lx <= Math.log(maxH); lx += 0.05) {
        const x = Math.exp(lx);
        const y = curve.a * Math.pow(x, curve.b);
        xFit.push(x);
        yFit.push(y);
      }
      traces.push({
        x: curve.horizons,
        y: curve.median_lookback,
        type: "scatter",
        mode: "markers",
        name: `${label} (b=${curve.b.toFixed(2)})`,
        marker: { color, size: 6, opacity: 0.7 },
        legendgroup: ds,
        hovertemplate: `${label}<br>H=%{x}<br>L*=%{y:.0f}<extra></extra>`,
      });
      traces.push({
        x: xFit,
        y: yFit,
        type: "scatter",
        mode: "lines",
        name: `${label} fit`,
        showlegend: false,
        line: { color, width: 2 },
        legendgroup: ds,
        hoverinfo: "skip",
      });
    }
    const layout = {
      margin: { l: 56, r: 16, t: 12, b: 48 },
      xaxis: {
        type: "log",
        title: { text: "Forecast horizon H", font: { size: 12 } },
        gridcolor: "#eee",
        zeroline: false,
        dtick: "D2",
      },
      yaxis: {
        type: "log",
        title: { text: "Optimal lookback L*", font: { size: 12 } },
        gridcolor: "#eee",
        zeroline: false,
      },
      legend: {
        orientation: "v",
        x: 1.02,
        y: 1,
        font: { size: 11, family: "ui-sans-serif, system-ui" },
        bgcolor: "rgba(255,255,255,0)",
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      hovermode: "closest",
    };
    Plotly.newPlot(host, traces, layout, { displayModeBar: false, responsive: true });
  }

  function heatmapFig(matrixData, host, opts) {
    if (!matrixData) {
      host.innerHTML = `<p style="padding:14px; font-family: ui-sans-serif; color: #888;">No data</p>`;
      return;
    }
    const z = matrixData.values.map((row) => row.map((v) => (v != null ? log10(Math.max(v, 1e-9)) : null)));
    const trace = {
      type: "heatmap",
      x: matrixData.horizons,
      y: matrixData.series,
      z,
      colorscale: opts.colorscale || "Viridis",
      zmin: opts.zmin,
      zmax: opts.zmax,
      hovertemplate: `series %{y}<br>H=%{x}<br>${opts.label}=%{z:.2f}<extra></extra>`,
      colorbar: {
        title: { text: opts.label, font: { size: 11 } },
        thickness: 12,
        len: 0.9,
        tickfont: { size: 10 },
      },
    };
    const layout = {
      title: { text: opts.title, font: { size: 13 }, x: 0.04, y: 0.97, xanchor: "left" },
      margin: { l: 130, r: 16, t: 32, b: 44 },
      xaxis: { title: { text: "H", font: { size: 11 } }, type: "category" },
      yaxis: {
        title: { text: "series", font: { size: 11 }, standoff: 4 },
        type: "category",
        automargin: true,
        tickfont: { size: 9.5 },
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
    };
    Plotly.newPlot(host, [trace], layout, { displayModeBar: false, responsive: true });
  }

  function paretoFig(data, host) {
    const traces = [];
    for (const ds of data.dataset_keys) {
      const p = data.horizon_pareto[ds];
      if (!p) continue;
      traces.push({
        x: p.group_sizes,
        y: p.degradation_pct,
        type: "scatter",
        mode: "lines+markers",
        name: data.dataset_labels[ds],
        line: { color: data.dataset_colors[ds], width: 2 },
        marker: { size: 6 },
        hovertemplate: `${data.dataset_labels[ds]}<br>g<sub>h</sub>=%{x}<br>+%{y:.2f}% MSE<extra></extra>`,
      });
    }
    const layout = {
      margin: { l: 60, r: 16, t: 12, b: 48 },
      xaxis: {
        title: { text: "Horizon group size g<sub>h</sub>", font: { size: 12 } },
        type: "category",
        gridcolor: "#eee",
      },
      yaxis: {
        title: { text: "MSE degradation (%)", font: { size: 12 } },
        gridcolor: "#eee",
        zeroline: true,
        zerolinecolor: "#888",
        zerolinewidth: 1,
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "v", x: 1.02, y: 1, font: { size: 11 } },
    };
    Plotly.newPlot(host, traces, layout, { displayModeBar: false, responsive: true });
  }

  function augFig(data, host) {
    const labels = [];
    const none = [];
    const time = [];
    const freq = [];
    const colors = [];
    for (const ds of data.dataset_keys) {
      const a = data.aug_selection[ds];
      if (!a) continue;
      labels.push(data.dataset_labels[ds]);
      none.push(a.selection.none * 100);
      time.push(a.selection.time * 100);
      freq.push(a.selection.freq * 100);
      colors.push(data.dataset_colors[ds]);
    }
    const traces = [
      { x: labels, y: none, name: "none", type: "bar", marker: { color: "#bdbdbd" } },
      { x: labels, y: time, name: "time-domain noise", type: "bar", marker: { color: "#6baed6" } },
      { x: labels, y: freq, name: "freq-domain noise", type: "bar", marker: { color: "#fd8d3c" } },
    ];
    const layout = {
      barmode: "stack",
      margin: { l: 60, r: 16, t: 12, b: 60 },
      xaxis: {
        title: { text: "", font: { size: 12 } },
        tickangle: -25,
      },
      yaxis: {
        title: { text: "Selection rate (%)", font: { size: 12 } },
        range: [0, 100],
        gridcolor: "#eee",
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "h", y: -0.32, x: 0, font: { size: 11 } },
    };
    Plotly.newPlot(host, traces, layout, { displayModeBar: false, responsive: true });
  }

  function mainResultsTable(data, host) {
    const t = data.main_results;
    if (!t || !t.datasets) { host.innerHTML = ""; return; }
    const headers = ["Ours"].concat(t.linear_models.slice(1)).concat(t.nonlinear_models);
    const nLin = t.linear_models.length;
    let html = `<table class="main-results">
      <thead>
        <tr>
          <th class="sticky-left"></th><th class="sticky-left"></th>
          <th colspan="${nLin}" class="group group-linear">Linear models</th>
          <th colspan="${t.nonlinear_models.length}" class="group group-nonlinear">Nonlinear models</th>
        </tr>
        <tr>
          <th class="sticky-left">Dataset</th><th class="sticky-left">H</th>
          ${headers.map((h, i) => `<th class="${i === 0 ? "ours" : ""}">${h}</th>`).join("")}
        </tr>
      </thead><tbody>`;
    for (const ds of t.datasets) {
      const ddRows = ds.rows;
      // For best / second-best, examine all numeric rows EXCEPT the Avg row separately.
      ddRows.forEach((row, ri) => {
        // Mark best and 2nd-best in this row
        const vals = row.values;
        const sortedIdx = vals.map((v, i) => ({ v, i }))
          .sort((a, b) => a.v - b.v);
        const bestVal = sortedIdx[0].v;
        const secondVal = sortedIdx.find((s) => s.v > bestVal)?.v ?? null;
        const isAvg = row.H === "Avg";
        html += `<tr class="${isAvg ? "avg-row" : ""}">`;
        if (ri === 0) {
          html += `<td class="ds-cell sticky-left" rowspan="${ddRows.length}">${ds.name}</td>`;
        }
        html += `<td class="h-cell sticky-left">${row.H}</td>`;
        vals.forEach((v, i) => {
          const isBest = Math.abs(v - bestVal) < 1e-9;
          const isSecond = !isBest && secondVal != null && Math.abs(v - secondVal) < 1e-9;
          const cls = [
            "num",
            i === 0 ? "ours" : "",
            isBest ? "best" : (isSecond ? "second" : ""),
            isAvg ? "italic" : "",
          ].filter(Boolean).join(" ");
          html += `<td class="${cls}">${v.toFixed(3)}</td>`;
        });
        html += `</tr>`;
      });
    }
    html += `</tbody></table>`;
    host.innerHTML = html;
  }

  function benchmarkTable(data, host) {
    const t = data.benchmark_table;
    if (!t || !t.rows || t.rows.length === 0) {
      host.innerHTML = "";
      return;
    }
    // Group by dataset
    const datasets = {};
    for (const r of t.rows) {
      if (!datasets[r.dataset]) datasets[r.dataset] = [];
      datasets[r.dataset].push(r);
    }
    const cutoffs = Array.from(new Set(t.rows.map((r) => r.cutoff))).sort((a, b) => a - b);
    let html = `<table class="bench-table">
      <thead>
        <tr><th>Dataset</th><th>Scope</th>${cutoffs.map((c) => `<th>H=${c}</th>`).join("")}</tr>
      </thead><tbody>`;
    for (const [ds, rows] of Object.entries(datasets)) {
      const byH = Object.fromEntries(rows.map((r) => [r.cutoff, r]));
      for (const scope of ["global_mse", "local_mse", "search_mse"]) {
        const scopeLabel = { global_mse: "global", local_mse: "local", search_mse: "search (ours)" }[scope];
        html += `<tr>`;
        if (scope === "global_mse") {
          html += `<td class="ds-cell" rowspan="3">${ds}</td>`;
        }
        html += `<td class="scope-cell">${scopeLabel}</td>`;
        for (const c of cutoffs) {
          const r = byH[c];
          if (!r) {
            html += `<td>—</td>`;
            continue;
          }
          const v = r[scope];
          // Bold the search row (best by construction)
          const isBest = scope === "search_mse";
          html += `<td class="num${isBest ? " best" : ""}">${v.toFixed(3)}</td>`;
        }
        html += `</tr>`;
      }
    }
    html += "</tbody></table>";
    host.innerHTML = html;
  }

  function forecastFig(data, host) {
    const f = data.forecast;
    if (!f || !f.gt || f.gt.length === 0) {
      host.innerHTML = `<p style="padding:14px; color:#888; font-family: ui-sans-serif;">forecast data unavailable</p>`;
      return;
    }
    const n = f.gt.length;
    const x = Array.from({ length: n }, (_, i) => i + 1);
    const traces = [
      { x, y: f.gt, type: "scatter", mode: "lines", name: "ground truth",
        line: { color: "#1a1a1a", width: 1.8 } },
      { x, y: f.local_pred, type: "scatter", mode: "lines", name: "tuned (local)",
        line: { color: "#0c6e6e", width: 2.2 } },
      { x, y: f.global_pred, type: "scatter", mode: "lines", name: "baseline (global)",
        line: { color: "#b53a1a", width: 1.6, dash: "dot" } },
    ];
    const layout = {
      margin: { l: 50, r: 16, t: 12, b: 48 },
      xaxis: { title: { text: "forecast step", font: { size: 12 } }, gridcolor: "#eee" },
      yaxis: { title: { text: "value (scaled)", font: { size: 12 } }, gridcolor: "#eee" },
      legend: { orientation: "h", x: 0, y: 1.12, font: { size: 12 } },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      annotations: [{
        text: `ETTh1 · series ${f.series}`,
        xref: "paper", yref: "paper", x: 1.0, y: -0.18,
        xanchor: "right", showarrow: false,
        font: { size: 11, color: "#999", family: "ui-monospace, monospace" },
      }],
    };
    Plotly.newPlot(host, traces, layout, { displayModeBar: false, responsive: true });
  }

  function init(data) {
    const lookbackHost = document.getElementById("fig-lookback");
    if (lookbackHost) lookbackFig(data, lookbackHost);

    heatmapFig(data.hp_heatmaps.etth1_r, document.getElementById("fig-hm-etth1-r"), {
      title: "ETTh1 · log₁₀ r",
      label: "log₁₀ r",
      colorscale: "Cividis",
      zmin: -3, zmax: 0,
    });
    heatmapFig(data.hp_heatmaps.weather_r, document.getElementById("fig-hm-weather-r"), {
      title: "Weather · log₁₀ r",
      label: "log₁₀ r",
      colorscale: "Cividis",
      zmin: -3, zmax: 0,
    });
    heatmapFig(data.hp_heatmaps.etth1_alpha, document.getElementById("fig-hm-etth1-a"), {
      title: "ETTh1 · log₁₀ α",
      label: "log₁₀ α",
      colorscale: "Viridis",
      zmin: -2, zmax: 6,
    });
    heatmapFig(data.hp_heatmaps.weather_alpha, document.getElementById("fig-hm-weather-a"), {
      title: "Weather · log₁₀ α",
      label: "log₁₀ α",
      colorscale: "Viridis",
      zmin: -2, zmax: 6,
    });

    const paretoHost = document.getElementById("fig-pareto");
    if (paretoHost) paretoFig(data, paretoHost);

    const augHost = document.getElementById("fig-aug");
    if (augHost) augFig(data, augHost);

    const forecastHost = document.getElementById("fig-forecast");
    if (forecastHost) forecastFig(data, forecastHost);

    const tableHost = document.getElementById("fig-benchmark");
    if (tableHost) benchmarkTable(data, tableHost);

    const mainHost = document.getElementById("fig-main-results");
    if (mainHost) mainResultsTable(data, mainHost);
  }

  function start() {
    const data = window.__FIGURES_DATA;
    if (!data) {
      console.error("figures: window.__FIGURES_DATA not loaded. Include assets/figures_data.js before this script.");
      return;
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => init(data));
    } else {
      init(data);
    }
  }
  start();
})();
