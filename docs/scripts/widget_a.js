/**
 * Lookback-context widget — colored background bar style.
 *
 * Each panel renders the time series in a uniform dark stroke, with a
 * COLORED HORIZONTAL BAR painted in the background spanning [anchor - L(H),
 * anchor]. The bar's color is mapped from H via a turbo-like ramp (cool =
 * short H, warm = long H). The visualization echoes the paper's teaser
 * panel (a).
 *
 * Data is read from window.__WIDGET_A_DATA (loaded via a <script> tag, so
 * the widget works on file:// without fetch / CORS).
 *
 * Two layouts via data-layout="multi" (2x4 small multiples) or "single".
 * Autoplay starts unconditionally if data-autoplay is not "false".
 */

(function () {
  "use strict";

  // ---- turbo-like colormap -----------------------------------------------
  const TURBO_STOPS = [
    [0.00, [48, 18, 59]],
    [0.20, [70, 105, 232]],
    [0.40, [60, 205, 220]],
    [0.55, [115, 230, 90]],
    [0.70, [240, 220, 60]],
    [0.85, [240, 130, 50]],
    [1.00, [180, 30, 30]],
  ];
  function horizonColor(H, alpha) {
    const t = Math.max(0, Math.min(1, (H - 24) / (900 - 24)));
    let i = 0;
    while (i < TURBO_STOPS.length - 1 && TURBO_STOPS[i + 1][0] < t) i++;
    const [t0, c0] = TURBO_STOPS[i];
    const [t1, c1] = TURBO_STOPS[i + 1];
    const u = (t - t0) / Math.max(t1 - t0, 1e-9);
    const r = Math.round(c0[0] + (c1[0] - c0[0]) * u);
    const g = Math.round(c0[1] + (c1[1] - c0[1]) * u);
    const b = Math.round(c0[2] + (c1[2] - c0[2]) * u);
    return alpha == null
      ? `rgb(${r}, ${g}, ${b})`
      : `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  function snapHorizon(H, available) {
    if (!available || available.length === 0) return null;
    let best = available[0];
    let bestDiff = Math.abs(H - best);
    for (const h of available) {
      const d = Math.abs(H - h);
      if (d < bestDiff) {
        bestDiff = d;
        best = h;
      }
    }
    return best;
  }

  function renderPanel(svg, x, y, w, h, ds, H, opts) {
    opts = opts || {};
    const snippet = ds.snippet;
    const n = snippet.length;
    if (n < 2) return;

    // Choose a visible window that fits the largest L*(H) on the left and the
    // largest H on the right, so neither bar gets clipped at any H.
    const maxL = Math.max(0, ...Object.values(ds.lookback_at_h || {}).map((v) => parseInt(v, 10)));
    const maxH = Math.max(720, ...Object.keys(ds.lookback_at_h || {}).map(Number));
    const pastSpan = Math.max(maxL + 20, 440);
    const futureSpan = maxH + 20;
    const visibleSpan = pastSpan + futureSpan;
    let displayStart = Math.max(0, n - visibleSpan - 10);
    const displayEnd = Math.min(n, displayStart + visibleSpan);
    displayStart = Math.max(0, displayEnd - visibleSpan);
    const totalShown = displayEnd - displayStart;
    const anchorIdxAbs = displayStart + pastSpan;

    function sx(absIdx) {
      const rel = (absIdx - displayStart) / Math.max(totalShown - 1, 1);
      return x + rel * w;
    }

    let ymin = Infinity, ymax = -Infinity;
    for (let i = displayStart; i < displayEnd; i++) {
      const v = snippet[i];
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    }
    const yrange = Math.max(ymax - ymin, 1e-6);
    const ypadFrac = 0.18;
    const yLo = ymin - ypadFrac * yrange;
    const yHi = ymax + ypadFrac * yrange;
    function sy(v) {
      return y + (1 - (v - yLo) / (yHi - yLo)) * h;
    }

    const xAnchor = sx(anchorIdxAbs);

    // Panel frame
    svg.appendChild(svgEl("rect", {
      x, y, width: w, height: h,
      fill: "white", stroke: "#e5e3dc", "stroke-width": "1",
    }));

    // === Slice-driven future bands + active L* bar =========================
    // Each slice = one horizon group searched with its own L*. We draw every
    // slice's future band; the active slice (whose [start, end] brackets H)
    // is opaque, the rest are translucent.
    const slices = Array.isArray(ds.slices) ? ds.slices : [];
    const drawPredictions = opts.drawPredictions === true && slices.length > 0;

    let activeSlice = null;
    let activeIdx = -1;
    if (slices.length > 0) {
      let bestDist = Infinity;
      slices.forEach((s, i) => {
        if (s.start <= H && H <= s.end) {
          activeSlice = s;
          activeIdx = i;
        }
        const d = Math.abs(H - s.end);
        if (d < bestDist) { bestDist = d; if (activeSlice == null) { activeSlice = s; activeIdx = i; } }
      });
    }

    // Fallback to legacy snap when slices aren't available.
    let snappedH, L;
    if (activeSlice) {
      snappedH = activeSlice.end;
      L = activeSlice.L;
    } else {
      const availH = Object.keys(ds.lookback_at_h || {}).map(Number).sort((a, b) => a - b);
      snappedH = snapHorizon(H, availH);
      L = snappedH != null ? parseInt(ds.lookback_at_h[String(snappedH)], 10) : null;
    }

    // L* bar on the left of the anchor (single bar for the active slice only).
    if (L != null && L > 0) {
      const xSpanLeft = sx(anchorIdxAbs - L);
      svg.appendChild(svgEl("rect", {
        x: xSpanLeft,
        y: y + 1,
        width: Math.max(xAnchor - xSpanLeft, 1),
        height: h - 2,
        fill: horizonColor(snappedH, 0.38),
        stroke: horizonColor(snappedH, 0.9),
        "stroke-width": "1.2",
      }));
    }

    // Per-slice dashed bands to the right of the anchor.
    let activeFutureWidth = 0;
    if (slices.length > 0) {
      slices.forEach((s, i) => {
        const isActive = i === activeIdx;
        const op = isActive ? 1.0 : 0.25;
        const xL = sx(anchorIdxAbs + s.start - 1);
        const xR = sx(Math.min(anchorIdxAbs + s.end, displayEnd - 1));
        const w_ = xR - xL;
        if (w_ < 1) return;
        svg.appendChild(svgEl("rect", {
          x: xL, y: y + 1,
          width: w_, height: h - 2,
          fill: horizonColor(s.end, 0.18 * op),
          stroke: horizonColor(s.end, 0.7 * op),
          "stroke-width": "1.0",
          "stroke-dasharray": "4 3",
        }));
        if (isActive) activeFutureWidth = w_;
      });
    } else if (snappedH != null) {
      // Legacy: single cumulative band when there's no slice data.
      const futureWidth = sx(Math.min(anchorIdxAbs + snappedH, displayEnd - 1)) - xAnchor;
      if (futureWidth > 1) {
        svg.appendChild(svgEl("rect", {
          x: xAnchor, y: y + 1,
          width: futureWidth, height: h - 2,
          fill: horizonColor(snappedH, 0.18),
          stroke: horizonColor(snappedH, 0.7),
          "stroke-width": "1.0",
          "stroke-dasharray": "4 3",
        }));
        activeFutureWidth = futureWidth;
      }
    }

    // === TIME SERIES (uniform dark stroke on top of the colored bars) =====
    let pathFull = "";
    for (let i = displayStart; i < displayEnd; i++) {
      pathFull += (i === displayStart ? "M" : "L") +
        sx(i).toFixed(2) + " " + sy(snippet[i]).toFixed(2) + " ";
    }
    svg.appendChild(svgEl("path", {
      d: pathFull,
      fill: "none",
      stroke: "#1a1a1a",
      "stroke-width": "1.2",
      "stroke-linejoin": "round",
    }));

    // === Prediction lines per slice (single layout only) ===================
    if (drawPredictions) {
      slices.forEach((s, i) => {
        if (!Array.isArray(s.pred) || s.pred.length === 0) return;
        const isActive = i === activeIdx;
        let pathPred = "";
        let moved = false;
        for (let j = 0; j < s.pred.length; j++) {
          const absIdx = anchorIdxAbs + s.start - 1 + j;
          if (absIdx >= displayEnd) break;
          const xj = sx(absIdx);
          const yj = sy(s.pred[j]);
          pathPred += (moved ? "L" : "M") + xj.toFixed(2) + " " + yj.toFixed(2) + " ";
          moved = true;
        }
        svg.appendChild(svgEl("path", {
          d: pathPred,
          fill: "none",
          stroke: horizonColor(s.end, 1.0),
          "stroke-width": isActive ? "2.0" : "1.2",
          "stroke-linejoin": "round",
          "stroke-opacity": String(isActive ? 1.0 : 0.35),
        }));
      });
    }

    // === Anchor (prediction-start) dashed line ============================
    svg.appendChild(svgEl("line", {
      x1: xAnchor, x2: xAnchor,
      y1: y + 4, y2: y + h - 4,
      stroke: "#555",
      "stroke-width": "0.9",
      "stroke-dasharray": "3 3",
    }));

    // Title (top-left)
    const title = svgEl("text", {
      x: x + 7, y: y + 14,
      "font-family": "ui-sans-serif, system-ui, sans-serif",
      "font-size": opts.smallTitle ? "11" : "13",
      "font-weight": "600",
      fill: ds.color || "#1a1a1a",
    });
    title.textContent = ds.label;
    svg.appendChild(title);

    // L* label (left of anchor) and H label (right of anchor). In the
    // small-multiples (multi) layout, drop the labels to the panel floor so they
    // clear the top-left dataset title and stay off the curve (the series has
    // headroom below its minimum). The single-panel layout keeps them up top.
    const labelFont = opts.smallTitle ? "10.5" : "11";
    const labelColor = horizonColor(snappedH, 1.0);
    const labelY = opts.smallTitle ? (y + h - 6) : (y + 14);
    if (L != null && L > 0) {
      const lLbl = svgEl("text", {
        x: xAnchor - 4, y: labelY,
        "text-anchor": "end",
        "font-family": "ui-monospace, monospace",
        "font-size": labelFont,
        fill: labelColor,
        "font-weight": "600",
      });
      lLbl.textContent = `L⋆ = ${L}`;
      svg.appendChild(lLbl);
    }
    if (snappedH != null && activeFutureWidth > 1) {
      const hLbl = svgEl("text", {
        x: xAnchor + 4, y: labelY,
        "text-anchor": "start",
        "font-family": "ui-monospace, monospace",
        "font-size": labelFont,
        fill: labelColor,
        "font-weight": "600",
      });
      hLbl.textContent = `H = ${snappedH}`;
      svg.appendChild(hLbl);
    }
  }

  function setupWidget(host, data) {
    const layout = host.dataset.layout || "multi";
    const autoplay = host.dataset.autoplay !== "false";

    host.classList.add("la-widget");
    host.innerHTML = "";

    if (layout === "single") {
      const chipsHost = document.createElement("div");
      chipsHost.className = "la-chips";
      host.appendChild(chipsHost);
      data.datasets.forEach((ds, i) => {
        const b = document.createElement("button");
        b.className = "la-chip" + (i === 0 ? " active" : "");
        b.type = "button";
        b.textContent = ds.label;
        b.style.setProperty("--chip-color", ds.color || "#0c6e6e");
        b.addEventListener("click", () => {
          chipsHost.querySelectorAll(".la-chip").forEach((el, j) => el.classList.toggle("active", j === i));
          host.activeIdx = i;
          draw();
        });
        chipsHost.appendChild(b);
      });
    }

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.classList.add("la-plot");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    host.appendChild(svg);

    const controls = document.createElement("div");
    controls.className = "la-controls";
    controls.innerHTML = `
      <button type="button" class="la-play">❚❚ Pause</button>
      <input class="la-slider" type="range" min="24" max="720" step="24" value="192">
      <span class="la-readout">H = 192</span>
    `;
    host.appendChild(controls);

    const playBtn = controls.querySelector(".la-play");
    const slider = controls.querySelector(".la-slider");
    const readout = controls.querySelector(".la-readout");

    // Paint the slider track with the same turbo ramp used for horizon colors,
    // so the track itself acts as the colormap legend. We anchor the gradient
    // at the turbo stops that fall inside the slider's H range so the CSS
    // interpolation matches horizonColor() exactly between anchors.
    (function paintTrack() {
      const minH = Number(slider.min);
      const maxH = Number(slider.max);
      const stops = [[0, horizonColor(minH)]];
      for (const [tA] of TURBO_STOPS) {
        const HA = 24 + (900 - 24) * tA;
        if (HA <= minH || HA >= maxH) continue;
        const p = (HA - minH) / (maxH - minH);
        stops.push([p, horizonColor(HA)]);
      }
      stops.push([1, horizonColor(maxH)]);
      slider.style.background =
        "linear-gradient(to right, " +
        stops.map(([p, c]) => `${c} ${(p * 100).toFixed(2)}%`).join(", ") +
        ")";
    })();

    host.activeIdx = 0;
    host.currentH = 192;
    host.dirSign = 1;
    host.playing = false;

    function draw() {
      let cols, rows, panelW, panelH, viewW, viewH;
      if (layout === "multi") {
        cols = 4;
        rows = 2;
        panelW = 215;
        panelH = 120;
        const padX = 8;
        const padTop = 20;     // room for the H/legend title text
        const padInner = 14;   // between rows
        const padBottom = 6;
        viewW = cols * panelW + (cols + 1) * padX;
        viewH = padTop + rows * panelH + (rows - 1) * padInner + padBottom;
        svg.setAttribute("viewBox", `0 0 ${viewW} ${viewH}`);
        svg.style.height = "308px";
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        data.datasets.forEach((ds, i) => {
          const r = Math.floor(i / cols);
          const c = i % cols;
          const px = padX + c * (panelW + padX);
          const py = padTop + r * (panelH + padInner);
          renderPanel(svg, px, py, panelW, panelH, ds, host.currentH, { smallTitle: true, drawPredictions: false });
        });
        const t = svgEl("text", {
          x: viewW / 2, y: 13, "text-anchor": "middle",
          "font-family": "ui-sans-serif, system-ui, sans-serif",
          "font-size": "13", "font-weight": "600", fill: "#1a1a1a",
        });
        t.textContent = `Forecast horizon H = ${host.currentH}  ·  colored bar width = searched optimal lookback L⋆(H)`;
        svg.appendChild(t);
      } else {
        viewW = 880;
        viewH = 260;
        svg.setAttribute("viewBox", `0 0 ${viewW} ${viewH}`);
        svg.style.height = "280px";
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        const ds = data.datasets[host.activeIdx];
        renderPanel(svg, 20, 30, viewW - 40, viewH - 60, ds, host.currentH, { drawPredictions: true });
      }
      readout.textContent = `H = ${host.currentH}`;
    }

    // === Animation (setInterval; no IntersectionObserver). ===============
    const STEP = 24;
    const FRAME_MS = 220;
    let timer = null;

    function tick() {
      if (!host.playing) return;
      let nextH = host.currentH + host.dirSign * STEP;
      if (nextH > 720) { nextH = 720; host.dirSign = -1; }
      if (nextH < 24)  { nextH = 24;  host.dirSign = 1; }
      host.currentH = nextH;
      slider.value = host.currentH;
      try { draw(); } catch (e) { console.error("widget draw error", e); }
    }
    function play() {
      host.playing = true;
      playBtn.textContent = "❚❚ Pause";
      if (timer) clearInterval(timer);
      timer = setInterval(tick, FRAME_MS);
    }
    function pause() {
      host.playing = false;
      playBtn.textContent = "▶ Play";
      if (timer) { clearInterval(timer); timer = null; }
    }

    playBtn.addEventListener("click", () => (host.playing ? pause() : play()));
    slider.addEventListener("input", (e) => {
      pause();
      host.currentH = parseInt(e.target.value, 10);
      draw();
    });

    draw();
    if (autoplay) play();
  }

  function init(data) {
    const hosts = document.querySelectorAll("[data-widget=lookback-span]");
    hosts.forEach((host) => setupWidget(host, data));
  }

  function start() {
    const data = window.__WIDGET_A_DATA;
    if (!data || !data.datasets) {
      console.error("widget_a: window.__WIDGET_A_DATA not loaded. Include assets/widget_a_data.js before this script.");
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
