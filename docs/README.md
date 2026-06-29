# Project website — Time Series Hyperparameter Optimization

Static site for the paper *How Good Can Linear Models Be for Time-Series
Forecasting?*

## Layout

```
website/
├── index.html              # The article (single scroll)
├── app/                    # Live Ridge playground (Widget B)
│   ├── index.html
│   ├── demo.js
│   ├── ridge.js            # Ridge solver / scalers / augmentation kernels
│   ├── ridge.worker.js     # Web Worker entry
│   └── data/
│       ├── ETTh1.csv       # Full ETTh1, the only bundled dataset
│       └── manifest.json   # Searched-optimal HPs + naive baseline
├── styles/
│   ├── main.css
│   └── demo.css
├── scripts/
│   ├── widget_a.js         # Inline lookback-span explorer
│   └── vendor/
│       ├── plotly.min.js
│       └── papaparse.min.js
├── assets/
│   ├── figures/            # Curated PNGs (rasterized from figures/*.pdf)
│   └── widget_a_data.json  # Per-dataset snippets + L*(H) for Widget A
└── build_data.py           # Regenerates app/data/ + assets/widget_a_data.json
```

## Local preview

```bash
cd website && python -m http.server 8080
# then visit http://localhost:8080/
```

The playground is at `http://localhost:8080/app/`.

## Regenerating data assets

After updating the underlying experiments in `../exps/`, regenerate the
bundled dataset, the manifest, and Widget A data:

```bash
python website/build_data.py
```

This reads:

- `../data/ETTh1.csv` (bundled CSV)
- `../exps/exp3_*_gs1/local_results.csv` and `../exps/exp8_*_sgs1_pooled/local_results.csv`
  (for searched-optimal HPs and L*(H))

## Browser support

- Modern Chrome / Safari / Firefox. The playground uses an ES-module Web
  Worker (`new Worker(url, {type: 'module'})`), which is supported in
  Chrome 80+, Safari 15+, Firefox 114+.
- No build step, no framework.

## Deploying to `pub.sakana.ai/ts-param-search/`

The site uses only relative paths, so dropping the `website/` directory
under any subpath should work without changes. The biggest assets are
Plotly (~4.4 MB), the bundled ETTh1 CSV (~2.5 MB), and the curated PNGs
(~4 MB). Total bundle size is around 14 MB.
