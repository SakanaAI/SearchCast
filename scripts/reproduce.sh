#!/bin/bash
# Reproduce the paper's main hyperparameter search (series-grouping sweep).
#
# For each dataset we sweep `local_series_group_size` (sgs) over the divisors of
# its series count and run the canonical configuration. The best sgs is the one
# minimizing mean Local-MSE across the four cutoffs in benchmark_comparison.csv.
#
# Usage:
#   scripts/reproduce.sh                 # run every dataset
#   scripts/reproduce.sh etth1 weather   # run a subset
#
# Results are written to exps/exp8_<dataset>_sgs<N>_pooled/.
set -e

# Each entry: <key>:<csv_path>:<space-separated sgs sweep>
# sgs values are the divisors of each dataset's series count.
CONFIGS=(
  "etth1:data/ETTh1.csv:1 3 7"
  "etth2:data/ETTh2.csv:1 3 7"
  "ettm1:data/ETTm1.csv:1 3 7"
  "ettm2:data/ETTm2.csv:1 3 7"
  "exchange:data/exchange_rate.csv:1 2 4 8"
  "weather:data/weather.csv:1 3 7 11 21"
)

run_dataset() {
  local csv="$1"; local name="$2"; shift 2
  for sgs in "$@"; do
    echo "=== ${name}: series_group_size=${sgs} ==="
    python -u optuna_ridge.py \
      --input_csv "${csv}" \
      --output_dir "exps/exp8_${name}_sgs${sgs}_pooled" \
      --scaler_scope local --scaler_method mean \
      --local_horizon_group_size 24 \
      --local_series_group_size "${sgs}" \
      --instance_norm \
      --n_folds 3 \
      --pool_series \
      --n_trials 20
  done
}

wanted=("$@")  # optional dataset-name filter

for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r name csv sweep <<< "${entry}"
  if [ ${#wanted[@]} -gt 0 ]; then
    match=0
    for w in "${wanted[@]}"; do [ "${w}" = "${name}" ] && match=1; done
    [ "${match}" -eq 1 ] || continue
  fi
  # shellcheck disable=SC2086
  run_dataset "${csv}" "${name}" ${sweep}
done

echo "Done. Best sgs per dataset = the run with lowest mean Local MSE in benchmark_comparison.csv."
