#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

"$PY" remote_evaluate_fixed_pnp_ensemble.py \
  --samples tuning_results/paper_full_gaussian_pnp_seed42_runs4/samples \
  --images 100 \
  --label gaussian_pnp_fixed \
  --output ensemble_fixed_gaussian \
  > tuning_logs/ensemble_fixed_gaussian.log 2>&1

echo "GAUSSIAN_ENSEMBLE_DONE"
