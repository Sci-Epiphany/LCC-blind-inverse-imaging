#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

"$PY" remote_evaluate_fixed_pnp_ensemble.py \
  --samples tuning_results/paper_full_motion_pnp_seed42_runs4/samples \
  --images 200 \
  --label motion_pnp_fixed \
  --output ensemble_fixed_motion \
  > tuning_logs/ensemble_fixed_motion.log 2>&1

echo "MOTION_ENSEMBLE_DONE"
