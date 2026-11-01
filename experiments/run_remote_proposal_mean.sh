#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

"$PY" posterior_sample.py \
  +data=test-realblur \
  +kernel_data=nonlinear \
  +model=ffhq256ddpm \
  +task=blind_nonlinear_blur \
  +sampler=edm_daps \
  task_group=pixel \
  save_dir=tuning_results \
  name=proposal_c4_scale0 \
  seed=42 \
  num_runs=1 \
  save_samples=false \
  batch_size=4 \
  data.start_id=0 \
  data.end_id=4 \
  sampler.blind=False \
  sampler.nonlinear=True \
  +task.pixel.operator.proposal_candidates=4 \
  +task.pixel.operator.proposal_noise_scale=0.0 \
  sampler.annealing_scheduler_config.num_steps=200 \
  sampler.diffusion_scheduler_config.num_steps=5 \
  gpu=0 \
  > tuning_logs/proposal_c4_scale0.log 2>&1

echo "PROPOSAL_MEAN_DONE"
