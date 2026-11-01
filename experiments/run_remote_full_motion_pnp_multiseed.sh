#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

NAME=paper_full_motion_pnp_seed42_runs4
echo "START ${NAME}"
"$PY" posterior_sample_fixed_input.py \
  +data=test-ffhq \
  +kernel_data=motion64 \
  +model=ffhq256ddpm \
  +kernel_model=kernel64ddpm \
  +task=blind_motion_blur \
  +sampler=edm_daps \
  task_group=pixel \
  save_dir=tuning_results \
  name="$NAME" \
  seed=42 \
  num_runs=4 \
  save_samples=true \
  batch_size=32 \
  data.start_id=0 \
  data.end_id=200 \
  sampler.blind=True \
  sampler.nonlinear=False \
  sampler.annealing_scheduler_config.num_steps=200 \
  sampler.diffusion_scheduler_config.num_steps=5 \
  task.pixel.mcmc_sampler_config.num_steps=100 \
  task.pixel.mcmc_sampler_config.lr.image=3.92e-5 \
  task.pixel.mcmc_sampler_config.lr.kernel=3.33e-9 \
  task.pixel.mcmc_sampler_config.reg_terms.lamb=35 \
  task.pixel.mcmc_sampler_config.reg_terms.beta=20000000 \
  task.pixel.mcmc_sampler_config.tau=0.01 \
  +task.pixel.mcmc_sampler_config.kernel_regularizer=pnp \
  +fixed_measurement_dir=/home/gsn/PhD_thesis/DAPS/archived_inputs/motion/measurements \
  gpu=0 \
  > "tuning_logs/${NAME}.log" 2>&1
echo "FINISH ${NAME}"

echo "FULL_MOTION_PNP_SEED42_RUNS4_DONE"
