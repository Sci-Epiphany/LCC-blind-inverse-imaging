#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

GPU=${1:-0}
IMAGES=${2:-4}
OUTER=${3:-50}
MCMC=${4:-20}
SEED=${5:-42}
TRUST=${6:-0.01}
TAG=${7:-adaptive}
NAME="kernel_adapt_${TAG}_t${TRUST}_n${IMAGES}_o${OUTER}_m${MCMC}_s${SEED}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

"$PY" posterior_sample_fixed_input.py \
  +data=test-ffhq \
  +kernel_data=gaussian64 \
  +model=ffhq256ddpm \
  +kernel_model=kernel64ddpm \
  +task=blind_gaussian_blur \
  +sampler=edm_daps \
  task_group=pixel \
  save_dir=tuning_results \
  name="$NAME" \
  seed="$SEED" \
  num_runs=1 \
  save_samples=true \
  batch_size=8 \
  data.start_id=0 \
  data.end_id="$IMAGES" \
  sampler.blind=True \
  sampler.nonlinear=False \
  sampler.annealing_scheduler_config.num_steps="$OUTER" \
  sampler.diffusion_scheduler_config.num_steps=5 \
  task.pixel.mcmc_sampler_config.num_steps="$MCMC" \
  task.pixel.mcmc_sampler_config.lr.image=5e-5 \
  task.pixel.mcmc_sampler_config.lr.kernel=1e-9 \
  task.pixel.mcmc_sampler_config.reg_terms.lamb=20 \
  task.pixel.mcmc_sampler_config.reg_terms.beta=2000000 \
  task.pixel.mcmc_sampler_config.tau=null \
  +task.pixel.mcmc_sampler_config.kernel_regularizer=pnp \
  +task.pixel.mcmc_sampler_config.kernel_step_mode=trust_region \
  +task.pixel.mcmc_sampler_config.kernel_trust_fraction="$TRUST" \
  +task.pixel.mcmc_sampler_config.kernel_line_search_steps=5 \
  +fixed_measurement_dir="$ROOT"/archived_inputs/gaussian/measurements \
  +fixed_kernel_dir="$ROOT"/archived_inputs/gaussian/kernel_gts \
  gpu=0 \
  > "tuning_logs/${NAME}.log" 2>&1

echo "$NAME"
