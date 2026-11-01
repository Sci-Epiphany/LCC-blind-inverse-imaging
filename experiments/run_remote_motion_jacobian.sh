#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

LR_K=${1:-3.33e-7}
GPU=${2:-0}
IMAGES=${3:-4}
OUTER=${4:-50}
MCMC=${5:-20}
SEED=${6:-42}
NOISE_SCALE=${7:-0.1}
TAG=${8:-motion}
NAME="motion_jac_${TAG}_lk${LR_K}_q${NOISE_SCALE}_n${IMAGES}_o${OUTER}_m${MCMC}_s${SEED}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

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
  task.pixel.mcmc_sampler_config.lr.image=3.92e-5 \
  task.pixel.mcmc_sampler_config.lr.kernel="$LR_K" \
  task.pixel.mcmc_sampler_config.reg_terms.lamb=35 \
  task.pixel.mcmc_sampler_config.reg_terms.beta=20000000 \
  task.pixel.mcmc_sampler_config.tau=null \
  +task.pixel.mcmc_sampler_config.kernel_regularizer=pnp \
  +task.pixel.mcmc_sampler_config.kernel_noise_scale="$NOISE_SCALE" \
  +fixed_measurement_dir="$ROOT"/archived_inputs/motion/measurements \
  +fixed_kernel_dir="$ROOT"/tuning_results/paper_full_motion_pnp_seed42_runs4/kernel_gts \
  gpu=0 \
  > "tuning_logs/${NAME}.log" 2>&1

echo "$NAME"
