#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

LR_K=${1:?kernel learning rate}
LAMB=${2:?HQS lambda}
BETA=${3:?HQS beta}
TAU=${4:?measurement scale}
GPU=${5:-0}
IMAGES=${6:-4}
OUTER=${7:-50}
MCMC=${8:-20}
TAG=${9:-ablation}
NOISE_SCALE=${10:-1}
SEED=${11:-42}
NAME="kernel_ab_${TAG}_lk${LR_K}_l${LAMB}_b${BETA}_t${TAU}_q${NOISE_SCALE}_n${IMAGES}_o${OUTER}_m${MCMC}"

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
  task.pixel.mcmc_sampler_config.lr.kernel="$LR_K" \
  task.pixel.mcmc_sampler_config.reg_terms.lamb="$LAMB" \
  task.pixel.mcmc_sampler_config.reg_terms.beta="$BETA" \
  task.pixel.mcmc_sampler_config.tau="$TAU" \
  +task.pixel.mcmc_sampler_config.kernel_regularizer=pnp \
  +task.pixel.mcmc_sampler_config.kernel_noise_scale="$NOISE_SCALE" \
  +fixed_measurement_dir="$ROOT"/archived_inputs/gaussian/measurements \
  +fixed_kernel_dir="$ROOT"/archived_inputs/gaussian/kernel_gts \
  gpu=0 \
  > "tuning_logs/${NAME}.log" 2>&1

echo "$NAME"
