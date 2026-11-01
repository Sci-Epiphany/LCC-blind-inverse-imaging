#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

GPU=${1:-0}
IMAGES=${2:-4}
OUTER=${3:-50}
MCMC=${4:-20}
SEED=${5:-42}
TAG=${6:-nonlinear}
KERNEL_TRAJECTORY=${7:-proposal}
RETURN_CLEAN=${8:-false}
KERNEL_PRIOR_WEIGHT=${9:-1e5}
KERNEL_NOISE_SCALE=${10:-1}
KERNEL_LR=${11:-2e-9}
DATA_SET=${12:-test-realblur}
BATCH_SIZE=${13:-4}
PROPOSAL_MODE=${14:-sample}
PROPOSAL_CANDIDATES=${15:-1}
PROPOSAL_NOISE_SCALE=${16:-1}
AUTO_LR_KERNEL=${17:-false}
IMAGE_LR=${18:-1e-5}
TAU=${19:-null}
RUNS=${20:-1}
IMAGE_NOISE_SCALE=${21:-1}
NAME="nonlinear_${TAG}_${KERNEL_TRAJECTORY}_clean${RETURN_CLEAN}_x${IMAGE_LR}_klr${KERNEL_LR}_kw${KERNEL_PRIOR_WEIGHT}_kn${KERNEL_NOISE_SCALE}_xn${IMAGE_NOISE_SCALE}_${PROPOSAL_MODE}${PROPOSAL_CANDIDATES}x${PROPOSAL_NOISE_SCALE}_n${IMAGES}_o${OUTER}_m${MCMC}_r${RUNS}_s${SEED}_${DATA_SET}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

"$PY" posterior_sample.py \
  +data="$DATA_SET" \
  +kernel_data=nonlinear \
  +model=ffhq256ddpm \
  +task=blind_nonlinear_blur \
  +sampler=edm_daps \
  task_group=pixel \
  save_dir=tuning_results \
  name="$NAME" \
  seed="$SEED" \
  num_runs="$RUNS" \
  save_samples=true \
  batch_size="$BATCH_SIZE" \
  data.start_id=0 \
  data.end_id="$IMAGES" \
  sampler.blind=False \
  sampler.nonlinear=True \
  sampler.annealing_scheduler_config.num_steps="$OUTER" \
  sampler.diffusion_scheduler_config.num_steps=5 \
  task.pixel.mcmc_sampler_config.num_steps="$MCMC" \
  task.pixel.mcmc_sampler_config.lr.image="$IMAGE_LR" \
  task.pixel.mcmc_sampler_config.lr.kernel="$KERNEL_LR" \
  task.pixel.mcmc_sampler_config.reg_terms.lamb=300 \
  task.pixel.mcmc_sampler_config.reg_terms.beta=20000000 \
  task.pixel.mcmc_sampler_config.tau="$TAU" \
  +task.pixel.mcmc_sampler_config.kernel_trajectory="$KERNEL_TRAJECTORY" \
  +task.pixel.mcmc_sampler_config.return_clean="$RETURN_CLEAN" \
  +task.pixel.mcmc_sampler_config.kernel_prior_weight="$KERNEL_PRIOR_WEIGHT" \
  +task.pixel.mcmc_sampler_config.kernel_noise_scale="$KERNEL_NOISE_SCALE" \
  +task.pixel.mcmc_sampler_config.image_noise_scale="$IMAGE_NOISE_SCALE" \
  +task.pixel.operator.proposal_mode="$PROPOSAL_MODE" \
  +task.pixel.operator.proposal_candidates="$PROPOSAL_CANDIDATES" \
  +task.pixel.operator.proposal_noise_scale="$PROPOSAL_NOISE_SCALE" \
  +task.pixel.mcmc_sampler_config.auto_lr_kernel="$AUTO_LR_KERNEL" \
  gpu=0 \
  > "tuning_logs/${NAME}.log" 2>&1

echo "$NAME"
