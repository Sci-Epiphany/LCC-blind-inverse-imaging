#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python
GPU=$1
NAME=$2
ANNEAL=$3
ODE=$4
MCMC=$5
LRX=$6
LRK=$7
LAMB=$8
BETA=$9
TAU=${10}
SEED=${11:-42}

mkdir -p "$ROOT/tuning_logs" "$ROOT/tuning_results"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

nohup "$PY" posterior_sample.py \
  +data=test-ffhq \
  +kernel_data=gaussian64 \
  +model=ffhq256ddpm \
  +kernel_model=kernel64ddpm \
  +task=blind_gaussian_blur \
  +sampler=edm_daps \
  task_group=pixel \
  save_dir=tuning_results \
  name="$NAME" \
  num_runs=1 \
  seed="$SEED" \
  save_samples=false \
  batch_size=4 \
  data.start_id=0 \
  data.end_id=4 \
  sampler.blind=True \
  sampler.nonlinear=False \
  sampler.annealing_scheduler_config.num_steps="$ANNEAL" \
  sampler.diffusion_scheduler_config.num_steps="$ODE" \
  task.pixel.mcmc_sampler_config.num_steps="$MCMC" \
  task.pixel.mcmc_sampler_config.lr.image="$LRX" \
  task.pixel.mcmc_sampler_config.lr.kernel="$LRK" \
  task.pixel.mcmc_sampler_config.reg_terms.lamb="$LAMB" \
  task.pixel.mcmc_sampler_config.reg_terms.beta="$BETA" \
  task.pixel.mcmc_sampler_config.tau="$TAU" \
  gpu=0 \
  > "tuning_logs/${NAME}.log" 2>&1 < /dev/null &

echo $!
