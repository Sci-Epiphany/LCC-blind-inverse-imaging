#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

for candidate in \
  "lrk2e9_tau001 2e-9 0.01 200 100 5e-5 2e-9" \
  "lrk2e9_tau002 2e-9 0.02 200 100 5e-5 2e-9" \
  "lrk4e9_tau001 4e-9 0.01 200 100 5e-5 4e-9" \
  "outer400_mcmc50 4e-9 0.02 400 50 1e-4 8e-9"; do
  read -r label lr_k tau outer_steps mcmc_steps lr_x effective_lr_k <<< "$candidate"
  for seed in 42 123 2026 7; do
    name="seed${seed}_pnp_${label}"
    echo "START ${name}"
    "$PY" posterior_sample.py \
      +data=test-ffhq \
      +kernel_data=gaussian64 \
      +model=ffhq256ddpm \
      +kernel_model=kernel64ddpm \
      +task=blind_gaussian_blur \
      +sampler=edm_daps \
      task_group=pixel \
      save_dir=tuning_results \
      name="$name" \
      seed="$seed" \
      num_runs=1 \
      save_samples=false \
      batch_size=4 \
      data.start_id=0 \
      data.end_id=4 \
      sampler.blind=True \
      sampler.nonlinear=False \
      sampler.annealing_scheduler_config.num_steps="$outer_steps" \
      sampler.diffusion_scheduler_config.num_steps=5 \
      task.pixel.mcmc_sampler_config.num_steps="$mcmc_steps" \
      task.pixel.mcmc_sampler_config.lr.image="$lr_x" \
      task.pixel.mcmc_sampler_config.lr.kernel="$effective_lr_k" \
      task.pixel.mcmc_sampler_config.reg_terms.lamb=20 \
      task.pixel.mcmc_sampler_config.reg_terms.beta=2000000 \
      task.pixel.mcmc_sampler_config.tau="$tau" \
      +task.pixel.mcmc_sampler_config.kernel_regularizer=pnp \
      gpu=0 \
      > "tuning_logs/${name}.log" 2>&1
    echo "FINISH ${name}"
  done
done

echo "ALL_PNP_PARAMETER_RUNS_DONE"
