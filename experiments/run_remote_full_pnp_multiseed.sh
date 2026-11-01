#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

for seed in 123 2026 7; do
  name="validated_full_gaussian_pnp_seed${seed}"
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
    batch_size=32 \
    data.start_id=0 \
    data.end_id=100 \
    sampler.blind=True \
    sampler.nonlinear=False \
    sampler.annealing_scheduler_config.num_steps=200 \
    sampler.diffusion_scheduler_config.num_steps=5 \
    task.pixel.mcmc_sampler_config.num_steps=100 \
    task.pixel.mcmc_sampler_config.lr.image=5e-5 \
    task.pixel.mcmc_sampler_config.lr.kernel=4e-9 \
    task.pixel.mcmc_sampler_config.reg_terms.lamb=20 \
    task.pixel.mcmc_sampler_config.reg_terms.beta=2000000 \
    task.pixel.mcmc_sampler_config.tau=0.02 \
    +task.pixel.mcmc_sampler_config.kernel_regularizer=pnp \
    gpu=0 \
    > "tuning_logs/${name}.log" 2>&1
  echo "FINISH ${name}"
done

echo "ALL_FULL_PNP_RUNS_DONE"
