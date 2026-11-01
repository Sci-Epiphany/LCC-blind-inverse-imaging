#!/usr/bin/env bash
set -u

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

for seed in 7 42 123 314 777 2026 2027 4096; do
  name="seed${seed}_paper200m100_pnp"
  echo "START seed=$seed PnP"
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
  echo "END seed=$seed status=$?"
done

echo "ALL_PNP_RUNS_DONE"
