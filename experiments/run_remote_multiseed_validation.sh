#!/usr/bin/env bash
set -u

ROOT=/home/gsn/PhD_thesis/DAPS
PY=/home/gsn/anaconda3/envs/BlindDPS/bin/python
GPU=${1:-1}

mkdir -p "$ROOT/tuning_logs" "$ROOT/tuning_results"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

run_case() {
  local seed=$1
  local name=$2
  local anneal=$3
  local mcmc=$4
  local lrx=$5
  local lrk=$6

  echo "START seed=$seed name=$name anneal=$anneal mcmc=$mcmc lrx=$lrx lrk=$lrk"
  "$PY" posterior_sample.py \
    +data=test-ffhq \
    +kernel_data=gaussian64 \
    +model=ffhq256ddpm \
    +kernel_model=kernel64ddpm \
    +task=blind_gaussian_blur \
    +sampler=edm_daps \
    task_group=pixel \
    save_dir=tuning_results \
    name="seed${seed}_${name}" \
    seed="$seed" \
    num_runs=1 \
    save_samples=false \
    batch_size=4 \
    data.start_id=0 \
    data.end_id=4 \
    sampler.blind=True \
    sampler.nonlinear=False \
    sampler.annealing_scheduler_config.num_steps="$anneal" \
    sampler.diffusion_scheduler_config.num_steps=5 \
    task.pixel.mcmc_sampler_config.num_steps="$mcmc" \
    task.pixel.mcmc_sampler_config.lr.image="$lrx" \
    task.pixel.mcmc_sampler_config.lr.kernel="$lrk" \
    task.pixel.mcmc_sampler_config.reg_terms.lamb=20 \
    task.pixel.mcmc_sampler_config.reg_terms.beta=2000000 \
    task.pixel.mcmc_sampler_config.tau=0.02 \
    gpu=0 \
    > "tuning_logs/seed${seed}_${name}.log" 2>&1
  echo "END seed=$seed name=$name status=$?"
}

for seed in 7 314 777 2027 4096; do
  run_case "$seed" baseline_200m100_ode5 200 100 5e-5 4e-9
  run_case "$seed" outer400_mcmc50_x2 400 50 1e-4 8e-9
done

echo "ALL_VALIDATION_RUNS_DONE"
