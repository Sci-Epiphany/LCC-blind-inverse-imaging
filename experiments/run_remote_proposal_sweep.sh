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
  local name=$1
  local candidates=$2
  local noise_scale=$3

  echo "START proposal name=$name candidates=$candidates scale=$noise_scale"
  "$PY" posterior_sample.py \
    +data=test-realblur \
    +kernel_data=nonlinear \
    +model=ffhq256ddpm \
    +task=blind_nonlinear_blur \
    +sampler=edm_daps \
    task_group=pixel \
    save_dir=tuning_results \
    name="$name" \
    seed=42 \
    num_runs=1 \
    save_samples=false \
    batch_size=4 \
    data.start_id=0 \
    data.end_id=4 \
    sampler.blind=False \
    sampler.nonlinear=True \
    +task.pixel.operator.proposal_candidates="$candidates" \
    +task.pixel.operator.proposal_noise_scale="$noise_scale" \
    sampler.annealing_scheduler_config.num_steps=200 \
    sampler.diffusion_scheduler_config.num_steps=5 \
    gpu=0 \
    > "tuning_logs/${name}.log" 2>&1
  echo "END proposal name=$name status=$?"
}

run_case proposal_c1_scale1 1 1.0
run_case proposal_c4_scale05 4 0.5
run_case proposal_c4_scale1 4 1.0

echo "ALL_PROPOSAL_RUNS_DONE"
