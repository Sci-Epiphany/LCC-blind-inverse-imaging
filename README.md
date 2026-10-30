# Latent Control Consistency for Blind Inverse Imaging

Reference implementation for the paper **Latent Control Consistency for Blind
Inverse Imaging**.

This repository contains the code used by the reported Gaussian, motion, and
KernelWizard nonlinear deblurring experiments. It is a curated research
release: datasets, checkpoints, generated samples, notebooks, and local IDE
files are intentionally excluded.

## Method and code scope

The reported benchmark follows a decoupled procedure:

1. A probability-flow ODE predicts clean image and operator states at the
   current noise level.
2. Joint Langevin/MCMC steps apply measurement feedback, a prior anchor, and
   kernel regularization.
3. Fresh noise maps the corrected states to the next annealing level.

The linear branch is implemented by `BlindDAPS` in `sampler.py`; the nonlinear
branch is implemented by `Nonlinear_DAPS`. Their image updates share the same
annealing and measurement-feedback structure, but use task-specific operator
solutions and regularization. See [docs/METHOD_CODE_MAP.md](docs/METHOD_CODE_MAP.md).

The optional slow-operator controller discussed as an extension in the paper
is **not** the implementation used for the main tables. Files from the
separate spatial-inconsistency/aberration study are not included here.

## Installation

Python 3.10 and a recent PyTorch build with CUDA are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

The BKSE KernelWizard modules retain their upstream license in
`forward_operator/bkse/LICENSE`.

## Checkpoints and data

Large model weights are not distributed in this repository. Place the
following files under `checkpoints/`:

- `ffhq256.pt`: image diffusion checkpoint.
- `kernel_checkpoint.pt`: linear kernel diffusion checkpoint.
- `kernel_denoiser_64.pth`: kernel PnP denoiser used by the linear branch.

The nonlinear branch additionally expects the KernelWizard checkpoint at
`forward_operator/bkse/experiments/pretrained/GOPRO_wVAE.pth`.

FFHQ test images are expected under `dataset/test-ffhq/`. The ImageNet-C and
GoPro-style nonlinear inputs use the paths recorded in `configs/paper/`.
Checkpoint and dataset licenses are separate from this source release.

## Running the paper configurations

The three complete recovered configurations are in `configs/paper/`:

```text
configs/paper/gaussian.yaml
configs/paper/motion.yaml
configs/paper/nonlinear.yaml
```

Use the archived Hydra configuration corresponding to the desired benchmark.
The core benchmark settings are:

- probability-flow ODE steps: `5`;
- annealing levels: `200`;
- Langevin/MCMC steps per level: `100`;
- noise range: `sigma_max=100`, `sigma_min=0.1`;
- four repeated outputs for LCC aggregation.

The original entry point is:

```bash
python posterior_sample.py \
  +data=test-ffhq \
  +kernel_data=gaussian64 \
  +model=ffhq256ddpm \
  +kernel_model=kernel64ddpm \
  +task=blind_gaussian_blur \
  +sampler=edm_daps \
  task_group=pixel \
  sampler.blind=True \
  sampler.nonlinear=False \
  sampler.diffusion_scheduler_config.num_steps=5 \
  sampler.annealing_scheduler_config.num_steps=200 \
  batch_size=1 \
  data.start_id=0 \
  data.end_id=1 \
  save_dir=results \
  name=gaussian_demo \
  gpu=0
```

The nonlinear run uses `+kernel_data=nonlinear`,
`+task=blind_nonlinear_blur`, `sampler.blind=False`, and
`sampler.nonlinear=True`.

## Validation

Run the source-only smoke test without loading model checkpoints:

```bash
python -m pytest -q tests
python scripts/check_release.py
```

## Reporting and aggregation

All reported full-reference means use every retained test image. Failed cases
are not filtered out. LCC metrics average four stochastic runs per image before
comparison. Per-image paired confidence intervals and failure-aware summaries
are provided in the paper supplement.

## Citation

Please cite the paper when using this code. Citation metadata is provided in
`CITATION.cff`.
