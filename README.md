# Latent Control Consistency for Blind Inverse Imaging

Reference implementation for the paper **Latent Control Consistency for Blind
Inverse Imaging**.

Code repository: <https://github.com/Sci-Epiphany/LCC-blind-inverse-imaging>

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
4. A fixed-image HQS solve refines the kernel, followed by a fixed-kernel
   regularized image solve.
5. Equal-weight fusion combines two label-free coordinate solutions.

The linear branch is implemented by `BlindDAPS` in `sampler.py`; the nonlinear
branch is implemented by `Nonlinear_DAPS`. Their image updates share the same
annealing and measurement-feedback structure, but use task-specific operator
solutions and regularization. See [docs/METHOD_CODE_MAP.md](docs/METHOD_CODE_MAP.md).

The optional slow-operator controller discussed as an extension in the paper
is **not** the implementation used for the main tables. Files from the
separate spatial-inconsistency/aberration study are not included here.

For the linear branch, `kernel_regularizer` selects `pnp`, `l1`, or `l2`.
`kernel_data_gradient: raw` differentiates through nonnegativity and simplex
normalization; `normalized` retains the scale-separated coordinate used by the
second fusion branch. The likelihood scale can be set to `null`, in which case
the known forward-operator noise standard deviation is used.

The optimized Gaussian entry in the paper uses two independently initialized
coordinate branches and one output from each. It obtains 27.5842 dB PSNR,
0.7934 SSIM, 0.1914 LPIPS, and 1.528e-7 per-kernel-normalized MSE on the
100-image fixed-input benchmark. Motion reaches 21.1626 dB after the
consistency refinement; the remaining supervised-domain gap is reported.

The nonlinear release maps KernelWizard inputs back to its trained `[0,1]`
domain, treats the proposal as an attracting prior, and derives the kernel
step from `eta_x * sqrt(N_x/N_k)`. On all 100 nonlinear images, the corrected
single sample reaches 24.6667 dB and two-sample fusion reaches 24.9520 dB;
the paired 20-image full-schedule audit reaches 26.0851 versus 24.0788 dB.
A four-seed thermal-noise ablation improves the small development pilot from
22.7111 to 22.7317 dB at a `0.1` image-noise multiplier. This gain is too
small for the headline benchmark. Observation-residual proposal selection and
post-MCMC kernel reprojection regress PSNR, so both are kept as documented
negative results.

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
- normalized-coordinate schedule: `200` levels, `100` MCMC steps;
- accelerated raw-coordinate schedule: `50` levels, `20` MCMC steps;
- noise range: `sigma_max=100`, `sigma_min=0.1`;
- Gaussian image fusion uses one output from each coordinate branch.

The optimized fixed-input launcher is:

```bash
bash experiments/run_remote_kernel_ablation.sh \
  4e-7 20 2000000 0.02 0 100 50 20 optimized 0.1 123
```

It writes the raw-coordinate branch. The normalized-coordinate branch uses
`kernel_data_gradient=normalized`, `lr.kernel=4e-9`, and the `200/100`
schedule. `experiments/refine_saved_kernels.py`,
`experiments/evaluate_image_kernel_reconciliation.py`, and
`experiments/evaluate_image_fusion.py` implement the remaining stages.

The general entry point is:

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

## Additional experiments

`experiments/` contains the exact-Jacobian audit, terminal consistency
refinement, two-branch fusion, PnP-versus-L1 audit, cross-kernel validation,
KernelWizard proposal diagnostic, and the earlier automatic parameter search.
See `experiments/README.md` and `experiments/lcc_opt_results/README.md`.

## Reporting and aggregation

All reported full-reference means use every retained test image. Failed cases
are not filtered out. The optimized Gaussian result fuses one independent
output from each coordinate branch; motion retains four runs. Per-image paired
confidence intervals and failure-aware summaries are provided in the paper
supplement. Kernel MSE always normalizes each predicted and target kernel
independently.

## Citation

Please cite the paper when using this code. Citation metadata is provided in
`CITATION.cff`.
