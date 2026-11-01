# Theory-code consistency audit

Audit date: 2026-11-01.

## Reported implementation

The main benchmark is produced by:

```text
/home/gsn/PhD_thesis/DAPS/sampler.py
  BlindDAPS       -> linear Gaussian/motion
  Nonlinear_DAPS  -> BKSE KernelWizard nonlinear
```

The recovered configurations specify:

- `N_ode=5`;
- `N_A=200`;
- `N=100` Langevin/MCMC steps per annealing level;
- separate image and kernel learning rates;
- a Gaussian prior anchor and HQS kernel regularization in the linear branch;
- a BKSE KernelWizard kernel solution in the nonlinear branch.

## Correspondence

| Paper statement | Code evidence | Status |
|---|---|---|
| Probability-flow prior prediction at each noise level | `sampler.py`: `DiffusionPFODE` in `BlindDAPS.sample` and `Nonlinear_DAPS.sample` | aligned |
| Joint image/operator posterior correction | `cores/mcmc.py`: `MCMCSampler.score_fn`; `cores/mcmc_nl.py`: `MCMCNLSampler.score_fn` | aligned |
| Coordinate-wise image and kernel steps | `get_lr` returns a separate rate for each component | aligned |
| Linear kernel HQS and simplex projection | `MCMCSampler.score_fn`, `regularization_step`, `normalize_kernel` | aligned |
| BKSE KernelWizard nonlinear operator | `forward_operator.NonlinearBlur` and `nonlinear_config.yaml` | aligned |
| `N_A=200`, `N_ode=5`, `N=100` | three recovered configuration files in `experiments/source_data/` | aligned |
| Signed residual-Jacobian theory | `BlindBlurOperator.gradient` and `MCMCSampler.score_fn` | kernel normalization Jacobian implemented explicitly |
| Noise-calibrated likelihood | `MCMCSampler.score_fn` with `tau=null` | aligned with the forward-operator noise level |
| Terminal consistency | `refine_saved_kernels.py`, `evaluate_image_kernel_reconciliation.py` | fixed-image kernel solve then fixed-kernel image solve |
| Two-branch fusion | `evaluate_image_fusion.py`, `evaluate_kernel_fusion.py` | fixed equal weights, no ground truth in selection |

## Corrections made

- The main text no longer says that the reported benchmark uses a
  two-timescale/frozen-operator controller.
- The framework figure now says `Decoupled annealing`, and the correction loop
  is labeled `Langevin Dynamics`.
- The slow-operator and residual-gated controllers are explicitly marked as
  extensions not used for Tables I and II.
- The nonlinear text no longer claims that linear and nonlinear branches share
  one literal inference controller. They share the formulation and outer
  anneal, but have task-specific operator solutions.
- The linear kernel regularizer is now an explicit configuration variable.
  The paper default is PnP, matching the component ablation and selected
  method. The earlier retained source hard-coded L1 despite loading the PnP
  denoiser.
- The inexact-descent proof now uses the standard conservative bound
  `F+ <= F - eta/2 ||grad F||^2 + eta/2 eps^2` for
  `eta <= 1/(2L)`. The previous coefficient was stronger than the stated
  smoothness assumptions support.
- Files and experiments from `revision_2026/unified_v13_20261022`, including
  rotation, aberration, compound-blur, cross-domain, and spatial-kernel tests,
  are excluded from this article and from the clean code release.
- The archived kernel gradient differentiated the normalized tensor instead
  of the latent updated by the sampler. The release differentiates through
  both the nonnegativity and simplex-normalization maps and includes a
  finite-difference Jacobian test.
- The archived kernel-MSE evaluator normalized the whole batch. The corrected
  evaluator normalizes each predicted and target kernel independently.
- Thermal noise is separated from deterministic kernel-step scaling by a
  configurable multiplier; the likelihood temperature is tied to the known
  measurement noise.

## Remaining limitation

The general signed-residual control objective is broader than the explicit
score decomposition implemented by the benchmark. The linear kernel-coordinate
Jacobian is now exact; the image/operator cross-blocks and the nonlinear
KernelWizard Jacobian are not materialized as a dense matrix.

`tests/test_residual_gradient.py` verifies
`grad(0.5 ||e||^2) = J^T e` for a linear observation residual and a Gaussian
prior residual. This is an implementation/mathematics consistency test; the
stability propositions are not claimed to be the source of benchmark gains.

The linear branch already generates a simple kernel from Gaussian noise using
its own kernel diffusion score model. BKSE KernelWizard is an alternative
nonlinear operator decoder, not the only component that generates a kernel.

On eight paired seeds and four images, correcting the regularizer to PnP
improves PSNR by +1.6285 dB (95% interval [+1.3673,+1.8989], 31 wins/1 loss)
over L1. The archived K-MSE values from that audit are not used because they
predate the per-kernel normalization fix.

The optimized 100-image Gaussian result reaches 27.5842 dB PSNR, 0.7934 SSIM,
0.1914 LPIPS, and 1.528e-7 per-kernel MSE. The four-seed optimized pilot gives
28.0362 +/- 0.3364 dB. Motion reaches 21.1626 dB, 0.5851 SSIM, and 0.3457
LPIPS after terminal refinement, but remains below the supervised PSNR leader.
The paper reports this domain-dependent outcome rather than claiming a
universal improvement.
