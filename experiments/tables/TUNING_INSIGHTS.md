# Kernel-generation and scheduler audit

## What is already implemented

The linear kernel is already diffusion-generated from Gaussian noise:

- `BlindDAPS.get_start` samples the initial image and kernel states from the
  configured Gaussian prior.
- For the kernel state, `DiffusionPFODE` evaluates the `kernel64ddpm` score
  model from the current noise level.
- Joint MCMC/Langevin correction then refines the decoded simple kernel.
- Ground-truth kernels are used only for evaluation.

BKSE KernelWizard is a second operator representation. It predicts
`(mu_q, sigma_q)` from the current image and observation and samples a latent
`q`; it is not the only noise-to-kernel generator in the method.

## Historical configuration finding

The archived screenshot at `2025-04-26 16:03` reports a single-image
PSNR/LPIPS pair of `26.463/0.181`. The matching run began at `15:42:40` and
used an older single-image configuration:

```text
annealing=200, ODE=5, MCMC=100
lr_x=5e-5, lr_k=2e-9, tau=0.01
```

That run predates the final kernel regularization code and is not a full-sample
result. Across 846 retained later Hydra configurations, 839 use `200/5/100`
and seven use `200/5/1000`; no retained configuration supports a universal
`400/50` claim.

## Multi-seed scheduler comparison

The controlled Gaussian pilot uses four FFHQ images per seed. Eight seeds
compare the retained schedule with `400/50` and doubled step sizes:

| Variant | Seeds | Cases | PSNR | Delta | Wins/Losses | LPIPS | K-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline `200/100` | 8 | 32 | 25.0025 | -- | -- | 0.2172 | 1.492e-6 |
| `400/50`, 2x step | 8 | 32 | 24.6899 | -0.3126 | 10/22 | 0.2193 | 1.496e-6 |

The candidate reduces seed-to-seed PSNR variance from `0.2938` to `0.2146` dB,
but it does not improve mean fidelity. It is not a headline replacement.

## Kernel regularizer implementation audit

The retained source had `MCMCSampler.score_fn` hard-coded to the L1 branch even
though it loaded the kernel PnP denoiser. The paper's selected method and
component table use PnP. After making the regularizer explicit, an eight-seed
paired audit on the same four images gave:

| Regularizer | Cases | PSNR | SSIM | LPIPS | K-MSE |
|---|---:|---:|---:|---:|---:|
| L1 | 32 | 25.0025 | 0.7214 | 0.2172 | 1.492e-6 |
| PnP | 32 | **26.6310** | **0.7510** | **0.2050** | **1.358e-6** |

The paired PnP-L1 improvement is **+1.6285 dB** with a 95% bootstrap interval
of **[+1.3673,+1.8989] dB** and a **31/1** win/loss count. This is the clearest
reproducible improvement found in the retuning experiments and matches the
component ablation's original PnP signal.

A direct check of the previously remembered `lr_k=2e-9, tau=0.01` setting
was run with the corrected PnP implementation. On the same four images and
seed 42 it obtained 24.4794 dB PSNR, 0.6942 SSIM, 0.2449 LPIPS, and
1.397e-6 kernel MSE. The paper setting (`lr_k=4e-9, tau=0.02`) obtained
26.5444 dB, 0.7488 SSIM, 0.2018 LPIPS, and 1.348e-6 on the same cases.
The remembered setting is therefore a regression under the corrected
regularizer and is not selected.

## Fixed-input full rerun

A separate four-run Gaussian rerun loads the archived measurement images
rather than regenerating kernels. It therefore shares the comparison inputs.
The corrected PnP configuration obtains 25.1781 +/- 1.9595 dB PSNR,
0.7327 SSIM, 0.2232 LPIPS, and 2.2048e-6 kernel MSE over 400 outputs. All 400
outputs exceed 10 dB, versus 396/400 in the archived implementation. Compared
with the archived per-image mean, PSNR changes by +0.3610 dB
[+0.0791,+0.7017], LPIPS by -0.0104 [-0.0190,-0.0035], and kernel MSE by
+5.21e-8. Thus the full rerun improves fidelity and perceptual quality while
slightly increasing kernel MSE; it remains well below the comparison-method
kernel errors.

The corresponding fixed-input motion rerun obtains 20.0879 dB, 0.5696 SSIM,
0.3519 LPIPS, and 4.203e-6 kernel MSE. Four run means are 20.59, 20.50,
20.28, and 18.98 dB. The final low run makes the aggregate 0.32 dB below the
archived L1 implementation. Four-sample image fusion gives 20.082, 20.860, and
21.539 dB for K=1, 2, and 4, respectively; K=4 wins on 199/200 images and
costs four inference passes.

## Pre-BKSE latent proposal test

The nonlinear branch was modified experimentally to draw multiple BKSE latent
proposals and select the one with minimum observation error. Selection never
uses ground truth.

| Variant | PSNR | SSIM | LPIPS | K-MSE |
|---|---:|---:|---:|---:|
| One stochastic proposal | **20.2294** | **0.5250** | **0.4841** | 5.493 |
| Four, deterministic mean | 17.2974 | 0.4597 | 0.5455 | **1.545** |
| Four, 0.5x noise, hard select | 19.7270 | 0.5142 | 0.4909 | 2.635 |
| Four, 1.0x noise, hard select | 19.4015 | 0.4988 | 0.5086 | 5.728 |

The observation-only selector improves latent kernel matching but worsens
image restoration. The original single stochastic proposal remains best among
these variants. This pilot is intentionally excluded from the manuscript's
main tables.

## Automatic parameter search

Gaussian-process Expected-Improvement search selected
`lr_k=3.526e-9, tau=0.01532` on the four-image seed-42 development set,
raising PSNR from 26.5444 to 27.6135 dB. The gain did not survive held-out
seeds: the PSNR changes were -0.2580, +0.8150, and -1.6480 dB, for a mean of
-0.3637 dB and only one win in three seeds. The retained configuration is
therefore unchanged. The search implementation and full trace are preserved
to support future scale-normalized or noise-adaptive preconditioning.

## Recommendation

Do not claim a large accuracy gain from the remembered outer-loop setting. The
evidence supports a variance reduction only, at a mean-PSNR cost. A true
pre-BKSE diffusion prior would require a score model trained on KernelWizard
latents; candidate selection alone is not that model.

