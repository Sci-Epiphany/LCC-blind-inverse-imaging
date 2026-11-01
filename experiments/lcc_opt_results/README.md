# LCC optimization audit

This directory stores compact machine-readable outputs for the implementation
audit and the optimized consistency pipeline. The large image tensors remain on
the experiment host and are not part of the source release.

## Main findings

- The archived kernel gradient differentiated the normalized kernel but
  updated the unconstrained kernel latent, omitting the normalization
  Jacobian. The corrected evaluator uses a finite-difference test.
- The Gaussian raw-coordinate branch uses 50 annealing levels and 20 MCMC
  steps per level. The likelihood scale is tied to the known noise variance,
  and kernel thermal noise is applied with a factor of 0.1.
- A fixed-image HQS kernel solve and a fixed-kernel regularized image solve
  remove terminal image/operator ambiguity.
- Equal fusion of the raw- and normalized-coordinate branches is label-free.

## Headline result

On 100 fixed FFHQ Gaussian inputs, the equal two-branch fusion obtains
PSNR 27.5842 dB, SSIM 0.7934, LPIPS 0.1914, and per-kernel-normalized
K-MSE 1.528e-7.

The four-seed, four-image optimized single-branch audit gives
PSNR 28.0362 +/- 0.3364 dB and K-MSE 3.770e-7.

The retained 200-image, four-run motion output reaches PSNR 21.1626 dB,
SSIM 0.5851, LPIPS 0.3457, and K-MSE 3.916e-6 after consistency
refinement.

## Files

- `gaussian_s123_initial_metrics.json`: initial accelerated Gaussian branch.
- `gaussian_s123_initial_kernel.json`: corrected per-kernel MSE.
- `gaussian_s123_post_kernel.json`: fixed-image kernel refinement.
- `gaussian_s123_post_image.json`: fixed-kernel image reconciliation.
- `gaussian_two_branch_image_fusion.json`: image ensemble evaluation.
- `gaussian_two_branch_kernel_fusion.json`: kernel ensemble evaluation.
- `seed*_post_kernel.json`, `seed*_post_image.json`: four-seed audit.
- `motion_post_kernel.json`, `motion_post_image.json`: motion audit.

`../results/lcc_optimization.json` is the aggregated analysis consumed by the
paper-table audit.
