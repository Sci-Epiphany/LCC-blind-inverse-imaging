# Pre-BKSE latent proposal pilot

Four nonlinear test images, seed 42, one run per image/config.
Selection minimizes observation error only; ground truth is not used.

| Variant | N | PSNR | SSIM | LPIPS | K-MSE |
|---|---:|---:|---:|---:|---:|
| one stochastic proposal | 4 | 20.2294 | 0.5250 | 0.4841 | 5.493e+00 |
| four proposals, deterministic mean | 4 | 17.2974 | 0.4597 | 0.5455 | 1.545e+00 |
| four proposals, 0.5x noise | 4 | 19.7270 | 0.5142 | 0.4909 | 2.635e+00 |
| four proposals, 1.0x noise | 4 | 19.4015 | 0.4988 | 0.5086 | 5.728e+00 |
