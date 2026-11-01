# Corrected Jacobian and consistency refinement

## Four-seed Gaussian pilot

| Seed | PSNR | SSIM | LPIPS | K-MSE |
|---:|---:|---:|---:|---:|
| 42 | 28.0846 | 0.7802 | 0.1920 | 3.673e-07 |
| 7 | 27.5457 | 0.7707 | 0.1987 | 4.074e-07 |
| 2026 | 28.0205 | 0.7770 | 0.1934 | 3.950e-07 |
| 123 | 28.4940 | 0.7856 | 0.1906 | 3.380e-07 |

Mean PSNR **28.0362 ± 0.3364 dB**; K-MSE **3.770e-07**.

| Full Gaussian 100 images | PSNR | SSIM | LPIPS | K-MSE |
|---|---:|---:|---:|---:|
| Seed 123, two-branch equal fusion | 27.5842 | 0.7934 | 0.1914 | 1.528e-07 |

| Full motion 200 images × 4 runs | PSNR | SSIM | LPIPS | K-MSE |
|---|---:|---:|---:|---:|
| Optimized consistency refinement | 21.1626 | 0.5851 | 0.3457 | 3.916e-06 |
