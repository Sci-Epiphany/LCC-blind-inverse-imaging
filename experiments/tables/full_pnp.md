# Cross-kernel Gaussian PnP validation

Completed full runs: 3 (42, 123, 2026); 100 images per run.

Each seed also regenerates the test kernels. These runs measure generalization to new kernel realizations and are not substituted into the fixed-kernel comparison table.

| Seeds | N runs | Cases | PSNR | SSIM | LPIPS | K-MSE | SR@10 dB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 300 | 25.2441 | 0.7356 | 0.2228 | 2.125e-06 | 300/300 |

Paired per-image difference against the four-run archived mean:

| Metric | Delta | 95% CI | Wins/Losses |
|---|---:|---:|---:|
| PSNR (dB) | +0.4270 | [+0.1475, +0.7599] | 59/41 |
| LPIPS | -0.0107 | [-0.0194, -0.0039] | 64/36 |
| K-MSE | -2.818e-08 | [-2.924e-08, -2.709e-08] | 100/0 |

| Run | PSNR | Delta vs archived mean | 95% CI | W/L |
|---|---:|---:|---:|---:|
| seed123 | 25.3012 | +0.4841 | [+0.1938, +0.8200] | 62/38 |
| seed2026 | 25.2422 | +0.4251 | [+0.1119, +0.7899] | 54/46 |
| seed42 | 25.1889 | +0.3718 | [-0.0104, +0.7619] | 64/36 |
