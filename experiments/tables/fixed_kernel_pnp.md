# Fixed-kernel full PnP reruns

These runs use seed 42 and four independent stochastic sampler runs, so the generated benchmark kernels match the comparison outputs.

| Task | Images | Runs | PSNR | SSIM | LPIPS | K-MSE | SR@10 dB | Run means |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gaussian | 100 | 4 | 25.1781 | 0.7327 | 0.2232 | 2.205e-06 | 400/400 | 25.29 / 25.09 / 25.21 / 25.13 |
| Motion | 200 | 4 | 20.0879 | 0.5696 | 0.3519 | 4.203e-06 | 786/800 | 20.59 / 20.50 / 20.28 / 18.98 |

| Task | Metric | Delta | 95% CI | Wins/Losses |
|---|---|---:|---:|---:|
| Gaussian | PSNR (dB) | +0.3610 | [+0.0791, +0.7017] | 49/51 |
| Gaussian | LPIPS | -0.0104 | [-0.0190, -0.0035] | 60/40 |
| Gaussian | K-MSE | +5.212e-08 | [+5.196e-08, +5.228e-08] | 0/100 |
| Motion | PSNR (dB) | -0.3165 | [-0.6130, -0.0352] | 84/116 |
| Motion | LPIPS | +0.0088 | [+0.0004, +0.0181] | 82/118 |
| Motion | K-MSE | +1.935e-10 | [+7.721e-11, +3.119e-10] | 72/128 |
