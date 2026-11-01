# Multi-seed loop-tuning pilot

Four FFHQ images per seed; one run per image/config; seeds 7, 42, 123, 314, 777, 2026, 2027, 4096.

| Variant | Seeds | Images/seed | PSNR | Seed SD | SSIM | LPIPS | K-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 200/100, ODE 5 | 8 | 4 | 25.0025 | 0.2938 | 0.7214 | 0.2172 | 1.492e-06 |
| outer 400, MCMC 50, 2x step | 8 | 4 | 24.6899 | 0.2146 | 0.7191 | 0.2193 | 1.496e-06 |
| ODE 10, MCMC 50 | 3 | 4 | 24.8306 | 0.4193 | 0.7096 | 0.2105 | 1.520e-06 |

| Candidate vs baseline | Seeds | Pairs | Delta PSNR [95% CI] | Std | Wins/Losses | K-MSE ratio |
|---|---:|---:|---:|---:|---:|---:|
| outer 400, MCMC 50, 2x step | 8 | 32 | -0.3126 [-0.6642,+0.0696] | 1.0711 | 10/22 | 1.003 |
| ODE 10, MCMC 50 | 3 | 12 | -0.0095 [-0.6043,+0.5668] | 1.0294 | 7/5 | 1.022 |
