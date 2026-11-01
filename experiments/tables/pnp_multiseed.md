# Paired multi-seed PnP regularizer audit

Seeds: 7, 42, 123, 314, 777, 2026, 2027, 4096; four images per seed.

| Regularizer | Seeds | Cases | PSNR | Seed SD | SSIM | LPIPS | K-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 8 | 32 | 25.0025 | 0.2938 | 0.7214 | 0.2172 | 1.492e-06 |
| PnP | 8 | 32 | 26.6310 | 0.2457 | 0.7510 | 0.2050 | 1.358e-06 |

Paired PnP-L1 PSNR delta: **+1.6285 dB** (95% CI [+1.3673, +1.8989]), wins/losses 31/1.
Mean PnP/L1 kernel-MSE ratio: 0.909.
