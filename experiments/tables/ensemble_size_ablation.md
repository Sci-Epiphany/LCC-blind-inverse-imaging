# Sample-count ablation

Fusion is the equal-weight pixel mean of all evaluated subsets without ground truth.
Two-sample results average all six pairs of the four complete runs.

| Task | Samples | N | PSNR | SSIM | Delta PSNR | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 1 | 100 | 24.8117 | 0.7204 | +0.0000 | 0/100 |
| Gaussian | 2 | 100 | 25.2511 | 0.7499 | +0.4393 | 96/100 |
| Gaussian | 4 | 100 | 25.6333 | 0.7676 | +0.8216 | 99/100 |
| Motion | 1 | 197 | 20.3957 | 0.5792 | +0.0000 | 0/197 |
| Motion | 2 | 197 | 21.1864 | 0.6222 | +0.7907 | 197/197 |
| Motion | 4 | 197 | 21.8268 | 0.6520 | +1.4311 | 197/197 |
