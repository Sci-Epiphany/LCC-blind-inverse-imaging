# Paired image-level comparisons after the PnP fix

Gaussian and motion LCC values are averaged across completed full-data seeds before pairing. Nonlinear values use the retained four runs.

| Benchmark | Baseline | N | Delta PSNR (95% CI) | W/L | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| Gaussian | BKSE | 100 | +6.4937 [+5.9182, +7.0472] | 98/2 | 1.11e-17 |
| Gaussian | BlindDPS | 100 | +0.7545 [+0.2762, +1.2103] | 71/29 | 1.22e-05 |
| Gaussian | DMPlug | 100 | +1.2996 [+0.6903, +1.9370] | 70/30 | 4.36e-06 |
| Gaussian | DeblurGAN | 100 | -0.1415 [-0.5030, +0.1824] | 52/48 | 0.703 |
| Gaussian | DeepRFT | 100 | -0.2226 [-0.6196, +0.1277] | 49/51 | 0.728 |
| Gaussian | FFTFormer | 100 | +1.8939 [+1.5250, +2.2220] | 93/7 | 3e-15 |
| Gaussian | FastDiffEM | 100 | +12.8489 [+11.4671, +14.1601] | 88/12 | 2.14e-16 |
| Gaussian | SelfDeblur | 90 | +8.7156 [+8.1705, +9.2340] | 90/0 | 1.74e-16 |
| Nonlinear | BKSE | 100 | +0.3045 [-0.0829, +0.6611] | 64/36 | 0.00653 |
| Nonlinear | BlindDPS | 100 | +11.3743 [+9.9615, +12.7320] | 86/14 | 3.1e-16 |
| Nonlinear | DMPlug | 100 | +0.0567 [-0.7088, +0.8831] | 42/58 | 0.509 |
| Nonlinear | DeblurGAN | 100 | -3.5852 [-4.0189, -3.1608] | 2/98 | 5.6e-18 |
| Nonlinear | DeepRFT | 100 | -4.0115 [-4.5216, -3.5271] | 4/96 | 1.74e-17 |
| Nonlinear | FFTFormer | 100 | +0.1000 [-0.3776, +0.5518] | 55/45 | 0.258 |
| Nonlinear | FastDiffEM | 100 | +8.6898 [+6.8616, +10.5722] | 78/22 | 1.84e-11 |
| Nonlinear | SelfDeblur | 100 | +6.5512 [+5.9034, +7.1963] | 99/1 | 4.81e-18 |
