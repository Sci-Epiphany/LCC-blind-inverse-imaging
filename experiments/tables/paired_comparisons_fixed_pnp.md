# Paired image-level comparisons after the fixed-input PnP rerun

LCC Gaussian values average four stochastic runs on the archived measurement set. Nonlinear values retain the four archived runs.

| Benchmark | Baseline | N | Delta PSNR (95% CI) | W/L | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| Gaussian | BKSE | 100 | +6.4830 [+6.0256, +6.9428] | 100/0 | 3.9e-18 |
| Gaussian | BlindDPS | 100 | +0.7438 [+0.4324, +1.0981] | 74/26 | 2.16e-06 |
| Gaussian | DMPlug | 100 | +1.2888 [+0.7799, +1.8555] | 70/30 | 1.55e-05 |
| Gaussian | DeblurGAN | 100 | -0.1523 [-0.3775, +0.0739] | 51/49 | 0.403 |
| Gaussian | DeepRFT | 100 | -0.2333 [-0.5033, +0.0318] | 44/56 | 0.329 |
| Gaussian | FFTFormer | 100 | +1.8831 [+1.6540, +2.1162] | 94/6 | 7.56e-17 |
| Gaussian | FastDiffEM | 100 | +12.8382 [+11.4944, +14.0881] | 85/15 | 3.01e-16 |
| Gaussian | SelfDeblur | 90 | +8.6076 [+8.1142, +9.1114] | 90/0 | 1.74e-16 |
| Nonlinear | BKSE | 100 | +0.3045 [-0.0829, +0.6611] | 64/36 | 0.00653 |
| Nonlinear | BlindDPS | 100 | +11.3743 [+9.9615, +12.7320] | 86/14 | 3.1e-16 |
| Nonlinear | DMPlug | 100 | +0.0567 [-0.7088, +0.8831] | 42/58 | 0.509 |
| Nonlinear | DeblurGAN | 100 | -3.5852 [-4.0189, -3.1608] | 2/98 | 5.6e-18 |
| Nonlinear | DeepRFT | 100 | -4.0115 [-4.5216, -3.5271] | 4/96 | 1.74e-17 |
| Nonlinear | FFTFormer | 100 | +0.1000 [-0.3776, +0.5518] | 55/45 | 0.258 |
| Nonlinear | FastDiffEM | 100 | +8.6898 [+6.8616, +10.5722] | 78/22 | 1.84e-11 |
| Nonlinear | SelfDeblur | 100 | +6.5512 [+5.9034, +7.1963] | 99/1 | 4.81e-18 |
| Motion | BKSE | 200 | +0.0479 [-0.4302, +0.5117] | 92/108 | 0.867 |
| Motion | BlindDPS | 200 | +0.2857 [-0.2445, +0.8306] | 116/84 | 0.0542 |
| Motion | DMPlug | 200 | +1.6580 [+1.0735, +2.2603] | 126/74 | 2.07e-07 |
| Motion | DeblurGAN | 200 | -1.1028 [-1.6398, -0.5761] | 69/131 | 0.000102 |
| Motion | DeepRFT | 200 | -2.8420 [-3.4708, -2.2734] | 46/154 | 2.69e-17 |
| Motion | FFTFormer | 200 | +0.7619 [+0.2699, +1.2705] | 115/85 | 0.000629 |
| Motion | FastDiffEM | 200 | +6.7973 [+5.6180, +7.9442] | 151/49 | 8.48e-19 |
| Motion | SelfDeblur | 200 | +3.0191 [+2.5354, +3.4788] | 165/35 | 1.33e-22 |
