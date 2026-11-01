# Paired image-level comparisons

LCC values are averaged across four runs per image before pairing.

| Benchmark | Baseline | N | Delta PSNR (95% CI) | Wins | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| Gaussian | BKSE | 100 | +6.1219 [+5.4701, +6.7360] | 95/100 | 1.2e-16 |
| Gaussian | BlindDPS | 100 | +0.3827 [-0.0933, +0.8542] | 70/100 | 0.003 |
| Gaussian | DMPlug | 100 | +0.9278 [+0.3921, +1.5150] | 60/100 | 0.00589 |
| Gaussian | DeblurGAN | 100 | -0.5133 [-0.8718, -0.1858] | 45/100 | 0.0546 |
| Gaussian | DeepRFT | 100 | -0.5944 [-1.0264, -0.2029] | 48/100 | 0.0862 |
| Gaussian | FFTFormer | 100 | +1.5221 [+1.1524, +1.8750] | 85/100 | 2.55e-11 |
| Gaussian | FastDiffEM | 100 | +12.4771 [+11.1479, +13.7497] | 86/100 | 2.4e-16 |
| Gaussian | SelfDeblur | 90 | +8.2062 [+7.5487, +8.8412] | 89/90 | 2.06e-16 |
| Motion | BKSE | 200 | +0.3644 [-0.0743, +0.7933] | 97/200 | 0.291 |
| Motion | BlindDPS | 200 | +0.6022 [+0.0654, +1.1159] | 115/200 | 0.00297 |
| Motion | DMPlug | 200 | +1.9745 [+1.4221, +2.5204] | 133/200 | 7.23e-11 |
| Motion | DeblurGAN | 200 | -0.7863 [-1.2893, -0.2874] | 71/200 | 0.00345 |
| Motion | DeepRFT | 200 | -2.5255 [-3.0867, -1.9802] | 43/200 | 7.68e-16 |
| Motion | FFTFormer | 200 | +1.0785 [+0.6086, +1.5470] | 118/200 | 7.16e-06 |
| Motion | FastDiffEM | 200 | +7.1138 [+5.9503, +8.2636] | 151/200 | 9.45e-20 |
| Motion | SelfDeblur | 200 | +3.3356 [+2.9099, +3.7555] | 176/200 | 2.94e-27 |
| Nonlinear | BKSE | 100 | +0.3045 [-0.0829, +0.6611] | 64/100 | 0.00653 |
| Nonlinear | BlindDPS | 100 | +11.3743 [+9.9615, +12.7320] | 86/100 | 3.1e-16 |
| Nonlinear | DMPlug | 100 | +0.0567 [-0.7088, +0.8831] | 42/100 | 0.509 |
| Nonlinear | DeblurGAN | 100 | -3.5852 [-4.0189, -3.1608] | 2/100 | 5.6e-18 |
| Nonlinear | DeepRFT | 100 | -4.0115 [-4.5216, -3.5271] | 4/100 | 1.74e-17 |
| Nonlinear | FFTFormer | 100 | +0.1000 [-0.3776, +0.5518] | 55/100 | 0.258 |
| Nonlinear | FastDiffEM | 100 | +8.6898 [+6.8616, +10.5722] | 78/100 | 1.84e-11 |
| Nonlinear | SelfDeblur | 100 | +6.5512 [+5.9034, +7.1963] | 99/100 | 4.81e-18 |
