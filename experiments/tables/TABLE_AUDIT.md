# Table audit

Verified 102 numerical rows against retained artifacts.

| Table | Group | Method | Printed | Source | Status |
|---|---|---|---|---|---|
| linear | Gaussian | SelfDeblur | 16.61, 0.2745, 0.5456, 8.1e-05 | `distribution_audit.csv` | PASS |
| linear | Gaussian | DeblurGAN | 25.33, 0.7528, 0.3365, -- | `distribution_audit.csv` | PASS |
| linear | Gaussian | BKSE | 18.7, 0.4994, 0.4083, -- | `distribution_audit.csv` | PASS |
| linear | Gaussian | FFTFormer | 23.3, 0.7615, 0.3166, -- | `distribution_audit.csv` | PASS |
| linear | Gaussian | DeepRFT | 25.41, 0.7566, 0.3305, -- | `distribution_audit.csv` | PASS |
| linear | Gaussian | BlindDPS | 24.43, 0.7631, 0.2366, 1.06e-05 | `distribution_audit.csv` | PASS |
| linear | Gaussian | FastDiffEM | 12.34, 0.3863, 0.5297, 3.23e-05 | `distribution_audit.csv` | PASS |
| linear | Gaussian | DMPlug | 23.89, 0.7347, 0.3124, 0.00016 | `distribution_audit.csv` | PASS |
| linear | Gaussian | LCC | 25.18, 0.7327, 0.2232, 2.205e-06 | `ldcc_full_metrics.csv` | PASS |
| linear | Motion | SelfDeblur | 17.07, 0.3351, 0.5006, 6.05e-05 | `distribution_audit.csv` | PASS |
| linear | Motion | DeblurGAN | 21.19, 0.6406, 0.3913, -- | `distribution_audit.csv` | PASS |
| linear | Motion | BKSE | 20.04, 0.5886, 0.5104, -- | `distribution_audit.csv` | PASS |
| linear | Motion | FFTFormer | 19.33, 0.604, 0.4042, -- | `distribution_audit.csv` | PASS |
| linear | Motion | DeepRFT | 22.93, 0.7028, 0.308, -- | `distribution_audit.csv` | PASS |
| linear | Motion | BlindDPS | 19.8, 0.604, 0.3369, 1.97e-05 | `distribution_audit.csv` | PASS |
| linear | Motion | FastDiffEM | 13.29, 0.3084, 0.5213, 2.92e-05 | `distribution_audit.csv` | PASS |
| linear | Motion | DMPlug | 18.43, 0.5418, 0.4376, 0.000171 | `distribution_audit.csv` | PASS |
| linear | Motion | LCC | 20.09, 0.5696, 0.3519, 4.203e-06 | `ldcc_full_metrics.csv` | PASS |
| nonlinear | FFHQ | SelfDeblur | 17.1343, 0.4879, 0.3192 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | DeblurGAN | 27.2708, 0.2641, 0.7969 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | BKSE | 23.3811, 0.4094, 0.6681 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | FFTFormer | 23.5855, 0.2824, 0.7569 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | DeepRFT | 27.6971, 0.2477, 0.8195 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | BlindDPS | 12.3113, 0.502, 0.3771 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | FastDiffEM | 14.9957, 0.4901, 0.3638 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | DMPlug | 23.6289, 0.3301, 0.6865 | `distribution_audit.csv` | PASS |
| nonlinear | FFHQ | LCC | 23.6856, 0.2669, 0.702 | `ldcc_full_metrics.csv` | PASS |
| nonreference | ImageNet | SelfDeblur | 7.21, 5.372, 0.0893 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | DeblurGAN | 7.006, 4.162, 0.1477 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | BKSE | 6.516, 5.69, 0.1048 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | FFTFormer | 7.874, 3.722, 0.1693 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | DeepRFT | 8.7, 3.657, 0.1373 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | BlindDPS | 9.057, 4.659, 0.2313 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | FastDiffEM | 10.14, 4.85, 0.1827 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | DMPlug | 9.947, 3.173, 0.1212 | `NR_IQA_results.csv` | PASS |
| nonreference | ImageNet | LCC | 7.476, 5.163, 0.1539 | `NR_IQA_results.csv` | PASS |
| ablation | Gaussian development | Single branch | 10.2448, 2.1533, 34 | `super_parameter.xlsx` | PASS |
| ablation | Gaussian development | Single branch + HQS | 10.9856, 2.15179e-06, 36 | `super_parameter.xlsx` | PASS |
| ablation | Gaussian development | Dual parallel | 23.6779, 2.0685e-06, 37 | `super_parameter.xlsx` | PASS |
| ablation | Gaussian development | DP + HQS + l1 | 23.6789, 2.0685e-06, 40 | `super_parameter.xlsx` | PASS |
| ablation | Gaussian development | DP + HQS + l2 | 23.6788, 2.06857e-06, 40 | `super_parameter.xlsx` | PASS |
| ablation | Gaussian development | DP + HQS + PnP | 25.1036, 2.04749e-06, 61 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_lpips | B2 | 0.2329 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_lpips | C2 | 0.2275 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_lpips | D2 | 0.2261 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_lpips | E2 | 0.2314 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_lpips | F2 | 0.235 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_kmse | B3 | 2.07018e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_kmse | C3 | 2.05704e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_kmse | D3 | 1.998e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_kmse | E3 | 2.04727e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_x_kmse | F3 | 2.04872e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_lpips | B8 | 0.2353 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_lpips | C8 | 0.2263 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_lpips | D8 | 0.2261 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_lpips | E8 | 0.2288 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_lpips | F8 | 0.2307 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_kmse | B9 | 2.04528e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_kmse | C9 | 2.04944e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_kmse | D9 | 1.998e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_kmse | E9 | 2.05255e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | eta_k_kmse | F9 | 2.05308e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_lpips | B13 | 0.2271 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_lpips | C13 | 0.227 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_lpips | D13 | 0.2261 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_lpips | E13 | 0.2263 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_lpips | F13 | 0.2298 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_kmse | B14 | 2.04854e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_kmse | C14 | 2.04988e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_kmse | D14 | 1.998e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_kmse | E14 | 2.05328e-06 | `super_parameter.xlsx` | PASS |
| sensitivity | lambda_kmse | F14 | 2.05234e-06 | `super_parameter.xlsx` | PASS |
| regularizer | Gaussian multi-seed | L1 | 25.0025, 0.7214, 0.2172, 1.492e-06 | `pnp_multiseed.csv` | PASS |
| regularizer | Gaussian multi-seed | PnP | 26.631, 0.751, 0.205, 1.358e-06 | `pnp_multiseed.csv` | PASS |
| schedule | Gaussian multi-seed | baseline 200/100, ODE 5 | 25.0025, 0.2938, 0.2172, 1.492e-06 | `loop_multiseed_summary.csv` | PASS |
| schedule | Gaussian multi-seed | outer 400, MCMC 50, 2x step | 24.6899, 0.2146, 0.2193, 1.496e-06 | `loop_multiseed_summary.csv` | PASS |
| proposal | KernelWizard diagnostic | one stochastic proposal | 20.2294, 0.525, 0.4841, 5.493 | `proposal_tuning.csv` | PASS |
| proposal | KernelWizard diagnostic | four proposals, deterministic mean | 17.2974, 0.4597, 0.5455, 1.545 | `proposal_tuning.csv` | PASS |
| proposal | KernelWizard diagnostic | four proposals, 0.5x noise | 19.727, 0.5142, 0.4909, 2.635 | `proposal_tuning.csv` | PASS |
| proposal | KernelWizard diagnostic | four proposals, 1.0x noise | 19.4015, 0.4988, 0.5086, 5.728 | `proposal_tuning.csv` | PASS |
| cross-kernel PnP | 3 seeds | PnP | 25.2441, 0.7356, 0.2228, 2.125e-06, 300 | `full_pnp.json` | PASS |
| fixed-input PnP | Gaussian | corrected PnP | 25.1781, 0.7327, 0.2232, 2.205e-06, 400 | `fixed_kernel_pnp.json` | PASS |
| fixed-input PnP | Motion | corrected PnP | 20.0879, 0.5696, 0.3519, 4.203e-06, 786 | `fixed_kernel_pnp.json` | PASS |
| autotune | Gaussian 4-image pilot | GP-EI | 26.5444, 0.2018, 27.6135, 0.1897, -0.3637, 1 | `autotune_lrktau.json` | PASS |
| paired | Gaussian | DeepRFT | -0.23, -0.5, 0.03, 0.329 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Gaussian | DeblurGAN | -0.15, -0.38, 0.07, 0.403 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Gaussian | BlindDPS | 0.74, 0.43, 1.1, 2.16e-06 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Gaussian | DMPlug | 1.29, 0.78, 1.86, 1.55e-05 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Motion | DeepRFT | -2.84, -3.47, -2.27, 2.69e-17 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Motion | DeblurGAN | -1.1, -1.64, -0.58, 0.000102 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Motion | BlindDPS | 0.29, -0.24, 0.83, 0.0542 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Motion | DMPlug | 1.66, 1.07, 2.26, 2.07e-07 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Nonlinear | DeepRFT | -4.01, -4.52, -3.53, 1.7e-17 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Nonlinear | DeblurGAN | -3.59, -4.02, -3.16, 5.6e-18 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Nonlinear | FFTFormer | 0.1, -0.38, 0.55, 0.258 | `paired_comparisons_fixed_pnp.csv` | PASS |
| paired | Nonlinear | DMPlug | 0.06, -0.71, 0.88, 0.509 | `paired_comparisons_fixed_pnp.csv` | PASS |
| ensemble | Gaussian | 1-sample average | 25.162, 0.7303, 0.2237, 0, 0 | `remote_fixed_pnp_ensemble/gaussian/summary.json` | PASS |
| ensemble | Gaussian | 2-sample average | 25.668, 0.7604, 0.2147, 0.506, 99 | `remote_fixed_pnp_ensemble/gaussian/summary.json` | PASS |
| ensemble | Gaussian | 4-sample average | 25.971, 0.7773, 0.2124, 0.809, 100 | `remote_fixed_pnp_ensemble/gaussian/summary.json` | PASS |
| ensemble | Motion | 1-sample average | 20.082, 0.5683, 0.3522, 0, 0 | `remote_fixed_pnp_ensemble/motion/summary.json` | PASS |
| ensemble | Motion | 2-sample average | 20.86, 0.6105, 0.3421, 0.777, 200 | `remote_fixed_pnp_ensemble/motion/summary.json` | PASS |
| ensemble | Motion | 4-sample average | 21.539, 0.6411, 0.3377, 1.456, 199 | `remote_fixed_pnp_ensemble/motion/summary.json` | PASS |

All table rows use full available samples. Filtered values are reported separately in the failure-aware audit.
