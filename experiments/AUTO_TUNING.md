# Automatic tuning design

The kernel learning rate and data-consistency temperature are not tuned by
manual one-variable-at-a-time scans. `auto_tune_lrktau.py` treats the pair as
a two-dimensional search problem:

- `log10(lr_k)` spans `[1e-9, 1.6e-8]`;
- `tau` spans `[0.005, 0.040]`;
- the sampler, image learning rate, regularizer, and schedule remain fixed.

The first observations include the retained paper setting and the previously
reported sensitive setting. Four design points then cover the space. A small
pure-NumPy Gaussian process with a squared-exponential kernel is fitted in
the normalized two-dimensional space, and Expected Improvement proposes
three refinement points. This avoids introducing a heavyweight tuning
dependency into the research release.

The search objective is

```text
PSNR + 10*SSIM - 10*LPIPS - 0.2*log10(K-MSE / 1e-6).
```

The selected candidates are not accepted from the tuning seed alone. The top
three are rerun on seeds 123, 2026, and 7 and compared with the corresponding
same-seed PnP baselines. Selection uses mean validation-objective improvement
and requires cross-seed wins.

The tuning images are a small validation subset of FFHQ. Consequently, this
procedure is appropriate for selecting development settings, not for
per-image test-time adaptation. The later implementation audit showed that
this search inherited an unaccounted kernel-normalization Jacobian. The
optimized release therefore fixes that Jacobian, sets `tau` from the known
measurement noise, and separates deterministic step scaling from thermal noise.
The old search is retained as a negative control rather than a selection rule.

## Completed search outcome

The GP-EI search found `lr_k=3.5255e-9, tau=0.015316` on the seed-42
development set, reaching 27.6135 dB PSNR and 0.1897 LPIPS versus
26.5444 dB and 0.2018 for the retained setting. On held-out seeds 123, 2026,
and 7, however, the PSNR changes were -0.2580, +0.8150, and -1.6480 dB
(mean -0.3637 dB), with only one of three objective wins. The candidate is
therefore not adopted. This negative result is retained in the supplement
because it directly demonstrates why multi-seed validation is necessary.
