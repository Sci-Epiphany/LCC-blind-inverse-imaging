# Reproducibility notes

## Benchmark protocol

- Dataset: FFHQ for the full-reference linear and nonlinear experiments.
- Test counts: 100 Gaussian, 200 motion, and 100 nonlinear images.
- Image size: 256 x 256.
- Linear kernels: 64 x 64.
- Probability-flow ODE steps: 5.
- Annealing levels: 200.
- Inner Langevin/MCMC steps: 100.
- Noise range: sigma from 100 to 0.1 with a polynomial-7 schedule.
- LCC aggregation: mean over all retained images and four stochastic outputs.
- Baseline aggregation: all available per-image rows; no PSNR threshold.

## Seed and sample averaging

The benchmark stores four LCC outputs per image. The paper reports their
per-image mean and also evaluates a label-free equal-weight four-sample
ensemble. The ensemble uses no ground truth during fusion and costs four
inference passes.

## Failure policy

The original nonlinear BlindDPS and FastDiffEM summary tables omitted cases
below 10 dB. The paper recomputes them over all 100 cases and reports the
success rate separately. Do not use success-conditioned means as competing
headline statistics.

## Exclusions

Rotation, aberration, compound-blur, cross-domain, and Universal Aberration
Correction Network experiments belong to a separate spatial-inconsistency
study. They are not evidence for the results in this repository.
