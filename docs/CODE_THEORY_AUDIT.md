# Theory-code consistency audit

Audit date: 2026-10-30.

## Reported implementation

The main benchmark is produced by:

```text
/home/gsn/PhD_thesis/DAPS/sampler.py
  BlindDAPS       -> linear Gaussian/motion
  Nonlinear_DAPS  -> BKSE KernelWizard nonlinear
```

The recovered configurations specify:

- `N_ode=5`;
- `N_A=200`;
- `N=100` Langevin/MCMC steps per annealing level;
- separate image and kernel learning rates;
- a Gaussian prior anchor and HQS kernel regularization in the linear branch;
- a BKSE KernelWizard kernel solution in the nonlinear branch.

## Correspondence

| Paper statement | Code evidence | Status |
|---|---|---|
| Probability-flow prior prediction at each noise level | `sampler.py`: `DiffusionPFODE` in `BlindDAPS.sample` and `Nonlinear_DAPS.sample` | aligned |
| Joint image/operator posterior correction | `cores/mcmc.py`: `MCMCSampler.score_fn`; `cores/mcmc_nl.py`: `MCMCNLSampler.score_fn` | aligned |
| Coordinate-wise image and kernel steps | `get_lr` returns a separate rate for each component | aligned |
| Linear kernel HQS and simplex projection | `MCMCSampler.score_fn`, `regularization_step`, `normalize_kernel` | aligned |
| BKSE KernelWizard nonlinear operator | `forward_operator.NonlinearBlur` and `nonlinear_config.yaml` | aligned |
| `N_A=200`, `N_ode=5`, `N=100` | three recovered configuration files in `experiments/source_data/` | aligned |
| Signed residual-Jacobian theory | `J^T e` analysis in `LCC.tex` and supplement | conceptual formulation; benchmark uses explicit data/prior/regularizer terms |

## Corrections made

- The main text no longer says that the reported benchmark uses a
  two-timescale/frozen-operator controller.
- The framework figure now says `Decoupled annealing`, and the correction loop
  is labeled `Langevin Dynamics`.
- The slow-operator and residual-gated controllers are explicitly marked as
  extensions not used for Tables I and II.
- The nonlinear text no longer claims that linear and nonlinear branches share
  one literal inference controller. They share the formulation and outer
  anneal, but have task-specific operator solutions.
- Files and experiments from `revision_2026/unified_v13_20261022`, including
  rotation, aberration, compound-blur, cross-domain, and spatial-kernel tests,
  are excluded from this article and from the clean code release.

## Remaining limitation

The general signed-residual control objective is broader than the explicit
score decomposition implemented by the benchmark. The paper therefore treats
`J^T e` as the unifying analysis, not as a claim that every Jacobian block is
materialized in both production samplers.

`tests/test_residual_gradient.py` verifies
`grad(0.5 ||e||^2) = J^T e` for a linear observation residual and a Gaussian
prior residual. This is an implementation/mathematics consistency test; the
stability propositions are not claimed to be the source of benchmark gains.
