# Method-to-code map

This note records the evidence boundary between the paper formulation and the
released benchmark implementation.

| Paper component | Code location | Reported benchmark use |
|---|---|---|
| Probability-flow prior prediction | `cores/scheduler.py`, `sampler.py` | Image and operator prediction at every annealing level |
| Linear joint image/operator posterior update | `cores/mcmc.py` | `BlindDAPS`; observation gradient, Gaussian prior anchor, and HQS kernel regularization |
| Nonlinear image/operator posterior update | `cores/mcmc_nl.py` | `Nonlinear_DAPS`; BKSE KernelWizard operator solution |
| Gaussian and motion kernels | `forward_operator/__init__.py`, `forward_operator/blind_blur_utils.py` | Linear operator and adjoint/proximal calculations |
| BKSE KernelWizard branch | `forward_operator/bkse/` | Encoded nonlinear operator prediction |
| Metrics | `eval.py` | PSNR, SSIM, LPIPS, and kernel MSE |
| Configurations | `configs/paper/*.yaml` | Exact recovered settings for the three benchmarks |
| Kernel Jacobian | `forward_operator/__init__.py`, `cores/mcmc.py` | Pull the observation gradient through nonnegativity and simplex normalization |
| Terminal consistency | `experiments/refine_saved_kernels.py`, `experiments/evaluate_image_kernel_reconciliation.py` | Fixed-image kernel solve followed by fixed-kernel image reconciliation |
| Two-branch fusion | `experiments/evaluate_image_fusion.py`, `experiments/evaluate_kernel_fusion.py` | Equal, ground-truth-free averaging of raw- and normalized-coordinate outputs |

The main implementation performs a shared outer anneal and a joint inner
posterior update. Image and operator steps have separate learning rates but
are evaluated in the same Langevin loop.

The optional slow-operator trust-region schedule is a conceptual extension.
Its experimental files belong to a separate spatial-inconsistency study and
are not part of this repository or the paper's main results.

The signed residual-Jacobian identity in the paper supplies a common
control-theoretic interpretation. The raw coordinate differentiates through
the explicit kernel nonnegativity and normalization maps. The normalized
coordinate preserves the scale-separated update used as the second fusion
branch. Neither implementation forms the full state Jacobian.

For the linear branch, `task.pixel.mcmc_sampler_config.kernel_regularizer`
selects the HQS kernel regularizer. The paper default is `pnp`; `l1` and
`l2` remain available only for controlled ablation.

Kernel MSE is computed after independently normalizing each predicted and
target kernel. The earlier whole-batch normalization is not used.
