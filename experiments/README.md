# Experiment utilities

This directory contains the scripts and compact metric records for the
additional LCC experiments. Large checkpoints, datasets, generated images, and
paper figures are intentionally excluded.

## Optimized implementation audit

The release includes an exact kernel-coordinate Jacobian, noise-calibrated
likelihood scaling, separate kernel thermal noise, fixed-image kernel
refinement, fixed-kernel image reconciliation, and label-free two-branch
fusion.

```bash
bash run_remote_kernel_ablation.sh 4e-7 20 2000000 0.02 1 100 50 20 optimized 0.1 123
python refine_saved_kernels.py ...
python evaluate_image_kernel_reconciliation.py ...
python analyze_lcc_optimization.py
```

Compact results and the exact result-file mapping are in
`lcc_opt_results/README.md`. The optimized Gaussian fusion obtains
27.5842 dB PSNR and 1.528e-7 per-kernel MSE on 100 fixed inputs. Motion
improves from 20.0879 to 21.1626 dB after the same consistency refinement.
The `trust_region` and `block_simplex` kernel-step modes in the sampler are
retained as negative controls: their monotone local objectives destabilized
the coupled image state, so they are not used by any headline result.

Archived K-MSE values produced by the old whole-batch normalization are not
used. `recompute_kernel_metrics.py` normalizes each predicted and ground-truth
kernel independently.

## Kernel regularizer

The linear sampler supports `kernel_regularizer: pnp|l1|l2`. The paper
configuration uses `pnp`.

```bash
bash run_remote_pnp_multiseed.sh
python analyze_pnp_multiseed.py
```

The paired audit covers four images and eight seeds.

## Annealing schedule

The schedule audit compares `200` outer annealing levels with `100` MCMC
steps against `400/50` with proportionally larger steps.

```bash
bash run_remote_loop_sweep.sh
python analyze_loop_tuning.py
```

The alternative schedule reduces seed variance but does not improve mean
PSNR.

## Fixed-input full reruns

The sampler seed controls both stochastic output and generated test kernels.
`posterior_sample_fixed_input.py` reuses archived measurement inputs when
direct comparison with retained baseline outputs is required.

```bash
bash run_remote_full_pnp_seed42_runs4.sh
bash run_remote_full_motion_pnp_multiseed.sh
python analyze_fixed_kernel_pnp.py
```

The rerun scripts write to paths documented in `docs/CODE_THEORY_AUDIT.md`.

## KernelWizard proposal diagnostic

`run_remote_proposal_sweep.sh` draws multiple BKSE latent proposals and selects
them using observation error. Lower encoded kernel MSE does not imply better
image fidelity, so this negative diagnostic is excluded from the main method.

```bash
bash run_remote_proposal_sweep.sh
python analyze_proposal_tuning.py
```

## Automatic parameter search

`auto_tune_lrktau.py` performs Gaussian-process Expected-Improvement search in
log-`lr_k`/`tau` space and validates candidates on held-out seeds. The complete
procedure and negative held-out result are documented in `AUTO_TUNING.md`.

```bash
CUDA_VISIBLE_DEVICES=0 python auto_tune_lrktau.py
```

## Result tables

`tables/` contains aggregate CSV, JSON, and Markdown records.
`remote_full_pnp_results/` contains the cross-kernel validation metrics.
`remote_paper_pnp_results/` contains the fixed-input full reruns.
`remote_parameter_sweep/` contains the automatic-search trace.
`source_data/` preserves the corrected metric arrays and the archived
pre-correction arrays used as baselines.
