"""Build the selective LCC revision package with provenance data."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = ROOT / "LCC_revision_updated.zip"

ROOT_FILES = [
    "LCC.tex",
    "LCC.pdf",
    "LCC.bbl",
    "LCC_supplement.tex",
    "LCC_supplement.pdf",
    "LCC_supplement.bbl",
    "refs.bib",
    "zoteroLib.bib",
    "IEEEtran.cls",
    "THEORY_REVISION.md",
    "SOURCE_AUDIT.md",
    "SOURCE_HASHES.txt",
    "CODE_THEORY_AUDIT.md",
    "METHOD_CODE_MAP.md",
    "FIGURE_AUDIT.md",
    "KERNEL_BRANCH_RETHINK.md",
    "Graphical_Abstract_Caption.txt",
]

FIGURES = [
    "LCC_framework_synced.png",
    "LCC_framework_synced.vsdx",
    "deblur_comparison_lcc.pdf",
    "nonlinear_deblur_comparison_lcc.pdf",
    "Graphical_Abstract_final.png",
    "Graphical_Abstract_final.pdf",
]

REMOTE_CODE = [
    "sampler.py",
]

SCRIPTS = [
    "README.md",
    "AUTO_TUNING.md",
    "analyze_existing_results.py",
    "audit_paper_tables.py",
    "build_graphical_abstract.py",
    "build_qualitative_figures.py",
    "build_release_package.py",
    "collect_ldcc_metrics.py",
    "evaluate_seed_ensemble.py",
    "paired_comparisons.py",
    "remote_evaluate_seed_ensemble.py",
    "remote_ensemble_size_ablation.py",
    "ensemble_size_ablation.py",
    "run_controller_ablation.py",
    "analyze_loop_tuning.py",
    "analyze_pnp_multiseed.py",
    "analyze_proposal_tuning.py",
    "analyze_full_pnp.py",
    "analyze_fixed_kernel_pnp.py",
    "analyze_autotune_lrktau.py",
    "auto_tune_lrktau.py",
    "paired_comparisons_pnp.py",
    "posterior_sample_fixed_input.py",
    "remote_evaluate_fixed_pnp_ensemble.py",
    "run_remote_fixed_gaussian_ensemble.sh",
    "run_remote_fixed_motion_ensemble.sh",
    "run_remote_full_pnp.sh",
    "run_remote_full_pnp_multiseed.sh",
    "run_remote_full_pnp_seed42_runs4.sh",
    "run_remote_full_motion_pnp_multiseed.sh",
    "run_remote_pnp_ablation.sh",
    "run_remote_pnp_multiseed.sh",
    "run_remote_pnp_parameter_sweep.sh",
    "run_remote_proposal_mean.sh",
    "run_remote_proposal_single05.sh",
    "run_remote_proposal_sweep.sh",
    "run_remote_loop_sweep.sh",
    "run_remote_multiseed_sweep.sh",
    "run_remote_multiseed_validation.sh",
    "recompute_kernel_metrics.py",
    "refine_saved_kernels.py",
    "evaluate_image_kernel_reconciliation.py",
    "evaluate_image_fusion.py",
    "evaluate_kernel_fusion.py",
    "evaluate_no_reference_candidates.py",
    "analyze_lcc_optimization.py",
    "run_remote_kernel_ablation.sh",
    "run_remote_kernel_jacobian.sh",
    "run_remote_kernel_adaptive.sh",
    "run_remote_kernel_block.sh",
    "run_remote_motion_jacobian.sh",
    "run_remote_nonlinear_ablation.sh",
    "analyze_nonlinear_optimization.py",
    "evaluate_repeated_run_fusion.py",
    "evaluate_multiseed_image_fusion.py",
    "inspect_kernelwizard_uncertainty.py",
]


def add_tree(entries: list[tuple[Path, str]], directory: Path, arc_prefix: str) -> None:
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        entries.append((path, f"{arc_prefix}/{path.relative_to(directory).as_posix()}"))


def main() -> None:
    entries: list[tuple[Path, str]] = []
    missing: list[Path] = []

    def add(path: Path, arcname: str) -> None:
        if path.is_file():
            entries.append((path, arcname))
        else:
            missing.append(path)

    for name in ROOT_FILES:
        add(ROOT / name, name)
    for name in FIGURES:
        add(ROOT / "Figures" / name, f"Figures/{name}")
    for name in REMOTE_CODE:
        add(ROOT / "remote_code" / name, f"remote_code/{name}")
    for name in SCRIPTS:
        add(HERE / name, f"experiments/{name}")

    add_tree(entries, HERE / "source_data", "experiments/source_data")
    add_tree(entries, HERE / "lcc_opt_results", "experiments/lcc_opt_results")
    add_tree(entries, HERE / "optimized_figures", "experiments/optimized_figures")
    add_tree(entries, HERE / "remote_ensemble_result", "experiments/remote_ensemble_result")
    add_tree(
        entries,
        HERE / "remote_ensemble_size_result",
        "experiments/remote_ensemble_size_result",
    )
    for path in sorted((HERE / "results").iterdir()):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
            add(path, f"experiments/results/{path.name}")
    for root_name in ("remote_full_pnp_results", "remote_paper_pnp_results"):
        root = HERE / root_name
        if root.exists():
            for path in sorted(root.glob("seed*/metrics.json")):
                add(
                    path,
                    f"experiments/{root_name}/{path.parent.name}/metrics.json",
                )
    parameter_root = HERE / "remote_parameter_sweep"
    if parameter_root.exists():
        for path in sorted(parameter_root.glob("*/metrics.json")):
            add(
                path,
                f"experiments/remote_parameter_sweep/{path.parent.name}/metrics.json",
            )
        for path in sorted(parameter_root.glob("*/config.yaml")):
            add(
                path,
                f"experiments/remote_parameter_sweep/{path.parent.name}/config.yaml",
            )
    ensemble_root = HERE / "remote_fixed_pnp_ensemble"
    if ensemble_root.exists():
        for path in sorted(ensemble_root.glob("*/summary.json")):
            add(
                path,
                f"experiments/remote_fixed_pnp_ensemble/{path.parent.name}/summary.json",
            )

    audit_path = HERE / "results" / "distribution_audit.csv"
    with audit_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["source"])
            destination = (
                "experiments/comparison_source_data/"
                f"{row['benchmark']}/{row['method']}/{source.name}"
            )
            add(source, destination)

    add(ROOT.parent / "LCC_code_repo.zip", "code/LCC_code_repo.zip")

    if missing:
        raise FileNotFoundError("Missing release inputs:\n" + "\n".join(map(str, missing)))

    forbidden = ("archive_out_of_scope", "aberration", "crossdomain", "rotation")
    leaked = [arc for _, arc in entries if any(token in arc.lower() for token in forbidden)]
    if leaked:
        raise RuntimeError("Out-of-scope artifacts selected:\n" + "\n".join(leaked))

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, arcname in sorted(entries, key=lambda item: item[1]):
            archive.write(path, arcname)
        digest_rows = []
        for path, arcname in sorted(entries, key=lambda item: item[1]):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digest_rows.append(f"{digest}  {arcname}")
        archive.writestr("ZIP_MANIFEST.sha256", "\n".join(digest_rows) + "\n")

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    sidecar = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {OUTPUT.name}\n", encoding="ascii")
    print(f"{OUTPUT.name}: {len(entries)} files, {OUTPUT.stat().st_size} bytes")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
