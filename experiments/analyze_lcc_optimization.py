"""Aggregate the corrected-Jacobian and consistency-refinement experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, pstdev


HERE = Path(__file__).resolve().parent
OPT = HERE / "lcc_opt_results"
LOOP = HERE / "remote_loop_tuning"
SOURCE = HERE / "remote_paper_pnp_results"
RESULTS = HERE / "results"
SEEDS = (42, 7, 2026, 123)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_ci(values: list[float], draws: int = 10000) -> tuple[float, float]:
    rng = random.Random(20261101)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def paired_delta(
    candidate: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
    metric: str,
    higher_is_better: bool,
) -> dict[str, float | int]:
    names = sorted(set(candidate) & set(reference))
    deltas = [
        candidate[name][metric] - reference[name][metric]
        for name in names
    ]
    low, high = bootstrap_ci(deltas)
    return {
        "delta": mean(deltas),
        "ci_low": low,
        "ci_high": high,
        "wins": sum(
            value > 0 if higher_is_better else value < 0
            for value in deltas
        ),
        "losses": sum(
            value < 0 if higher_is_better else value > 0
            for value in deltas
        ),
        "cases": len(deltas),
    }


def image_reference(metrics_path: Path) -> dict[str, dict[str, float]]:
    data = load(metrics_path)
    return {
        f"{index:05d}_run{run:04d}.png": {
            "psnr": data["image"]["psnr"]["sample"][index][run],
            "ssim": data["image"]["ssim"]["sample"][index][run],
            "lpips": data["image"]["lpips"]["sample"][index][run],
        }
        for index in range(len(data["image"]["psnr"]["sample"]))
        for run in range(len(data["image"]["psnr"]["sample"][index]))
    }


def main() -> None:
    multiseed = {"seeds": {}, "aggregate": {}}
    flat_candidate = {}
    flat_reference = {}
    for seed in SEEDS:
        image_summary = load(OPT / f"seed{seed}_post_image.json")["regularization"]["0.1"]
        kernel_summary = load(OPT / f"seed{seed}_post_kernel.json")
        seed_values = {
            "psnr": image_summary["psnr"],
            "ssim": image_summary["ssim"],
            "lpips": image_summary["lpips"],
            "kernel_mse": kernel_summary["refined_mse"],
        }
        multiseed["seeds"][str(seed)] = seed_values
        baseline_path = LOOP / f"seed{seed}_paper200m100_pnp" / "metrics.json"
        baseline = load(baseline_path)["image"]
        flat_reference[str(seed)] = {
            "psnr": mean(baseline["psnr"]["sample"][0]),
            "ssim": mean(baseline["ssim"]["sample"][0]),
            "lpips": mean(baseline["lpips"]["sample"][0]),
        }
        flat_candidate[str(seed)] = seed_values

    for metric in ("psnr", "ssim", "lpips", "kernel_mse"):
        values = [entry[metric] for entry in multiseed["seeds"].values()]
        multiseed["aggregate"][metric] = {
            "mean": mean(values),
            "std": pstdev(values),
        }
    multiseed["aggregate"]["image_paired_vs_baseline"] = {
        metric: paired_delta(
            flat_candidate,
            flat_reference,
            metric,
            higher_is_better=metric in {"psnr", "ssim"},
        )
        for metric in ("psnr", "ssim", "lpips")
    }

    gaussian_image = load(OPT / "gaussian_s123_post_image.json")["regularization"]["0.1"]
    gaussian_kernel = load(OPT / "gaussian_s123_post_kernel.json")
    gaussian_initial = load(OPT / "gaussian_s123_initial_metrics.json")
    gaussian_initial_kernel = load(OPT / "gaussian_s123_initial_kernel.json")
    gaussian_image_only = load(
        OPT / "gaussian_s123_image_only.json"
    )["regularization"]["0.1"]
    gaussian_fusion = load(
        OPT / "gaussian_two_branch_image_fusion.json"
    )["fusion"]["0.50"]
    gaussian_kernel_fusion = load(
        OPT / "gaussian_two_branch_kernel_fusion.json"
    )["fusion"]["0.50"]["kernel_mse"]
    old_gaussian_image = image_reference(
        SOURCE / "gaussian_seed42_runs4" / "metrics.json"
    )
    old_gaussian_image = {
        name: values
        for name, values in old_gaussian_image.items()
        if name.endswith("_run0000.png")
    }
    old_gaussian_kernel = load(OPT / "old_gaussian_kernel_corrected.json")
    old_gaussian_kernel = {
        f"{int(index):05d}_run0000.png": {
            "kernel_mse": old_gaussian_kernel["per_image"][index]["mean"]
        }
        for index in old_gaussian_kernel["per_image"]
    }
    gaussian_full = {
        "initial": {
            metric: mean(gaussian_initial["image"][metric]["mean"])
            for metric in ("psnr", "ssim", "lpips")
        },
        "post_kernel_mse": gaussian_kernel["refined_mse"],
        "initial_kernel_mse": gaussian_initial_kernel["kernel_mse"],
        "image_only": {
            metric: gaussian_image_only[metric]
            for metric in ("psnr", "ssim", "lpips")
        },
        "post_image": {
            metric: gaussian_image[metric]
            for metric in ("psnr", "ssim", "lpips")
        },
        "paired_image_vs_old_pnp": {
            metric: paired_delta(
                gaussian_image["samples"],
                old_gaussian_image,
                metric,
                higher_is_better=metric in {"psnr", "ssim"},
            )
            for metric in ("psnr", "ssim", "lpips")
        },
        "two_branch_equal_fusion": {
            "psnr": gaussian_fusion["psnr"],
            "ssim": gaussian_fusion["ssim"],
            "lpips": gaussian_fusion["lpips"],
            "kernel_mse": gaussian_kernel_fusion,
            "paired_image_vs_old_pnp": {
                metric: paired_delta(
                    gaussian_fusion["samples"],
                    old_gaussian_image,
                    metric,
                    higher_is_better=metric in {"psnr", "ssim"},
                )
                for metric in ("psnr", "ssim", "lpips")
            },
        },
    }

    motion_image = load(OPT / "motion_post_image.json")["regularization"]["0.03"]
    motion_kernel = load(OPT / "motion_post_kernel.json")
    old_motion_image = image_reference(
        SOURCE / "motion_seed42_runs4" / "metrics.json"
    )
    motion_full = {
        "initial": {
            "psnr": mean([
                value
                for row in load(SOURCE / "motion_seed42_runs4" / "metrics.json")["image"]["psnr"]["sample"]
                for value in row
            ]),
        },
        "post_kernel_mse": motion_kernel["refined_mse"],
        "post_image": {
            metric: motion_image[metric]
            for metric in ("psnr", "ssim", "lpips")
        },
        "paired_image_vs_old_pnp": {
            metric: paired_delta(
                motion_image["samples"],
                old_motion_image,
                metric,
                higher_is_better=metric in {"psnr", "ssim"},
            )
            for metric in ("psnr", "ssim", "lpips")
        },
    }

    payload = {
        "four_seed_gaussian": multiseed,
        "full_gaussian_100_seed123": gaussian_full,
        "full_motion_200x4": motion_full,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lcc_optimization.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Corrected Jacobian and consistency refinement",
        "",
        "## Four-seed Gaussian pilot",
        "",
        "| Seed | PSNR | SSIM | LPIPS | K-MSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, values in multiseed["seeds"].items():
        lines.append(
            f"| {seed} | {values['psnr']:.4f} | {values['ssim']:.4f} | "
            f"{values['lpips']:.4f} | {values['kernel_mse']:.3e} |"
        )
    aggregate = multiseed["aggregate"]
    lines.extend([
        "",
        f"Mean PSNR **{aggregate['psnr']['mean']:.4f} ± "
        f"{aggregate['psnr']['std']:.4f} dB**; K-MSE "
        f"**{aggregate['kernel_mse']['mean']:.3e}**.",
        "",
        "| Full Gaussian 100 images | PSNR | SSIM | LPIPS | K-MSE |",
        "|---|---:|---:|---:|---:|",
        f"| Seed 123, two-branch equal fusion | "
        f"{gaussian_full['two_branch_equal_fusion']['psnr']:.4f} | "
        f"{gaussian_full['two_branch_equal_fusion']['ssim']:.4f} | "
        f"{gaussian_full['two_branch_equal_fusion']['lpips']:.4f} | "
        f"{gaussian_full['two_branch_equal_fusion']['kernel_mse']:.3e} |",
        "",
        "| Full motion 200 images × 4 runs | PSNR | SSIM | LPIPS | K-MSE |",
        "|---|---:|---:|---:|---:|",
        f"| Optimized consistency refinement | {motion_full['post_image']['psnr']:.4f} | "
        f"{motion_full['post_image']['ssim']:.4f} | "
        f"{motion_full['post_image']['lpips']:.4f} | "
        f"{motion_full['post_kernel_mse']:.3e} |",
    ])
    (RESULTS / "lcc_optimization.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
