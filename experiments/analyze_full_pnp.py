"""Compare corrected full-data PnP runs with the archived Gaussian runs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PNP_ROOT = HERE / "remote_full_pnp_results"
BASELINE = HERE / "source_data" / "gaussian_metrics_archived_l1.json"


def bootstrap_ci(values: list[float], draws: int = 10000) -> tuple[float, float]:
    rng = random.Random(20261031)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def read_metrics(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "psnr": [row[0] for row in data["image"]["psnr"]["sample"]],
        "ssim": [row[0] for row in data["image"]["ssim"]["sample"]],
        "lpips": [row[0] for row in data["image"]["lpips"]["sample"]],
        "kernel_mse": [row[0] for row in data["kernel"]["kernel_mse"]["sample"]],
    }


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_images = {
        metric: [mean(row) for row in baseline[key][metric]["sample"]]
        for key, metric in (
            ("image", "psnr"),
            ("image", "lpips"),
            ("kernel", "kernel_mse"),
        )
    }

    runs: dict[int, dict[str, list[float]]] = {}
    for path in sorted(PNP_ROOT.glob("seed*/metrics.json")):
        seed = int(path.parent.name.removeprefix("seed"))
        runs[seed] = read_metrics(path)
    if not runs:
        raise FileNotFoundError(f"No full PnP metrics found under {PNP_ROOT}")

    lengths = {len(values["psnr"]) for values in runs.values()}
    if lengths != {100}:
        raise ValueError(f"Expected 100 images per PnP run, found {sorted(lengths)}")

    pnp_image_means = {
        metric: [
            mean(runs[seed][metric][index] for seed in runs)
            for index in range(100)
        ]
        for metric in ("psnr", "ssim", "lpips", "kernel_mse")
    }
    paired = {}
    for metric in ("psnr", "lpips", "kernel_mse"):
        deltas = [
            candidate - reference
            for candidate, reference in zip(
                pnp_image_means[metric], baseline_images[metric], strict=True
            )
        ]
        low, high = bootstrap_ci(deltas)
        higher_is_better = metric == "psnr"
        paired[metric] = {
            "delta": mean(deltas),
            "ci_low": low,
            "ci_high": high,
            "wins": sum(
                value > 0 if higher_is_better else value < 0 for value in deltas
            ),
            "losses": sum(
                value < 0 if higher_is_better else value > 0 for value in deltas
            ),
        }

    rows = []
    for seed, values in runs.items():
        seed_deltas = [
            candidate - reference
            for candidate, reference in zip(
                values["psnr"], baseline_images["psnr"], strict=True
            )
        ]
        low, high = bootstrap_ci(seed_deltas)
        rows.append(
            {
                "run": f"seed{seed}",
                "n_images": len(values["psnr"]),
                "psnr": mean(values["psnr"]),
                "psnr_delta_vs_archived_mean": mean(seed_deltas),
                "psnr_ci_low": low,
                "psnr_ci_high": high,
                "psnr_wins": sum(value > 0 for value in seed_deltas),
                "ssim": mean(values["ssim"]),
                "lpips": mean(values["lpips"]),
                "kernel_mse": mean(values["kernel_mse"]),
            }
        )

    aggregate = {
        "completed_runs": len(runs),
        "seeds": sorted(runs),
        "psnr": mean(value for values in runs.values() for value in values["psnr"]),
        "ssim": mean(value for values in runs.values() for value in values["ssim"]),
        "lpips": mean(value for values in runs.values() for value in values["lpips"]),
        "kernel_mse": mean(
            value for values in runs.values() for value in values["kernel_mse"]
        ),
        "success_10db": sum(
            value > 10.0 for values in runs.values() for value in values["psnr"]
        ),
        "n_cases": sum(len(values["psnr"]) for values in runs.values()),
        "archived_lcc": {
            "psnr": mean(baseline_images["psnr"]),
            "lpips": mean(baseline_images["lpips"]),
            "kernel_mse": mean(baseline_images["kernel_mse"]),
        },
        "paired_per_image_vs_archived_mean": paired,
        "runs": rows,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "full_pnp.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS / "full_pnp_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cross-kernel Gaussian PnP validation",
        "",
        f"Completed full runs: {len(runs)} ({', '.join(map(str, sorted(runs)))}); "
        "100 images per run.",
        "",
        "Each seed also regenerates the test kernels. These runs measure "
        "generalization to new kernel realizations and are not substituted "
        "into the fixed-kernel comparison table.",
        "",
        "| Seeds | N runs | Cases | PSNR | SSIM | LPIPS | K-MSE | SR@10 dB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {len(runs)} | {len(runs)} | {aggregate['n_cases']} | {aggregate['psnr']:.4f} | "
        f"{aggregate['ssim']:.4f} | {aggregate['lpips']:.4f} | "
        f"{aggregate['kernel_mse']:.3e} | "
        f"{aggregate['success_10db']}/{aggregate['n_cases']} |",
        "",
        "Paired per-image difference against the four-run archived mean:",
        "",
        "| Metric | Delta | 95% CI | Wins/Losses |",
        "|---|---:|---:|---:|",
    ]
    for metric, label, format_spec in (
        ("psnr", "PSNR (dB)", ".4f"),
        ("lpips", "LPIPS", ".4f"),
        ("kernel_mse", "K-MSE", ".3e"),
    ):
        value = paired[metric]
        lines.append(
            f"| {label} | {value['delta']:+{format_spec}} | "
            f"[{value['ci_low']:+{format_spec}}, {value['ci_high']:+{format_spec}}] | "
            f"{value['wins']}/{value['losses']} |"
        )
    lines.extend(
        [
            "",
            "| Run | PSNR | Delta vs archived mean | 95% CI | W/L |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['psnr']:.4f} | "
            f"{row['psnr_delta_vs_archived_mean']:+.4f} | "
            f"[{row['psnr_ci_low']:+.4f}, {row['psnr_ci_high']:+.4f}] | "
            f"{row['psnr_wins']}/{100 - row['psnr_wins']} |"
        )
    (RESULTS / "full_pnp.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
