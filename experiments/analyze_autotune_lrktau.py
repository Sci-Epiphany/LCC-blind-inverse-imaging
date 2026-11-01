"""Summarize completed or partially completed GP tuning runs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean

import yaml


HERE = Path(__file__).resolve().parent
PNP = HERE / "remote_parameter_sweep"
RESULTS = HERE / "results"
BASELINES = HERE / "remote_loop_tuning"


def read_metrics(directory: Path) -> dict[str, float]:
    data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return {
        "psnr": mean(row[0] for row in data["image"]["psnr"]["sample"]),
        "ssim": mean(row[0] for row in data["image"]["ssim"]["sample"]),
        "lpips": mean(row[0] for row in data["image"]["lpips"]["sample"]),
        "kmse": mean(row[0] for row in data["kernel"]["kernel_mse"]["sample"]),
    }


def objective(row: dict[str, float]) -> float:
    return (
        row["psnr"]
        + 10.0 * row["ssim"]
        - 10.0 * row["lpips"]
        - 0.2 * math.log10(row["kmse"] / 1.0e-6)
    )


def read_candidate(directory: Path) -> dict[str, float]:
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    sampler = config["task"]["pixel"]["mcmc_sampler_config"]
    return {
        "lr_kernel": float(sampler["lr"]["kernel"]),
        "tau": float(sampler["tau"]),
    }


def main() -> None:
    rows = []
    for directory in sorted(PNP.glob("autotune_*")):
        if "validate" in directory.name:
            continue
        metrics = read_metrics(directory)
        rows.append(
            {
                "name": directory.name,
                **read_candidate(directory),
                **metrics,
                "objective": objective(metrics),
            }
        )

    cached = [
        (PNP / "seed42_pnp_lrk2e9_tau001", 2e-9, 0.01),
        (HERE / "remote_loop_tuning" / "seed42_paper200m100_pnp", 4e-9, 0.02),
    ]
    for directory, lr_kernel, tau in cached:
        metrics = read_metrics(directory)
        rows.append(
            {
                "name": directory.name,
                "lr_kernel": lr_kernel,
                "tau": tau,
                **metrics,
                "objective": objective(metrics),
            }
        )

    validation = []
    for directory in sorted(PNP.glob("autotune_validate_*_seed*")):
        if not (directory / "metrics.json").is_file():
            continue
        match = re.match(r"autotune_validate_(\d+)_seed(\d+)$", directory.name)
        if not match:
            continue
        rank, seed_text = match.groups()
        seed = int(seed_text)
        metrics = read_metrics(directory)
        baseline = read_metrics(BASELINES / f"seed{seed}_paper200m100_pnp")
        validation.append(
            {
                "rank": int(rank),
                "seed": seed,
                **read_candidate(directory),
                "delta_psnr": metrics["psnr"] - baseline["psnr"],
                "delta_lpips": metrics["lpips"] - baseline["lpips"],
                "delta_objective": objective(metrics) - objective(baseline),
                "metrics": metrics,
            }
        )

    rows.sort(key=lambda row: row["objective"], reverse=True)
    validation_ranking = []
    for rank in sorted({row["rank"] for row in validation}):
        group = [row for row in validation if row["rank"] == rank]
        validation_ranking.append(
            {
                "rank": rank,
                "lr_kernel": group[0]["lr_kernel"],
                "tau": group[0]["tau"],
                "seeds": [row["seed"] for row in group],
                "mean_delta_psnr": mean(row["delta_psnr"] for row in group),
                "mean_delta_lpips": mean(row["delta_lpips"] for row in group),
                "mean_delta_objective": mean(
                    row["delta_objective"] for row in group
                ),
                "wins": sum(row["delta_objective"] > 0 for row in group),
            }
        )
    validation_ranking.sort(
        key=lambda row: row["mean_delta_objective"], reverse=True
    )
    payload = {
        "search_runs": rows,
        "validation": validation,
        "validation_ranking": validation_ranking,
        "selected": rows[0] if rows else None,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "autotune_lrktau.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Automatic lr_k / tau search",
        "",
        "| Candidate | lr_k | tau | PSNR | SSIM | LPIPS | K-MSE | Objective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['lr_kernel']:.4e} | {row['tau']:.5f} | "
            f"{row['psnr']:.4f} | {row['ssim']:.4f} | {row['lpips']:.4f} | "
            f"{row['kmse']:.3e} | {row['objective']:.4f} |"
        )
    if validation:
        lines.extend(
            [
                "",
                "Held-out validation by candidate rank:",
                "",
                "| Rank | lr_k | tau | Seed | Delta PSNR | Delta LPIPS | Delta objective |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in validation:
            lines.append(
                f"| {row['rank']} | {row['lr_kernel']:.4e} | {row['tau']:.5f} | "
                f"{row['seed']} | {row['delta_psnr']:+.4f} | "
                f"{row['delta_lpips']:+.4f} | {row['delta_objective']:+.4f} |"
            )
        lines.extend(
            [
                "",
                "| Rank | lr_k | tau | Seeds | mean Delta PSNR | mean Delta LPIPS | mean Delta objective | Wins |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in validation_ranking:
            lines.append(
                f"| {row['rank']} | {row['lr_kernel']:.4e} | {row['tau']:.5f} | "
                f"{len(row['seeds'])} | {row['mean_delta_psnr']:+.4f} | "
                f"{row['mean_delta_lpips']:+.4f} | "
                f"{row['mean_delta_objective']:+.4f} | "
                f"{row['wins']}/{len(row['seeds'])} |"
            )
    (RESULTS / "autotune_lrktau.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
