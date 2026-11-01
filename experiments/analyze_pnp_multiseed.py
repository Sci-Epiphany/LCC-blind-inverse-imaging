"""Paired L1-versus-PnP kernel regularizer audit."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from statistics import mean, pstdev


HERE = Path(__file__).resolve().parent
ROOT = HERE / "remote_loop_tuning"
RESULTS = HERE / "results"
SEED_RE = re.compile(r"^seed(\d+)_")


def bootstrap_ci(values: list[float], draws: int = 5000) -> tuple[float, float]:
    rng = random.Random(20261031)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def read_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "psnr": mean(row[0] for row in data["image"]["psnr"]["sample"]),
        "ssim": mean(row[0] for row in data["image"]["ssim"]["sample"]),
        "lpips": mean(row[0] for row in data["image"]["lpips"]["sample"]),
        "kmse": mean(row[0] for row in data["kernel"]["kernel_mse"]["sample"]),
    }


def main() -> None:
    baseline = {}
    pnp = {}
    for directory in ROOT.glob("seed*_baseline_200m100_ode5"):
        match = SEED_RE.match(directory.name)
        if match:
            data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            baseline[int(match.group(1))] = {
                "psnr": [row[0] for row in data["image"]["psnr"]["sample"]],
                "ssim": [row[0] for row in data["image"]["ssim"]["sample"]],
                "lpips": [row[0] for row in data["image"]["lpips"]["sample"]],
                "kmse": [row[0] for row in data["kernel"]["kernel_mse"]["sample"]],
            }
    for directory in ROOT.glob("seed*_paper200m100_pnp"):
        match = SEED_RE.match(directory.name)
        if match:
            data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            pnp[int(match.group(1))] = {
                "psnr": [row[0] for row in data["image"]["psnr"]["sample"]],
                "ssim": [row[0] for row in data["image"]["ssim"]["sample"]],
                "lpips": [row[0] for row in data["image"]["lpips"]["sample"]],
                "kmse": [row[0] for row in data["kernel"]["kernel_mse"]["sample"]],
            }

    seeds = sorted(set(baseline) & set(pnp))
    rows = []
    for name, values in (("L1", baseline), ("PnP", pnp)):
        flat = [values[seed][metric] for seed in seeds for metric in ("psnr",)]
        seed_means = [mean(values[seed]["psnr"]) for seed in seeds]
        rows.append(
            {
                "regularizer": name,
                "seeds": len(seeds),
                "images": 4,
                "cases": sum(map(len, flat)),
                "psnr": mean(value for group in flat for value in group),
                "cv": pstdev(seed_means),
                "ssim": mean(
                    value for seed in seeds for value in values[seed]["ssim"]
                ),
                "lpips": mean(
                    value for seed in seeds for value in values[seed]["lpips"]
                ),
                "kmse": mean(
                    value for seed in seeds for value in values[seed]["kmse"]
                ),
            }
        )

    deltas = [
        pnp[seed]["psnr"][index] - baseline[seed]["psnr"][index]
        for seed in seeds
        for index in range(4)
    ]
    kmse_ratios = [
        pnp[seed]["kmse"][index] / baseline[seed]["kmse"][index]
        for seed in seeds
        for index in range(4)
    ]
    ci_low, ci_high = bootstrap_ci(deltas)
    comparison = {
        "pairs": len(deltas),
        "delta_psnr": mean(deltas),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": sum(value > 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
        "mean_kmse_ratio": mean(kmse_ratios),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "pnp_multiseed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "pnp_multiseed.json").write_text(
        json.dumps({"summary": rows, "paired": comparison}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Paired multi-seed PnP regularizer audit",
        "",
        f"Seeds: {', '.join(map(str, seeds))}; four images per seed.",
        "",
        "| Regularizer | Seeds | Cases | PSNR | Seed SD | SSIM | LPIPS | K-MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regularizer']} | {row['seeds']} | {row['cases']} | "
            f"{row['psnr']:.4f} | {row['cv']:.4f} | {row['ssim']:.4f} | "
            f"{row['lpips']:.4f} | {row['kmse']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"Paired PnP-L1 PSNR delta: **{comparison['delta_psnr']:+.4f} dB** "
            f"(95% CI [{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]), "
            f"wins/losses {comparison['wins']}/{comparison['losses']}.",
            f"Mean PnP/L1 kernel-MSE ratio: {comparison['mean_kmse_ratio']:.3f}.",
        ]
    )
    (RESULTS / "pnp_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
