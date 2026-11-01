"""Analyze the multi-seed outer-loop/MCMC pilot."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


HERE = Path(__file__).resolve().parent
ROOT = HERE / "remote_loop_tuning"
RESULTS = HERE / "results"
PATTERN = re.compile(r"^seed(\d+)_(.+)$")
LABELS = {
    "baseline_200m100_ode5": "baseline 200/100, ODE 5",
    "outer400_mcmc50_x2": "outer 400, MCMC 50, 2x step",
    "ode10_200m50": "ODE 10, MCMC 50",
}


def bootstrap_ci(values: list[float], draws: int = 5000) -> tuple[float, float]:
    rng = random.Random(20261030 + len(values))
    means = sorted(
        mean(rng.choices(values, k=len(values))) for _ in range(draws)
    )

    def quantile(q: float) -> float:
        return means[int(round((len(means) - 1) * q))]

    return quantile(0.025), quantile(0.975)


def main() -> None:
    records = defaultdict(list)
    for path in sorted(ROOT.glob("seed*/metrics.json")):
        match = PATTERN.match(path.parent.name)
        if not match:
            continue
        seed, variant = int(match.group(1)), match.group(2)
        if variant not in LABELS:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        psnr = [row[0] for row in data["image"]["psnr"]["sample"]]
        ssim = [row[0] for row in data["image"]["ssim"]["sample"]]
        lpips = [row[0] for row in data["image"]["lpips"]["sample"]]
        kmse = [row[0] for row in data["kernel"]["kernel_mse"]["sample"]]
        for index, values in enumerate(zip(psnr, ssim, lpips, kmse)):
            records[(seed, index, variant)] = dict(zip(("psnr", "ssim", "lpips", "kmse"), values))

    seeds = sorted({seed for seed, _, _ in records})
    summary_rows = []
    for variant, label in LABELS.items():
        rows = [value for (_, _, name), value in records.items() if name == variant]
        variant_seeds = sorted(
            {seed for (seed, _, name) in records if name == variant}
        )
        if not rows:
            continue
        seed_psnr = [
            mean(
                value["psnr"]
                for (row_seed, _, name), value in records.items()
                if row_seed == seed and name == variant
            )
            for seed in variant_seeds
        ]
        summary_rows.append(
            {
                "variant": label,
                "n_seeds": len(variant_seeds),
                "n_images_per_seed": 4,
                "psnr_mean": mean(value["psnr"] for value in rows),
                "seed_mean_std": pstdev(seed_psnr),
                "ssim_mean": mean(value["ssim"] for value in rows),
                "lpips_mean": mean(value["lpips"] for value in rows),
                "kmse_mean": mean(value["kmse"] for value in rows),
            }
        )

    comparisons = []
    for variant in ("outer400_mcmc50_x2", "ode10_200m50"):
        deltas = []
        kmse_ratios = []
        variant_seeds = sorted(
            {seed for (seed, _, name) in records if name == variant}
        )
        for seed in variant_seeds:
            for image_id in range(4):
                base = records[(seed, image_id, "baseline_200m100_ode5")]
                candidate = records[(seed, image_id, variant)]
                deltas.append(candidate["psnr"] - base["psnr"])
                kmse_ratios.append(candidate["kmse"] / base["kmse"])
        comparisons.append(
            {
                "variant": LABELS[variant],
                "seeds": len(variant_seeds),
                "n_pairs": len(deltas),
                "delta_psnr": mean(deltas),
                "delta_psnr_std": pstdev(deltas),
                "delta_ci_low": bootstrap_ci(deltas)[0],
                "delta_ci_high": bootstrap_ci(deltas)[1],
                "wins": sum(value > 0 for value in deltas),
                "losses": sum(value < 0 for value in deltas),
                "mean_kmse_ratio": mean(kmse_ratios),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "loop_multiseed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (RESULTS / "loop_multiseed_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    (RESULTS / "loop_multiseed.json").write_text(
        json.dumps({"summary": summary_rows, "paired": comparisons}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Multi-seed loop-tuning pilot",
        "",
        f"Four FFHQ images per seed; one run per image/config; seeds "
        f"{', '.join(map(str, seeds))}.",
        "",
        "| Variant | Seeds | Images/seed | PSNR | Seed SD | SSIM | LPIPS | K-MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['n_seeds']} | {row['n_images_per_seed']} | "
            f"{row['psnr_mean']:.4f} | {row['seed_mean_std']:.4f} | "
            f"{row['ssim_mean']:.4f} | {row['lpips_mean']:.4f} | "
            f"{row['kmse_mean']:.3e} |"
        )
    lines.extend(
        [
            "",
            "| Candidate vs baseline | Seeds | Pairs | Delta PSNR [95% CI] | Std | Wins/Losses | K-MSE ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['variant']} | {row['seeds']} | {row['n_pairs']} | "
            f"{row['delta_psnr']:+.4f} [{row['delta_ci_low']:+.4f},{row['delta_ci_high']:+.4f}] | "
            f"{row['delta_psnr_std']:.4f} | {row['wins']}/{row['losses']} | "
            f"{row['mean_kmse_ratio']:.3f} |"
        )
    (RESULTS / "loop_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
