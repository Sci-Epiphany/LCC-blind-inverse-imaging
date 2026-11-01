"""Aggregate corrected KernelWizard and nonlinear MCMC experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

import numpy as np


HERE = Path(__file__).resolve().parent
NONLINEAR = HERE / "lcc_opt_results" / "nonlinear"
RESULTS = HERE / "results"


def load(name: str):
    return json.loads((NONLINEAR / name).read_text(encoding="utf-8"))


def metrics(data, images: int | None = None):
    rows = slice(None) if images is None else slice(0, images)
    return {
        metric: [
            mean(row)
            for row in data["image"][metric]["sample"][rows]
        ]
        for metric in ("psnr", "ssim", "lpips")
    }


def bootstrap_ci(values: list[float], draws: int = 10000) -> tuple[float, float]:
    rng = random.Random(20261101)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def main() -> None:
    variants = {
        "original 50×20": load("baseline_50m20.json"),
        "BKSE input/sign fix": load("inputfix_50m20.json"),
        "integrated kernel MCMC": load("auto_lr_50m20.json"),
        "image step 3e-5": load("best_50m20.json"),
        "100×50 schedule": load("best_100m50.json"),
    }
    table = {}
    for label, data in variants.items():
        values = metrics(data)
        table[label] = {
            name: float(np.mean(samples))
            for name, samples in values.items()
        }

    extended_variants = {
        "100×50, image noise 1.0": load("best_100m50.json"),
        "image noise 0.5": load("imgnoise_05_100m50.json"),
        "image noise 0.1": load("imgnoise_01_100m50.json"),
        "image noise 0.0": load("imgnoise_00_100m50.json"),
        "observation-selected proposal": load(
            "proposal_select4_05_100m50.json"
        ),
        "post-MCMC kernel reprojection": load(
            "kernel_reprojection_20_100m50.json"
        ),
        "200×25 schedule": load("schedule_200m25.json"),
        "200×50 schedule": load("schedule_200m50.json"),
    }
    extended_table = {}
    for label, data in extended_variants.items():
        values = metrics(data)
        extended_table[label] = {
            name: float(np.mean(samples))
            for name, samples in values.items()
        }

    old_path = HERE / "source_data" / "nonlinear_metrics.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = load("official20_best.json")
    old_values = metrics(old, images=20)
    new_values = metrics(new)
    paired = {}
    for name in ("psnr", "ssim", "lpips"):
        deltas = [
            candidate - reference
            for candidate, reference in zip(
                new_values[name], old_values[name], strict=True
            )
        ]
        low, high = bootstrap_ci(deltas)
        higher_better = name in {"psnr", "ssim"}
        paired[name] = {
            "delta": mean(deltas),
            "ci_low": low,
            "ci_high": high,
            "wins": sum(
                value > 0 if higher_better else value < 0
                for value in deltas
            ),
            "losses": sum(
                value < 0 if higher_better else value > 0
                for value in deltas
            ),
        }

    full20 = {
        "old": {
            name: float(np.mean(values))
            for name, values in old_values.items()
        },
        "new": {
            name: float(np.mean(values))
            for name, values in new_values.items()
        },
        "paired": paired,
        "success_10db": sum(
            value > 10.0 for value in new_values["psnr"]
        ),
    }
    full100 = load("full100_fast.json")
    full100_values = metrics(full100)
    full100_r2 = load("full100_r2.json")
    full100_r2_values = metrics(full100_r2)
    full100_fusion = load("full100_r2_fusion.json")["summary"]["2"]
    old_full = metrics(old)
    full100_paired = {}
    for name in ("psnr", "ssim", "lpips"):
        deltas = [
            candidate - reference
            for candidate, reference in zip(
                full100_values[name], old_full[name], strict=True
            )
        ]
        low, high = bootstrap_ci(deltas)
        higher_better = name in {"psnr", "ssim"}
        full100_paired[name] = {
            "delta": mean(deltas),
            "ci_low": low,
            "ci_high": high,
            "wins": sum(
                value > 0 if higher_better else value < 0
                for value in deltas
            ),
            "losses": sum(
                value < 0 if higher_better else value > 0
                for value in deltas
            ),
        }
    full100_summary = {
        "old": {
            name: float(np.mean(values))
            for name, values in old_full.items()
        },
        "corrected_single": {
            name: float(np.mean(values))
            for name, values in full100_values.items()
        },
        "corrected_two_run_mean": {
            name: float(np.mean(values))
            for name, values in full100_r2_values.items()
        },
        "two_sample_fusion": {
            name: full100_fusion[name]
            for name in ("psnr", "ssim", "lpips")
        },
        "paired_single_vs_old": full100_paired,
    }
    seed_files = (
        (42, "best_100m50.json"),
        (123, "seed123_100m50.json"),
        (2026, "seed2026_100m50.json"),
        (7, "seed7_100m50.json"),
    )
    multiseed = {}
    for seed, filename in seed_files:
        values = metrics(load(filename))
        multiseed[str(seed)] = {
            name: float(np.mean(samples))
            for name, samples in values.items()
        }
    multiseed["mean"] = {
        name: float(np.mean([multiseed[str(seed)][name] for seed, _ in seed_files]))
        for name in ("psnr", "ssim", "lpips")
    }
    multiseed["std"] = {
        name: float(np.std([multiseed[str(seed)][name] for seed, _ in seed_files]))
        for name in ("psnr", "ssim", "lpips")
    }
    multiseed["four_way_fusion"] = load(
        "multiseed_fusion_100m50.json"
    )["equal_fusion"]

    thermal_seed_files = (
        (42, "imgnoise_01_100m50.json"),
        (123, "imgnoise_01_seed123_100m50.json"),
        (2026, "imgnoise_01_seed2026_100m50.json"),
        (7, "imgnoise_01_seed7_100m50.json"),
    )
    thermal_multiseed = {}
    for seed, filename in thermal_seed_files:
        values = metrics(load(filename))
        thermal_multiseed[str(seed)] = {
            name: float(np.mean(samples))
            for name, samples in values.items()
        }
    for group in ("mean", "std"):
        aggregator = np.mean if group == "mean" else np.std
        thermal_multiseed[group] = {
            name: float(aggregator([
                thermal_multiseed[str(seed)][name]
                for seed, _ in thermal_seed_files
            ]))
            for name in ("psnr", "ssim", "lpips")
        }

    payload = {
        "four_image_schedule_ablation": table,
        "four_image_coupling_and_noise_ablation": extended_table,
        "fixed_protocol_20_images_seed42": full20,
        "four_seed_100x50_audit": multiseed,
        "four_seed_image_noise_0.1_audit": thermal_multiseed,
        "full_100_seed42": full100_summary,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "nonlinear_optimization.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Nonlinear optimization audit",
        "",
        "## Four-image mechanism and schedule ablation",
        "",
        "| Variant | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
    ]
    for label, values in table.items():
        lines.append(
            f"| {label} | {values['psnr']:.4f} | "
            f"{values['ssim']:.4f} | {values['lpips']:.4f} |"
        )
    lines.extend([
        "",
        "## Four-image thermal-noise and coupling ablation",
        "",
        "| Variant | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
    ])
    for label, values in extended_table.items():
        lines.append(
            f"| {label} | {values['psnr']:.4f} | "
            f"{values['ssim']:.4f} | {values['lpips']:.4f} |"
        )
    lines.extend([
        "",
        "## Twenty-image, seed-42, 200/100 protocol",
        "",
        "| Implementation | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
        f"| Original LCC-N | {full20['old']['psnr']:.4f} | "
        f"{full20['old']['ssim']:.4f} | {full20['old']['lpips']:.4f} |",
        f"| Corrected LCC-N | {full20['new']['psnr']:.4f} | "
        f"{full20['new']['ssim']:.4f} | {full20['new']['lpips']:.4f} |",
        "",
        f"Paired PSNR change: {paired['psnr']['delta']:+.4f} dB "
        f"({paired['psnr']['wins']}/{paired['psnr']['losses']}); "
        f"SSIM {paired['ssim']['delta']:+.4f}; "
        f"LPIPS {paired['lpips']['delta']:+.4f}.",
        "",
        "## Four-seed, four-image audit at 100×50",
        "",
        "| Seed | PSNR | SSIM | LPIPS |",
        "|---:|---:|---:|---:|",
    ])
    for seed in ("42", "123", "2026", "7"):
        values = multiseed[seed]
        lines.append(
            f"| {seed} | {values['psnr']:.4f} | "
            f"{values['ssim']:.4f} | {values['lpips']:.4f} |"
        )
    lines.append(
        f"| Mean | {multiseed['mean']['psnr']:.4f} ± "
        f"{multiseed['std']['psnr']:.4f} | "
        f"{multiseed['mean']['ssim']:.4f} | "
        f"{multiseed['mean']['lpips']:.4f} |"
    )
    lines.extend([
        "",
        "## Four-seed image thermal-noise check at 100×50",
        "",
        "| Seed | PSNR | SSIM | LPIPS |",
        "|---:|---:|---:|---:|",
    ])
    for seed in ("42", "123", "2026", "7"):
        values = thermal_multiseed[seed]
        lines.append(
            f"| {seed} | {values['psnr']:.4f} | "
            f"{values['ssim']:.4f} | {values['lpips']:.4f} |"
        )
    lines.append(
        f"| Mean | {thermal_multiseed['mean']['psnr']:.4f} ± "
        f"{thermal_multiseed['std']['psnr']:.4f} | "
        f"{thermal_multiseed['mean']['ssim']:.4f} | "
        f"{thermal_multiseed['mean']['lpips']:.4f} |"
    )
    fusion = multiseed["four_way_fusion"]
    lines.extend([
        "",
        f"Equal four-seed fusion: {fusion['psnr']:.4f} dB, "
        f"{fusion['ssim']:.4f} SSIM, {fusion['lpips']:.4f} LPIPS.",
        "",
        "## Full 100-image nonlinear audit",
        "",
        "| Variant | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
        f"| Original LCC-N | {full100_summary['old']['psnr']:.4f} | "
        f"{full100_summary['old']['ssim']:.4f} | "
        f"{full100_summary['old']['lpips']:.4f} |",
        f"| Corrected, one sample | "
        f"{full100_summary['corrected_single']['psnr']:.4f} | "
        f"{full100_summary['corrected_single']['ssim']:.4f} | "
        f"{full100_summary['corrected_single']['lpips']:.4f} |",
        f"| Corrected, two-sample fusion | "
        f"{full100_summary['two_sample_fusion']['psnr']:.4f} | "
        f"{full100_summary['two_sample_fusion']['ssim']:.4f} | "
        f"{full100_summary['two_sample_fusion']['lpips']:.4f} |",
        "",
        f"Single-sample paired PSNR gain: "
        f"{full100_paired['psnr']['delta']:+.4f} dB "
        f"({full100_paired['psnr']['wins']}/100 wins).",
    ])
    (RESULTS / "nonlinear_optimization.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
