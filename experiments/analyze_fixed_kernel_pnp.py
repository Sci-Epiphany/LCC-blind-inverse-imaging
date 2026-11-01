"""Analyze four-run PnP results that reuse the comparison benchmark kernels."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean, pstdev


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BENCHMARKS = {
    "Gaussian": (
        HERE / "remote_paper_pnp_results" / "gaussian_seed42_runs4" / "metrics.json",
        HERE / "source_data" / "gaussian_metrics_archived_l1.json",
    ),
    "Motion": (
        HERE / "remote_paper_pnp_results" / "motion_seed42_runs4" / "metrics.json",
        HERE / "source_data" / "motion_metrics_archived_l1.json",
    ),
}


def bootstrap_ci(values: list[float], draws: int = 10000) -> tuple[float, float]:
    rng = random.Random(20261031)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def main() -> None:
    output = {}
    for benchmark, (metrics_path, baseline_path) in BENCHMARKS.items():
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        images = {
            metric: [mean(row) for row in data["image"][metric]["sample"]]
            for metric in ("psnr", "ssim", "lpips")
        }
        kernel_mse = [
            mean(row) for row in data["kernel"]["kernel_mse"]["sample"]
        ]
        run_psnr = [
            mean(row[run] for row in data["image"]["psnr"]["sample"])
            for run in range(len(data["image"]["psnr"]["sample"][0]))
        ]
        baseline_images = {
            "psnr": [
                mean(row) for row in baseline["image"]["psnr"]["sample"]
            ],
            "lpips": [
                mean(row) for row in baseline["image"]["lpips"]["sample"]
            ],
            "kernel_mse": [
                mean(row) for row in baseline["kernel"]["kernel_mse"]["sample"]
            ],
        }
        own_images = {"psnr": images["psnr"], "lpips": images["lpips"], "kernel_mse": kernel_mse}
        paired = {}
        for metric in ("psnr", "lpips", "kernel_mse"):
            deltas = [
                candidate - reference
                for candidate, reference in zip(
                    own_images[metric], baseline_images[metric], strict=True
                )
            ]
            low, high = bootstrap_ci(deltas)
            higher_is_better = metric == "psnr"
            paired[metric] = {
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
            }
        output[benchmark] = {
            "n_images": len(images["psnr"]),
            "n_runs": len(run_psnr),
            "psnr": mean(value for row in data["image"]["psnr"]["sample"] for value in row),
            "ssim": mean(value for row in data["image"]["ssim"]["sample"] for value in row),
            "lpips": mean(value for row in data["image"]["lpips"]["sample"] for value in row),
            "kernel_mse": mean(
                value for row in data["kernel"]["kernel_mse"]["sample"] for value in row
            ),
            "success_10db": sum(
                value > 10.0
                for row in data["image"]["psnr"]["sample"]
                for value in row
            ),
            "run_psnr_means": run_psnr,
            "run_psnr_std": pstdev(run_psnr),
            "paired_vs_archived": paired,
        }

    if not output:
        raise FileNotFoundError("No fixed-kernel four-run PnP result is available")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fixed_kernel_pnp.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS / "fixed_kernel_pnp.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["benchmark", "images", "runs", "psnr", "ssim", "lpips", "kernel_mse", "success_10db", "run_psnr_means"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark, values in output.items():
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "images": values["n_images"],
                    "runs": values["n_runs"],
                    "psnr": values["psnr"],
                    "ssim": values["ssim"],
                    "lpips": values["lpips"],
                    "kernel_mse": values["kernel_mse"],
                    "success_10db": values["success_10db"],
                    "run_psnr_means": " / ".join(
                        f"{value:.4f}" for value in values["run_psnr_means"]
                    ),
                }
            )

    lines = [
        "# Fixed-kernel full PnP reruns",
        "",
        "These runs use seed 42 and four independent stochastic sampler runs, "
        "so the generated benchmark kernels match the comparison outputs.",
        "",
        "| Task | Images | Runs | PSNR | SSIM | LPIPS | K-MSE | SR@10 dB | Run means |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for benchmark, values in output.items():
        lines.append(
            f"| {benchmark} | {values['n_images']} | {values['n_runs']} | "
            f"{values['psnr']:.4f} | {values['ssim']:.4f} | "
            f"{values['lpips']:.4f} | {values['kernel_mse']:.3e} | "
            f"{values['success_10db']}/{values['n_images'] * values['n_runs']} | "
            + " / ".join(f"{value:.2f}" for value in values["run_psnr_means"])
            + " |"
        )
    lines.extend(
        [
            "",
            "| Task | Metric | Delta | 95% CI | Wins/Losses |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for benchmark, values in output.items():
        for metric, label, format_spec in (
            ("psnr", "PSNR (dB)", ".4f"),
            ("lpips", "LPIPS", ".4f"),
            ("kernel_mse", "K-MSE", ".3e"),
        ):
            paired = values["paired_vs_archived"][metric]
            lines.append(
                f"| {benchmark} | {label} | {paired['delta']:+{format_spec}} | "
                f"[{paired['ci_low']:+{format_spec}}, {paired['ci_high']:+{format_spec}}] | "
                f"{paired['wins']}/{paired['losses']} |"
            )
    (RESULTS / "fixed_kernel_pnp.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
