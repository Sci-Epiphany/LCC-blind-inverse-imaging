"""Paired comparisons for the final fixed-input corrected-PnP reruns."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyze_existing_results import collect
from paired_comparisons import METHOD_SEED_OFFSET, bootstrap_ci


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BLINDDAPS = HERE.parents[2] / "BlindDAPS"
FIXED = HERE / "remote_paper_pnp_results"
OLD_NONLINEAR = (
    HERE
    / "external_blinddaps"
    / "DAPS"
    / "results"
    / "nonlinear_blur_demo_ffhq"
    / "metrics.json"
)


def mean_runs(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        metric: np.asarray(data["image"][metric]["sample"], dtype=float).mean(axis=1)
        for metric in ("psnr", "ssim", "lpips")
    }


def main() -> None:
    _, raw = collect(BLINDDAPS.resolve())
    metrics_by_task = {
        "Gaussian": mean_runs(
            FIXED / "gaussian_seed42_runs4" / "metrics.json"
        ),
        "Nonlinear": mean_runs(OLD_NONLINEAR),
    }
    motion_path = FIXED / "motion_seed42_runs4" / "metrics.json"
    if motion_path.is_file():
        metrics_by_task["Motion"] = mean_runs(motion_path)

    rows = []
    for benchmark, own in metrics_by_task.items():
        for method, records in sorted(raw[benchmark].items()):
            paired = []
            for image_id, values in records.items():
                index = int(Path(image_id).stem)
                if index >= len(own["psnr"]):
                    continue
                paired.append(
                    (
                        index,
                        float(values["psnr"]),
                        float(values["ssim"]),
                        float(values["lpips"]),
                    )
                )
            if not paired:
                continue
            indices = np.asarray([row[0] for row in paired], dtype=int)
            base_psnr = np.asarray([row[1] for row in paired], dtype=float)
            base_ssim = np.asarray([row[2] for row in paired], dtype=float)
            base_lpips = np.asarray([row[3] for row in paired], dtype=float)
            deltas = own["psnr"][indices] - base_psnr
            p_value = (
                1.0
                if np.allclose(deltas, 0.0)
                else float(
                    wilcoxon(
                        deltas, alternative="two-sided", zero_method="wilcox"
                    ).pvalue
                )
            )
            low, high = bootstrap_ci(
                deltas, seed=8100 + METHOD_SEED_OFFSET.get(method, 0)
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "baseline": method,
                    "n_pairs": len(indices),
                    "lcc_psnr": float(own["psnr"][indices].mean()),
                    "baseline_psnr": float(base_psnr.mean()),
                    "delta_psnr": float(deltas.mean()),
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "wins": int((deltas > 0).sum()),
                    "losses": int((deltas < 0).sum()),
                    "wilcoxon_p": p_value,
                    "delta_ssim": float(
                        np.mean(own["ssim"][indices] - base_ssim)
                    ),
                    "delta_lpips": float(
                        np.mean(own["lpips"][indices] - base_lpips)
                    ),
                }
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "paired_comparisons_fixed_pnp.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "paired_comparisons_fixed_pnp.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Paired image-level comparisons after the fixed-input PnP rerun",
        "",
        "LCC Gaussian values average four stochastic runs on the archived "
        "measurement set. Nonlinear values retain the four archived runs.",
        "",
        "| Benchmark | Baseline | N | Delta PSNR (95% CI) | W/L | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['baseline']} | {row['n_pairs']} | "
            f"{row['delta_psnr']:+.4f} "
            f"[{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] | "
            f"{row['wins']}/{row['losses']} | {row['wilcoxon_p']:.3g} |"
        )
    (RESULTS / "paired_comparisons_fixed_pnp.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
