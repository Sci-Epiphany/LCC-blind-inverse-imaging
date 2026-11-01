"""Summarize label-free pre-BKSE latent proposal experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
ROOT = HERE / "remote_proposal_results"
RESULTS = HERE / "results"
NAMES = {
    "proposal_c1_scale1": "one stochastic proposal",
    "proposal_c4_scale0": "four proposals, deterministic mean",
    "proposal_c4_scale05": "four proposals, 0.5x noise",
    "proposal_c4_scale1": "four proposals, 1.0x noise",
}


def main() -> None:
    rows = []
    for directory, label in NAMES.items():
        data = json.loads((ROOT / directory / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": label,
                "n_images": len(data["image"]["psnr"]["sample"]),
                "psnr": mean(row[0] for row in data["image"]["psnr"]["sample"]),
                "ssim": mean(row[0] for row in data["image"]["ssim"]["sample"]),
                "lpips": mean(row[0] for row in data["image"]["lpips"]["sample"]),
                "kmse": mean(row[0] for row in data["kernel"]["kernel_mse"]["sample"]),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "proposal_tuning.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "proposal_tuning.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Pre-BKSE latent proposal pilot",
        "",
        "Four nonlinear test images, seed 42, one run per image/config.",
        "Selection minimizes observation error only; ground truth is not used.",
        "",
        "| Variant | N | PSNR | SSIM | LPIPS | K-MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['n_images']} | {row['psnr']:.4f} | "
            f"{row['ssim']:.4f} | {row['lpips']:.4f} | {row['kmse']:.3e} |"
        )
    (RESULTS / "proposal_tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
