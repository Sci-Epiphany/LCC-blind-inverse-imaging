"""Evaluate label-free fusion of two saved restoration branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from piq import LPIPS, psnr, ssim


def load(path: Path, device: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def save(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Weight of the left branch in left*w + right*(1-w).",
    )
    args = parser.parse_args()

    paths = sorted(args.right.glob("*.png"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = LPIPS(replace_pooling=True, reduction="none").eval().to(device)
    results = {}
    for weight in args.weights:
        estimates = []
        references = []
        names = []
        for right_path in paths:
            left_path = args.left / right_path.name
            if not left_path.is_file():
                raise FileNotFoundError(left_path)
            image_id = int(right_path.stem.split("_")[0])
            left = load(left_path, device)
            right = load(right_path, device)
            reference = load(
                args.ground_truth / f"{image_id:05d}.png", device
            )
            estimate = weight * left + (1.0 - weight) * right
            estimates.append(estimate)
            references.append(reference)
            names.append(right_path.name)
            save(
                estimate,
                args.output / f"left_{weight:.2f}" / right_path.name,
            )

        values = {"psnr": [], "ssim": [], "lpips": []}
        with torch.no_grad():
            for start in range(0, len(estimates), 8):
                estimate_batch = torch.cat(estimates[start:start + 8])
                reference_batch = torch.cat(references[start:start + 8])
                values["psnr"].append(
                    psnr(
                        reference_batch, estimate_batch,
                        data_range=1.0, reduction="none",
                    ).flatten()
                )
                values["ssim"].append(
                    ssim(
                        reference_batch, estimate_batch,
                        data_range=1.0, reduction="none",
                    ).flatten()
                )
                values["lpips"].append(
                    lpips_model(reference_batch, estimate_batch).flatten()
                )
        values = {
            name: torch.cat(chunks) for name, chunks in values.items()
        }
        results[f"{weight:.2f}"] = {
            "left_weight": weight,
            **{
                name: float(metric.mean())
                for name, metric in values.items()
            },
            "samples": {
                name: {
                    metric: float(values[metric][index])
                    for metric in values
                }
                for index, name in enumerate(names)
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"images": len(paths), "fusion": results}
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        weight: {
            name: result[name]
            for name in ("psnr", "ssim", "lpips")
        }
        for weight, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
