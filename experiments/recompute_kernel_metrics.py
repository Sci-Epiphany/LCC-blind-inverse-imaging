"""Recompute per-kernel MSE from saved kernel PNGs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


RUN_RE = re.compile(r"^(\d{5})_run(\d{4})\.png$")


def load_kernel(path: Path) -> np.ndarray:
    kernel = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return kernel / max(kernel.sum(), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    predictions: dict[int, list[Path]] = {}
    for path in args.samples.glob("*.png"):
        match = RUN_RE.match(path.name)
        if match:
            predictions.setdefault(int(match.group(1)), []).append(path)

    if not predictions:
        raise FileNotFoundError(f"No kernel PNGs found in {args.samples}")

    per_image = {}
    for image_id in sorted(predictions):
        gt_path = args.ground_truth / f"{image_id:05d}.png"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        gt = load_kernel(gt_path)
        values = []
        for path in sorted(predictions[image_id]):
            prediction = load_kernel(path)
            if prediction.shape != gt.shape:
                raise ValueError(
                    f"Kernel shape mismatch for {path}: "
                    f"{prediction.shape} versus {gt.shape}"
                )
            values.append(float(np.mean((prediction - gt) ** 2)))
        per_image[image_id] = {
            "runs": len(values),
            "mean": float(np.mean(values)),
            "values": values,
        }

    all_values = [
        value
        for result in per_image.values()
        for value in result["values"]
    ]
    payload = {
        "normalization": "per-kernel sum to one",
        "images": len(per_image),
        "runs": len(all_values) // len(per_image),
        "kernel_mse": float(np.mean(all_values)),
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"per-kernel normalized MSE: {payload['kernel_mse']:.8e} "
        f"over {payload['images']} images and {payload['runs']} runs"
    )


if __name__ == "__main__":
    main()
