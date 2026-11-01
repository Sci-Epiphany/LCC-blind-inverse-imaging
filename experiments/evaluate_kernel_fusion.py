"""Evaluate simplex fusion of two saved kernel branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path) -> np.ndarray:
    kernel = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return kernel / max(kernel.sum(), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--left-suffix", default="")
    args = parser.parse_args()

    right_paths = sorted(args.right.glob("*.png"))
    values = {}
    for weight in args.weights:
        errors = []
        per_image = {}
        for right_path in right_paths:
            image_id = int(right_path.stem.split("_")[0])
            left_path = args.left / (
                f"{image_id:05d}{args.left_suffix}.png"
                if args.left_suffix
                else right_path.name
            )
            left = load(left_path)
            right = load(right_path)
            fused = weight * left + (1.0 - weight) * right
            fused /= max(fused.sum(), 1e-12)
            if abs(weight - 0.5) < 1e-12:
                kernel_path = args.output / "kernels" / right_path.name
                kernel_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    (np.clip(fused, 0.0, 1.0) * 255.0).round().astype(np.uint8),
                    mode="L",
                ).save(kernel_path)
            gt = load(args.ground_truth / f"{image_id:05d}.png")
            error = float(np.mean((fused - gt) ** 2))
            errors.append(error)
            per_image[right_path.name] = error
        values[f"{weight:.2f}"] = {
            "left_weight": weight,
            "kernel_mse": float(np.mean(errors)),
            "samples": per_image,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"images": len(right_paths), "fusion": values}
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        weight: result["kernel_mse"]
        for weight, result in values.items()
    }, indent=2))


if __name__ == "__main__":
    main()
