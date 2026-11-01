"""Evaluate no-reference metrics for saved restoration candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyiqa
import torch
from PIL import Image


def load(path: Path, device: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {
        "niqe": pyiqa.create_metric("niqe", device=device),
        "maniqa": pyiqa.create_metric("maniqa", device=device),
    }
    results = {}
    for directory in sorted(path for path in args.root.iterdir() if path.is_dir()):
        paths = sorted(directory.glob("*.png"))
        if not paths:
            continue
        values = {name: [] for name in metrics}
        for path in paths:
            image = load(path, device)
            for name, metric in metrics.items():
                values[name].append(float(metric(image)))
        results[directory.name] = {
            name: float(np.mean(metric_values))
            for name, metric_values in values.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "directions": {"niqe": "lower", "maniqa": "higher"},
        "candidates": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
