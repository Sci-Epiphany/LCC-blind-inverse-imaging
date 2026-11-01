"""Evaluate equal fusion across independently seeded restoration directories."""

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


def score(
    estimates: list[torch.Tensor],
    references: list[torch.Tensor],
    device: str,
) -> dict[str, float]:
    lpips_model = LPIPS(replace_pooling=True, reduction="none").eval().to(device)
    estimate_batch = torch.cat(estimates)
    reference_batch = torch.cat(references)
    with torch.no_grad():
        return {
            "psnr": float(psnr(
                reference_batch, estimate_batch,
                data_range=1.0, reduction="mean",
            )),
            "ssim": float(ssim(
                reference_batch, estimate_batch,
                data_range=1.0, reduction="mean",
            )),
            "lpips": float(lpips_model(
                reference_batch, estimate_batch
            ).mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, nargs="+", type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(args.samples[0].glob("*.png"))
    estimates = {path.name: [] for path in paths}
    references = []
    for path in paths:
        image_id = int(path.stem.split("_")[0])
        reference_path = next(
            candidate for candidate in sorted(
                args.ground_truth.glob(f"{image_id:05d}.*")
            )
            if candidate.is_file()
        )
        references.append(load(reference_path, device))

    individual = {}
    for directory in args.samples:
        current = []
        for path in paths:
            value = load(directory / path.name, device)
            estimates[path.name].append(value)
            current.append(value)
        label = directory.parent.name if directory.name == "samples" else directory.name
        individual[label] = score(current, references, device)

    fused = [
        torch.stack(estimates[path.name]).mean(0)
        for path in paths
    ]
    payload = {
        "images": len(paths),
        "individual": individual,
        "equal_fusion": score(fused, references, device),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for path, image in zip(paths, fused):
        array = (
            image[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        ).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(args.output / path.name)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
