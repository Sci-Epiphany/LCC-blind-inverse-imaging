"""Inspect KernelWizard mean and variance on retained restoration outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from forward_operator import BlindNonlinearBlur


def load(path: Path, device: str) -> torch.Tensor:
    value = np.asarray(
        Image.open(path).convert("RGB"), dtype=np.float32
    ) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    operator = BlindNonlinearBlur(
        "forward_operator/bkse/options/generate_blur/default.yml",
        device=args.device,
        proposal_mode="mean",
    )
    values = []
    paths = sorted(args.samples.glob("*_run0000.png"))[:args.limit]
    with torch.no_grad():
        for path in paths:
            image_id = path.name.split("_")[0]
            measurement_path = next(
                candidate
                for candidate in sorted(
                    args.measurements.glob(f"{image_id}.*")
                )
                if candidate.is_file()
            )
            image = 2.0 * load(path, args.device) - 1.0
            measurement = 2.0 * load(measurement_path, args.device) - 1.0
            unit_image = ((image + 1.0) / 2.0).clamp(0, 1)
            unit_measurement = ((measurement + 1.0) / 2.0).clamp(0, 1)
            mean, sigma = operator.blur_model(unit_image, unit_measurement)
            values.append((mean.flatten(), sigma.flatten()))

    means = torch.cat([value[0] for value in values])
    sigmas = torch.cat([value[1] for value in values])
    for name, value in (("mean", means), ("sigma", sigmas)):
        print(
            f"{name}: min={value.min().item():.6g} "
            f"mean={value.mean().item():.6g} "
            f"max={value.max().item():.6g}"
        )


if __name__ == "__main__":
    main()
