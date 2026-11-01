"""Evaluate label-free fusion of four corrected-PnP runs on fixed inputs."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from piq import LPIPS, psnr, ssim


RUN_RE = re.compile(r"^(\d{5})(?:_run(\d{4}))?\.png$")


def load(path: Path, device: str) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)[None]
        .float()
        .div(255.0)
        .to(device)
    )


def select_runs(directory: Path, image_id: int) -> list[Path]:
    paths = sorted(
        path
        for path in directory.glob(f"{image_id:05d}*.png")
        if RUN_RE.match(path.name) and path.stat().st_size
    )
    if len(paths) != 4:
        raise ValueError(f"Expected four runs for {image_id:05d}, found {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--images", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_dir = Path(args.samples)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gt_dir = Path("/home/gsn/PhD_thesis/DAPS/dataset/test-ffhq")
    lpips_model = LPIPS(replace_pooling=True, reduction="none").eval().to(device)

    image_values: dict[int, dict[int, tuple[float, float, float]]] = {}
    for start in range(0, args.images, 8):
        batch = []
        references = []
        metadata = []
        for image_id in range(start, min(start + 8, args.images)):
            samples = [
                load(path, device) for path in select_runs(sample_dir, image_id)
            ]
            reference = load(gt_dir / f"{image_id:05d}.png", device)
            sizes = []
            for size in (1, 2, 4):
                for subset in itertools.combinations(samples, size):
                    batch.append(torch.stack(subset).mean(0))
                    sizes.append(size)
            references.append((reference, len(sizes)))
            metadata.append((image_id, sizes))
        estimates = torch.cat(batch, dim=0)
        reference_batch = torch.cat(
            [
                reference.expand(count, *reference.shape[1:])
                for reference, count in references
            ],
            dim=0,
        )
        with torch.no_grad():
            p_values = psnr(
                reference_batch, estimates, data_range=1.0, reduction="none"
            )
            s_values = ssim(
                reference_batch, estimates, data_range=1.0, reduction="none"
            )
            l_values = lpips_model(reference_batch, estimates).reshape(-1)
        offset = 0
        for image_id, sizes in metadata:
            values = {1: [], 2: [], 4: []}
            for size in sizes:
                values[size].append(
                    (
                        float(p_values[offset]),
                        float(s_values[offset]),
                        float(l_values[offset]),
                    )
                )
                offset += 1
            image_values[image_id] = {
                size: tuple(
                    float(np.mean([row[metric] for row in rows]))
                    for metric in range(3)
                )
                for size, rows in values.items()
            }

    single_psnr = np.asarray(
        [image_values[index][1][0] for index in range(args.images)]
    )
    summary = []
    for size in (1, 2, 4):
        p_values = np.asarray(
            [image_values[index][size][0] for index in range(args.images)]
        )
        s_values = np.asarray(
            [image_values[index][size][1] for index in range(args.images)]
        )
        l_values = np.asarray(
            [image_values[index][size][2] for index in range(args.images)]
        )
        summary.append(
            {
                "samples": size,
                "images": args.images,
                "psnr": float(p_values.mean()),
                "ssim": float(s_values.mean()),
                "lpips": float(l_values.mean()),
                "delta_psnr": float((p_values - single_psnr).mean()),
                "wins": int((p_values > single_psnr).sum()),
            }
        )
    payload = {
        "label": args.label,
        "fusion": "equal-weight mean over all label-free subsets",
        "summary": summary,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
