"""Evaluate label-free fusion of repeated stochastic restoration runs."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from piq import LPIPS, psnr, ssim


RUN_RE = re.compile(r"^(\d{5})_run(\d{4})\.png$")


def load(path: Path, device: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--images", type=int, default=None)
    args = parser.parse_args()

    runs: dict[int, dict[int, Path]] = {}
    for path in args.samples.glob("*.png"):
        match = RUN_RE.match(path.name)
        if match:
            runs.setdefault(int(match.group(1)), {})[int(match.group(2))] = path
    image_ids = sorted(runs)
    if args.images is not None:
        image_ids = image_ids[:args.images]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = LPIPS(replace_pooling=True, reduction="none").eval().to(device)
    summary = {}
    for count in sorted({1, 2, 4} & {len(runs[index]) for index in image_ids}):
        estimates = []
        references = []
        names = []
        for image_id in image_ids:
            available = [runs[image_id][run] for run in sorted(runs[image_id])]
            reference_path = next(
                path for path in sorted(
                    args.ground_truth.glob(f"{image_id:05d}.*")
                )
                if path.is_file()
            )
            reference = load(reference_path, device)
            for subset in itertools.combinations(available, count):
                estimates.append(
                    torch.stack([load(path, device) for path in subset]).mean(0)
                )
                references.append(reference)
                names.append((image_id, tuple(path.name for path in subset)))

        p_values = []
        s_values = []
        l_values = []
        with torch.no_grad():
            for start in range(0, len(estimates), 8):
                estimate_batch = torch.cat(estimates[start:start + 8])
                reference_batch = torch.cat(references[start:start + 8])
                p_values.append(psnr(
                    reference_batch, estimate_batch,
                    data_range=1.0, reduction="none",
                ).flatten())
                s_values.append(ssim(
                    reference_batch, estimate_batch,
                    data_range=1.0, reduction="none",
                ).flatten())
                l_values.append(
                    lpips_model(reference_batch, estimate_batch).flatten()
                )
        p_values = torch.cat(p_values)
        s_values = torch.cat(s_values)
        l_values = torch.cat(l_values)
        summary[str(count)] = {
            "samples": count,
            "cases": len(estimates),
            "psnr": float(p_values.mean()),
            "ssim": float(s_values.mean()),
            "lpips": float(l_values.mean()),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"images": len(image_ids), "summary": summary}
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
