"""Refine saved kernels with a fixed restored image and no ground truth."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from forward_operator.blind_blur_utils import HQS_ker


SAMPLE_RE = re.compile(r"^(\d{5})_run(\d{4})\.png$")


def load_rgb(path: Path, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def load_kernel(path: Path, device: str) -> torch.Tensor:
    kernel = torch.from_numpy(
        np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    )
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
    return kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def save_kernel(kernel: torch.Tensor, path: Path) -> None:
    array = kernel[0, 0].detach().cpu().numpy()
    array = np.clip(array / max(float(array.sum()), 1e-12), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=1e5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = {}
    image_paths = sorted(args.images.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    for image_path in image_paths:
        match = SAMPLE_RE.match(image_path.name)
        if not match:
            continue
        image_id = int(match.group(1))
        name = image_path.name
        measurement = load_rgb(args.measurements / f"{image_id:05d}.png", device)
        restored = load_rgb(image_path, device)
        kernel = load_kernel(args.kernels / name, device)
        refined = HQS_ker(
            measurement,
            restored,
            args.sigma,
            kernel,
            iters=args.iterations,
            ksize=kernel.shape[-1],
        )
        refined /= refined.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        gt = load_kernel(args.ground_truth / f"{image_id:05d}.png", device)
        output_path = args.output / "kernels" / name
        save_kernel(refined, output_path)
        saved_refined = load_kernel(output_path, device)
        outputs[name] = {
            "initial_mse": float(((kernel - gt) ** 2).mean()),
            "refined_mse_in_memory": float(((refined - gt) ** 2).mean()),
            "refined_mse": float(((saved_refined - gt) ** 2).mean()),
        }

    before = float(np.mean([value["initial_mse"] for value in outputs.values()]))
    after = float(np.mean([value["refined_mse"] for value in outputs.values()]))
    payload = {
        "images": len(outputs),
        "sigma": args.sigma,
        "beta": args.beta,
        "iterations": args.iterations,
        "initial_mse": before,
        "refined_mse": after,
        "samples": outputs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
