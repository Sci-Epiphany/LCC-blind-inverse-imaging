"""Evaluate a plug-in image update using refined kernels and saved inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.fft as fft
from PIL import Image
from piq import LPIPS, psnr, ssim

from forward_operator.blind_blur_utils import p2o


SAMPLE_RE = re.compile(r"^(\d{5})_run(\d{4})\.png$")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def load_kernel(path: Path) -> torch.Tensor:
    kernel = torch.from_numpy(load_rgb(path)[:, :, 0]).unsqueeze(0).unsqueeze(0)
    return kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def reconcile(
    restored: torch.Tensor,
    measurement: torch.Tensor,
    kernel: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    optical_transfer = fft.fft2(p2o(kernel, measurement.shape[-2:]))
    measurement_fft = fft.fft2(measurement)
    prior_fft = fft.fft2(restored)
    estimate = (
        optical_transfer.conj() * measurement_fft
        + regularization * prior_fft
    ) / (optical_transfer.abs().square() + regularization)
    return fft.ifft2(estimate).real.clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--regularization", type=float, nargs="+",
        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = LPIPS(replace_pooling=True, reduction="none").eval().to(device)
    samples = []
    image_paths = sorted(args.images.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    for image_path in image_paths:
        match = SAMPLE_RE.match(image_path.name)
        if not match:
            continue
        image_id = int(match.group(1))
        samples.append(
            (
                image_path,
                image_id,
                load_kernel(args.kernels / image_path.name).to(device),
            )
        )

    summary = {}
    for regularization in args.regularization:
        estimates = []
        references = []
        residuals = []
        gcv_values = []
        for image_path, image_id, kernel in samples:
            restored = to_tensor(load_rgb(image_path)).to(device)
            measurement = to_tensor(
                load_rgb(args.measurements / f"{image_id:05d}.png")
            ).to(device)
            reference = to_tensor(
                load_rgb(args.ground_truth / f"{image_id:05d}.png")
            ).to(device)
            estimate = reconcile(restored, measurement, kernel, regularization)
            estimates.append(estimate)
            references.append(reference)
            measured_estimate = fft.ifft2(
                fft.fft2(p2o(kernel, measurement.shape[-2:]))
                * fft.fft2(estimate)
            ).real
            residual = (measured_estimate - measurement) ** 2
            residuals.append(float(residual.mean()))
            optical_transfer = fft.fft2(
                p2o(kernel, measurement.shape[-2:])
            )
            gain = optical_transfer.abs().square() / (
                optical_transfer.abs().square() + regularization
            )
            degrees_of_freedom = float(gain.sum())
            gcv_values.append(
                float(residual.sum())
                / (
                    max(
                        residual.numel() - degrees_of_freedom,
                        1.0,
                    )
                    ** 2
                )
            )
            save_rgb(
                estimate,
                args.output / f"lambda_{regularization:.0e}" / image_path.name,
            )
        p_values = []
        s_values = []
        l_values = []
        with torch.no_grad():
            for start in range(0, len(estimates), 8):
                estimate_batch = torch.cat(estimates[start:start + 8])
                reference_batch = torch.cat(references[start:start + 8])
                p_values.append(psnr(
                    reference_batch, estimate_batch,
                    data_range=1.0, reduction="none"
                ).flatten())
                s_values.append(ssim(
                    reference_batch, estimate_batch,
                    data_range=1.0, reduction="none"
                ).flatten())
                l_values.append(
                    lpips_model(reference_batch, estimate_batch).flatten()
                )
        p_values = torch.cat(p_values)
        s_values = torch.cat(s_values)
        l_values = torch.cat(l_values)
        summary[f"{regularization:.8g}"] = {
            "psnr": float(p_values.mean()),
            "ssim": float(s_values.mean()),
            "lpips": float(l_values.mean()),
            "residual_mse": float(np.mean(residuals)),
            "gcv": float(np.mean(gcv_values)),
            "samples": {
                sample[0].name: {
                    "psnr": float(p_values[index]),
                    "ssim": float(s_values[index]),
                    "lpips": float(l_values[index]),
                }
                for index, sample in enumerate(samples)
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"images": len(samples), "regularization": summary}
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
