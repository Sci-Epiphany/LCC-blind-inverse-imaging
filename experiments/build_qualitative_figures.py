"""Rebuild the qualitative panels with corrected method labels.

The image data are the same retained benchmark outputs used by the paper.  The
only semantic change to the existing figure composition is ``Ours`` -> ``LCC``
and the consistent use of the actual DeblurGAN/FFTFormer labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
BLIND = HERE.parents[3] / "BlindDAPS"
COMP = BLIND / "Experiment" / "Comparisons"
DAPS = HERE.parent / "external_blinddaps" / "DAPS" / "results"
FIXED_FIG = HERE.parent / "fixed_pnp_figures"
OPTIMIZED_FIG = HERE.parent / "optimized_figures"
OUT = ROOT / "Figures"
EXTS = (".png", ".jpg", ".jpeg", ".bmp")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    }
)


def image_file(directory: Path, stem: str) -> Path:
    for extension in EXTS:
        path = directory / f"{stem}{extension}"
        if path.exists() and path.stat().st_size:
            return path
        run_zero = directory / f"{stem}_run0000{extension}"
        if run_zero.exists() and run_zero.stat().st_size:
            return run_zero
    raise FileNotFoundError(directory / stem)


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_kernel(path: Path, target: int = 64) -> tuple[np.ndarray, bool]:
    image = Image.open(path)
    pseudo = image.mode in {"RGB", "RGBA"}
    kernel = np.asarray(image.convert("L"), dtype=np.float32)
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
    h, w = kernel.shape
    ph, pw = target - h, target - w
    kernel = np.pad(
        kernel,
        ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
        mode="edge",
    )
    return kernel, pseudo


def draw_grid(
    rows: list[dict[str, tuple[Path, Path | None]]],
    methods: list[str],
    output: Path,
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(len(rows), len(methods), figsize=figsize, dpi=300)
    fig.subplots_adjust(wspace=0.035, hspace=0.015)
    for row_index, row in enumerate(rows):
        for col_index, method in enumerate(methods):
            axis = axes[row_index, col_index]
            image_path, kernel_path = row[method]
            axis.imshow(load_image(image_path))
            axis.axis("off")
            if row_index == 0:
                axis.set_title(method, fontsize=11.5, fontweight="bold", pad=4)
            if kernel_path is not None and kernel_path.stat().st_size:
                try:
                    kernel, _ = load_kernel(kernel_path)
                    inset = axis.inset_axes((0.75, 0.0, 0.25, 0.25))
                    inset.imshow(kernel, vmin=0, vmax=1)
                    inset.axis("off")
                except (OSError, ValueError):
                    pass
    fig.savefig(output, bbox_inches="tight", pad_inches=0.05, dpi=300, format="pdf")
    plt.close(fig)


def linear_rows() -> tuple[list[str], list[dict[str, tuple[Path, Path | None]]]]:
    g = COMP / "Gaussian_deblur"
    m = COMP / "Motion_deblur"
    gaussian_lcc_samples = OPTIMIZED_FIG / "gaussian" / "samples"
    gaussian_lcc = (
        (gaussian_lcc_samples, OPTIMIZED_FIG / "gaussian" / "kernels")
        if gaussian_lcc_samples.exists()
        else (
            DAPS / "gaussian_blur_demo" / "samples",
            DAPS / "gaussian_blur_demo" / "kernel_samples",
        )
    )
    motion_lcc_samples = OPTIMIZED_FIG / "motion" / "samples"
    motion_lcc = (
        (motion_lcc_samples, OPTIMIZED_FIG / "motion" / "kernels")
        if motion_lcc_samples.exists()
        else (
            DAPS / "motion_blur_demo" / "samples",
            DAPS / "motion_blur_demo" / "kernel_samples",
        )
    )
    specs = {
        "LR": (DAPS / "gaussian_blur_demo" / "measurements", None),
        "SelfDeblur": (g / "SelfDeblur" / "results" / "daps_samples" / "imgs", g / "SelfDeblur" / "results" / "daps_samples" / "kernels"),
        "BKSE": (g / "BKSE" / "samples", None),
        "FFTFormer": (g / "FFTFormer" / "daps_samples", None),
        "BlindDPS": (g / "blind_dps" / "sample", g / "blind_dps" / "kernel"),
        "DMPlug": (g / "DMPlug" / "results" / "sample", g / "DMPlug" / "results" / "kernel"),
        "LCC": gaussian_lcc,
        "HR": (g / "ground_truth" / "sample_gts", g / "ground_truth" / "kernel_gts"),
    }
    motion_specs = {
        "LR": (DAPS / "motion_blur_demo" / "measurements", None),
        "SelfDeblur": (m / "SelfDeblur" / "daps_motion" / "image", m / "SelfDeblur" / "daps_motion" / "kernel"),
        "BKSE": (m / "BKSE" / "motion_results", None),
        "FFTFormer": (m / "FFTFormer" / "motion" / "daps", None),
        "BlindDPS": (m / "BlindDPS" / "recon" / "images", m / "BlindDPS" / "recon" / "kernels"),
        "DMPlug": (m / "DMPlug" / "image", m / "DMPlug" / "kernel"),
        "LCC": motion_lcc,
        "HR": (m / "ground_truth" / "sample_gts", m / "ground_truth" / "kernel_gts"),
    }
    methods = list(specs)
    rows = []
    for row_index, stem in enumerate(("00065", "00011", "00041")):
        current = specs if row_index == 0 else motion_specs
        row = {}
        for method, (image_dir, kernel_dir) in current.items():
            row[method] = (
                image_file(image_dir, stem),
                image_file(kernel_dir, stem) if kernel_dir else None,
            )
        rows.append(row)
    return methods, rows


def nonlinear_rows() -> tuple[list[str], list[dict[str, tuple[Path, Path | None]]]]:
    n = COMP / "Nonlinear_deblur"
    real = COMP / "RealDeblurring"
    f_methods = {
        "LR": (DAPS / "nonlinear_blur_demo_ffhq" / "measurements", None),
        "SelfDeblur": (n / "SelfDeblur" / "daps_nonlinear" / "image", None),
        "BKSE": (n / "BKSE" / "nonlinear_results", None),
        "DeepRFT": (n / "DeepRFT" / "image", None),
        "BlindDPS": (n / "BlindDPS" / "recon" / "images", None),
        "DMPlug": (n / "DMPlug" / "image", None),
        "LCC": (DAPS / "nonlinear_blur_demo_ffhq" / "samples", None),
    }
    c_methods = {
        "LR": ("LR", None),
        "SelfDeblur": ("SelfDeblur", None),
        "BKSE": ("bkse", None),
        "DeepRFT": ("deeprft", None),
        "BlindDPS": ("blind-dps", None),
        "DMPlug": ("DMPlug", None),
        "LCC": ("daps", None),
    }
    methods = list(f_methods)
    rows = []
    for stem in ("00080", "00051"):
        row = {}
        for method, (image_dir, _) in f_methods.items():
            row[method] = (image_file(image_dir, stem), None)
        rows.append(row)
    all_images = {
        method: sorted(
            [p for p in (real / subdir).iterdir() if p.suffix.lower() in EXTS],
            key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.stem)],
        )
        for method, (subdir, _) in c_methods.items()
    }
    for index in (6, 8):
        rows.append({method: (all_images[method][index], None) for method in methods})
    return methods, rows


def main() -> None:
    linear_methods, linear_data = linear_rows()
    draw_grid(
        linear_data,
        linear_methods,
        OUT / "deblur_comparison_lcc.pdf",
        (20, 7.5),
    )
    nonlinear_methods, nonlinear_data = nonlinear_rows()
    draw_grid(
        nonlinear_data,
        nonlinear_methods,
        OUT / "nonlinear_deblur_comparison_lcc.pdf",
        (16, 9),
    )
    print("Wrote corrected qualitative figures")


if __name__ == "__main__":
    main()
