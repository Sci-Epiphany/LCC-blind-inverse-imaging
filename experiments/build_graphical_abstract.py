"""Build a corrected, project-local graphical abstract without generative edits.

The source image uses a three-panel magazine layout.  The central heading and
the entire right-side comparison panel are reconstructed deterministically so
that exact names and numbers can be verified against retained result files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Figures" / "Graphical_Abstract.png"
OUTPUT = ROOT / "Figures" / "Graphical_Abstract_final.png"
OUTPUT_PDF = ROOT / "Figures" / "Graphical_Abstract_final.pdf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "timesbd.ttf" if bold else "times.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def build_metric_panel(width: int, height: int) -> Image.Image:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )
    fig = plt.figure(figsize=(width / 200, height / 200), dpi=200, facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.05, 1.0), hspace=0.38)
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])

    methods = [
        "DeepRFT",
        "DeblurGAN",
        "LCC",
        "BlindDPS",
        "DMPlug",
        "FFTFormer",
        "BKSE",
        "SelfDeblur",
        "FastDiffEM",
    ]
    psnr = [25.41, 25.33, 27.58, 24.43, 23.89, 23.30, 18.70, 16.61, 12.34]
    colors = [
        "#56789a",
        "#6f7f90",
        "#a64b44",
        "#8e6aa8",
        "#cf5138",
        "#ed8b2f",
        "#3f9c55",
        "#327fa6",
        "#d36c42",
    ]
    bars = ax1.bar(methods, psnr, color=colors, edgecolor="#333333", linewidth=0.8)
    for method, bar, value in zip(methods, bars, psnr):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold" if method == "LCC" else "normal",
        )
    ax1.set_ylim(0.0, 30.0)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Gaussian blind deblurring: full-sample means")
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)

    kernel_methods = ["LCC", "BlindDPS", "FastDiffEM", "SelfDeblur", "DMPlug"]
    kernel_mse = [1.528e-7, 1.06e-5, 3.23e-5, 8.07e-5, 1.60e-4]
    kcolors = ["#a64b44", "#8e6aa8", "#d56b2d", "#337fa6", "#cf5138"]
    bars = ax2.bar(
        kernel_methods,
        kernel_mse,
        color=kcolors,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Kernel MSE (lower is better)")
    ax2.set_title("Kernel estimation on the same full sample")
    for method, bar, value in zip(kernel_methods, bars, kernel_mse):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:.1e}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold" if method == "LCC" else "normal",
        )
    ax2.grid(axis="y", which="both", alpha=0.22)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.10)

    temp = ROOT / "experiments" / "results" / "_ga_metric_panel.png"
    temp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(temp, dpi=200, facecolor="white")
    plt.close(fig)
    return Image.open(temp).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    canvas = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    grey = (210, 210, 210)
    border = (91, 91, 91)

    # Replace the stale middle-panel heading as one coherent rectangular label.
    draw.rectangle((1662, 15, 3325, 154), fill=grey)
    title = "LCC Overview"
    box = draw.textbbox((0, 0), title, font=font(66, bold=True))
    draw.text(
        (2493.5, 82),
        title,
        fill=(0, 0, 0),
        font=font(66, bold=True),
        anchor="mm",
    )

    # Rebuild the comparison panel.  The panel is independently bounded so the
    # new chart shares the original three-column layout without seam artifacts.
    right_left, right_top, right_right, right_bottom = 3342, 20, 6328, 2656
    draw.rectangle(
        (right_left, right_top, right_right, right_bottom),
        fill=(255, 255, 255),
        outline=border,
        width=8,
    )
    draw.rectangle((right_left + 8, right_top + 8, right_right - 8, 142), fill=grey)
    header = "Full-Sample Evaluation"
    box = draw.textbbox((0, 0), header, font=font(62, bold=True))
    draw.text(
        ((right_left + right_right - (box[2] - box[0])) / 2, 39),
        header,
        fill=(0, 0, 0),
        font=font(62, bold=True),
    )

    metric_panel = build_metric_panel(right_right - right_left - 120, right_bottom - 330)
    canvas.paste(metric_panel, (right_left + 60, 250))
    canvas.save(OUTPUT, quality=95)
    canvas.save(OUTPUT_PDF, "PDF", resolution=300.0)
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
