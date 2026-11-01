"""Verify every numerical table against retained result artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE_DATA = HERE / "source_data"

# Values printed in LCC.tex.  The verifier checks these against artifacts.
LINEAR = {
    "Gaussian": {
        "SelfDeblur": (16.61, 0.2745, 0.5456, 8.1e-5),
        "DeblurGAN": (25.33, 0.7528, 0.3365, None),
        "BKSE": (18.70, 0.4994, 0.4083, None),
        "FFTFormer": (23.30, 0.7615, 0.3166, None),
        "DeepRFT": (25.41, 0.7566, 0.3305, None),
        "BlindDPS": (24.43, 0.7631, 0.2366, 1.06e-5),
        "FastDiffEM": (12.34, 0.3863, 0.5297, 3.23e-5),
        "DMPlug": (23.89, 0.7347, 0.3124, 1.60e-4),
        "LCC": (27.58, 0.7934, 0.1914, 1.528e-7),
    },
    "Motion": {
        "SelfDeblur": (17.07, 0.3351, 0.5006, 6.05e-5),
        "DeblurGAN": (21.19, 0.6406, 0.3913, None),
        "BKSE": (20.04, 0.5886, 0.5104, None),
        "FFTFormer": (19.33, 0.6040, 0.4042, None),
        "DeepRFT": (22.93, 0.7028, 0.3080, None),
        "BlindDPS": (19.80, 0.6040, 0.3369, 1.97e-5),
        "FastDiffEM": (13.29, 0.3084, 0.5213, 2.92e-5),
        "DMPlug": (18.43, 0.5418, 0.4376, 1.71e-4),
        "LCC": (21.16, 0.5851, 0.3457, 3.916e-6),
    },
}

NONLINEAR = {
    "SelfDeblur": (17.1343, 0.4879, 0.3192),
    "DeblurGAN": (27.2708, 0.2641, 0.7969),
    "BKSE": (23.3811, 0.4094, 0.6681),
    "FFTFormer": (23.5855, 0.2824, 0.7569),
    "DeepRFT": (27.6971, 0.2477, 0.8195),
    "BlindDPS": (12.3113, 0.5020, 0.3771),
    "FastDiffEM": (14.9957, 0.4901, 0.3638),
    "DMPlug": (23.6289, 0.3301, 0.6865),
    "LCC": (23.6856, 0.2669, 0.7020),
}

NONREFERENCE = {
    "SelfDeblur": (7.2100, 5.3720, 0.0893),
    "DeblurGAN": (7.0060, 4.1620, 0.1477),
    "BKSE": (6.5160, 5.6900, 0.1048),
    "FFTFormer": (7.8740, 3.7220, 0.1693),
    "DeepRFT": (8.7000, 3.6570, 0.1373),
    "BlindDPS": (9.0570, 4.6590, 0.2313),
    "FastDiffEM": (10.1400, 4.8500, 0.1827),
    "DMPlug": (9.9470, 3.1730, 0.1212),
    "LCC": (7.4760, 5.1630, 0.1539),
}

ABLATION = {
    "Exact Jacobian": (25.2123, 0.7196, 0.2581, 2.887e-6),
    "+ image refinement": (25.4385, 0.7196, 0.2535, 2.887e-6),
    "+ both refinements": (26.3833, 0.7527, 0.2139, 4.680e-7),
    "Two-branch fusion": (27.5842, 0.7934, 0.1914, 1.528e-7),
}

PAIRED = {
    ("Gaussian", "previous LCC"): (2.298, 2.095, 2.498, 97, 3, 0.0584, -0.0281),
    ("Motion", "previous LCC"): (1.075, 0.984, 1.167, 681, 119, 0.0155, -0.0063),
}

SENSITIVITY = {
    "eta_x_lpips": {
        "cells": ["B2", "C2", "D2", "E2", "F2"],
        "expected": [0.2329, 0.2275, 0.2261, 0.2314, 0.2350],
    },
    "eta_k_lpips": {
        "cells": ["B8", "C8", "D8", "E8", "F8"],
        "expected": [0.2353, 0.2263, 0.2261, 0.2288, 0.2307],
    },
    "lambda_lpips": {
        "cells": ["B13", "C13", "D13", "E13", "F13"],
        "expected": [0.2271, 0.2270, 0.2261, 0.2263, 0.2298],
    },
}

PNP_MULTISEED = {
    "L1": (25.0025, 0.7214, 0.2172, 1.492e-6),
    "PnP": (26.6310, 0.7510, 0.2050, 1.358e-6),
}

LOOP_SCHEDULE = {
    "baseline 200/100, ODE 5": (25.0025, 0.2938, 0.2172, 1.492e-6),
    "outer 400, MCMC 50, 2x step": (24.6899, 0.2146, 0.2193, 1.496e-6),
}

PROPOSAL = {
    "one stochastic proposal": (20.2294, 0.5250, 0.4841, 5.493),
    "four proposals, deterministic mean": (17.2974, 0.4597, 0.5455, 1.545),
    "four proposals, 0.5x noise": (19.7270, 0.5142, 0.4909, 2.635),
    "four proposals, 1.0x noise": (19.4015, 0.4988, 0.5086, 5.728),
}

NONLINEAR_ABLATION = {
    "original 50×20": (15.6170, 0.3697, 0.6366),
    "BKSE input/sign fix": (19.1560, 0.4609, 0.5896),
    "integrated kernel MCMC": (21.5424, 0.5165, 0.5425),
    "image step 3e-5": (22.9883, 0.5782, 0.4563),
    "100×50 schedule": (23.6011, 0.6075, 0.4117),
}

NONLINEAR_EXTRA = {
    "100×50, image noise 1.0": (23.6011, 0.6075, 0.4117),
    "image noise 0.5": (23.6174, 0.6116, 0.4097),
    "image noise 0.1": (23.6401, 0.6131, 0.4091),
    "image noise 0.0": (23.6267, 0.6129, 0.4092),
    "observation-selected proposal": (23.5502, 0.6057, 0.4067),
    "post-MCMC kernel reprojection": (23.5632, 0.6108, 0.4091),
    "200×25 schedule": (23.0745, 0.5833, 0.4566),
    "200×50 schedule": (23.5402, 0.6061, 0.4130),
}

CROSS_PNP = (25.2441, 0.7356, 0.2228, 2.125e-6, 300)
FIXED_PNP = {
    "Gaussian initial": (25.2123, 0.7196, 0.2581, 2.887e-6, 100),
    "Gaussian + image": (25.4385, 0.7196, 0.2535, 2.887e-6, 100),
    "Gaussian + both": (26.3833, 0.7527, 0.2139, 4.680e-7, 100),
    "Gaussian fusion": (27.5842, 0.7934, 0.1914, 1.528e-7, 100),
    "Motion + both": (21.1626, 0.5851, 0.3457, 3.916e-6, 796),
}
AUTOTUNE = {
    "base": (26.5444, 0.2018),
    "selected": (27.6135, 0.1897, -0.3637, 1),
}


def read_csv_map(path: Path, key_name: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key_name]: row for row in csv.DictReader(handle)}


def read_xlsx_cells(path: Path) -> dict[str, str]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(node.text or "" for node in item.findall(".//m:t", ns))
            for item in strings_root.findall("m:si", ns)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    cells = {}
    for cell in sheet.findall(".//m:sheetData/m:row/m:c", ns):
        value = cell.find("m:v", ns)
        if value is None:
            continue
        text = value.text or ""
        if cell.get("t") == "s":
            text = strings[int(text)]
        cells[cell.get("r", "")] = text
    return cells


def close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def main() -> None:
    with (RESULTS / "distribution_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        comparison_rows = list(csv.DictReader(handle))
    own = read_csv_map(RESULTS / "ldcc_full_metrics.csv", "benchmark")
    optimized_path = HERE / "lcc_opt_results" / "lcc_optimization.json"
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
    nr_rows = read_csv_map(SOURCE_DATA / "NR_IQA_results.csv", "Method")
    ablation_cells = read_xlsx_cells(SOURCE_DATA / "super_parameter.xlsx")

    checks = []
    errors = []

    def comparison_name(benchmark: str, method: str) -> str:
        # analyze_existing_results.py canonicalizes historical source labels.
        return method

    for benchmark, methods in LINEAR.items():
        for method, (psnr, ssim, lpips, expected_kmse) in methods.items():
            if method == "LCC":
                if benchmark == "Gaussian":
                    result = optimized["full_gaussian_100_seed123"][
                        "two_branch_equal_fusion"
                    ]
                else:
                    result = optimized["full_motion_200x4"]
                    result = {
                        "psnr": result["post_image"]["psnr"],
                        "ssim": result["post_image"]["ssim"],
                        "lpips": result["post_image"]["lpips"],
                        "kernel_mse": result["post_kernel_mse"],
                    }
                actual = (
                    result["psnr"],
                    result["ssim"],
                    result["lpips"],
                    result["kernel_mse"],
                )
                source = "lcc_optimization.json"
            else:
                row = next(
                    item
                    for item in comparison_rows
                    if item["benchmark"] == benchmark
                    and item["method"] == comparison_name(benchmark, method)
                )
                actual = (
                    float(row["psnr_mean"]),
                    float(row["ssim_mean"]),
                    float(row["lpips_mean"]),
                    float(row["kernel_mse_mean"]),
                )
                source = "distribution_audit.csv"
            expected = (psnr, ssim, lpips, expected_kmse)
            tolerances = (0.006, 0.0006, 0.0006)
            ok = all(close(a, e, t) for a, e, t in zip(actual[:3], expected[:3], tolerances))
            if expected_kmse is not None:
                ok = ok and close(
                    actual[3], expected_kmse, max(1e-12, expected_kmse * 0.03)
                )
            checks.append(
                dict(
                    table="linear",
                    group=benchmark,
                    method=method,
                    expected=expected,
                    actual=actual,
                    source=source,
                    pass_=ok,
                )
            )
            if not ok:
                errors.append(f"linear {benchmark} {method}: {actual} != {expected}")

    for method, (psnr, lpips, ssim) in NONLINEAR.items():
        if method == "LCC":
            row = own["Nonlinear"]
            actual = (float(row["psnr_mean"]), float(row["lpips_mean"]), float(row["ssim_mean"]))
            source = "ldcc_full_metrics.csv"
        else:
            row = next(
                item
                for item in comparison_rows
                if item["benchmark"] == "Nonlinear"
                and item["method"] == comparison_name("Nonlinear", method)
            )
            actual = (float(row["psnr_mean"]), float(row["lpips_mean"]), float(row["ssim_mean"]))
            source = "distribution_audit.csv"
        expected = (psnr, lpips, ssim)
        tolerances = (0.006, 0.0006, 0.0006)
        ok = all(close(a, e, t) for a, e, t in zip(actual, expected, tolerances))
        checks.append(
            dict(
                table="nonlinear",
                group="FFHQ",
                method=method,
                expected=expected,
                actual=actual,
                source=source,
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"nonlinear {method}: {actual} != {expected}")

    nonlinear_opt = json.loads(
        (RESULTS / "nonlinear_optimization.json").read_text(encoding="utf-8")
    )
    nonlinear_checks = (
        ("nonlinear ablation", "four-image mechanism", NONLINEAR_ABLATION,
         nonlinear_opt["four_image_schedule_ablation"]),
        ("nonlinear thermal", "four-image coupling", NONLINEAR_EXTRA,
         nonlinear_opt["four_image_coupling_and_noise_ablation"]),
    )
    for table_name, group, expected_rows, actual_group in nonlinear_checks:
        for label, expected in expected_rows.items():
            row = actual_group[label]
            actual = (row["psnr"], row["ssim"], row["lpips"])
            ok = all(
                close(value, reference, tolerance)
                for value, reference, tolerance in zip(
                    actual,
                    expected,
                    (0.0001, 0.0001, 0.0001),
                    strict=True,
                )
            )
            checks.append(
                dict(
                    table=table_name,
                    group=group,
                    method=label,
                    expected=expected,
                    actual=actual,
                    source="nonlinear_optimization.json",
                    pass_=ok,
                )
            )
            if not ok:
                errors.append(
                    f"{table_name} {label}: {actual} != {expected}"
                )

    nr_alias = {"FastDiffEM": "fastem", "BKSE": "bkse", "BlindDPS": "blind-dps", "DeblurGAN": "DeblurGAN2", "LCC": "daps", "DeepRFT": "deeprft", "FFTFormer": "fftformer"}
    for method, expected in NONREFERENCE.items():
        row = nr_rows[nr_alias.get(method, method)]
        actual = (float(row["NIQE"]), float(row["NRQM"]), float(row["MANIQA"]))
        ok = all(close(a, e, 0.001) for a, e in zip(actual, expected))
        checks.append(
            dict(
                table="nonreference",
                group="ImageNet",
                method=method,
                expected=expected,
                actual=actual,
                source="NR_IQA_results.csv",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"nonreference {method}: {actual} != {expected}")

    gaussian = optimized["full_gaussian_100_seed123"]
    ablation_actual = {
        "Exact Jacobian": (
            gaussian["initial"]["psnr"],
            gaussian["initial"]["ssim"],
            gaussian["initial"]["lpips"],
            gaussian["initial_kernel_mse"],
        ),
        "+ image refinement": (
            gaussian["image_only"]["psnr"],
            gaussian["image_only"]["ssim"],
            gaussian["image_only"]["lpips"],
            gaussian["initial_kernel_mse"],
        ),
        "+ both refinements": (
            gaussian["post_image"]["psnr"],
            gaussian["post_image"]["ssim"],
            gaussian["post_image"]["lpips"],
            gaussian["post_kernel_mse"],
        ),
        "Two-branch fusion": (
            gaussian["two_branch_equal_fusion"]["psnr"],
            gaussian["two_branch_equal_fusion"]["ssim"],
            gaussian["two_branch_equal_fusion"]["lpips"],
            gaussian["two_branch_equal_fusion"]["kernel_mse"],
        ),
    }
    for method, actual in ablation_actual.items():
        expected = ABLATION[method]
        ok = (
            close(actual[0], expected[0], 0.006)
            and close(actual[1], expected[1], 0.0006)
            and close(actual[2], expected[2], 0.0006)
            and close(actual[3], expected[3], max(1e-12, expected[3] * 0.03))
        )
        checks.append(
            dict(
                table="ablation",
                group="Gaussian development",
                method=method,
                expected=expected,
                actual=actual,
                source="lcc_optimization.json",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"ablation {method}: {actual} != {expected}")

    for parameter, spec in SENSITIVITY.items():
        for cell, expected in zip(spec["cells"], spec["expected"]):
            actual = float(ablation_cells[cell])
            tolerance = 0.0001 if abs(expected) >= 0.01 else max(1e-12, abs(expected) * 0.03)
            ok = close(actual, expected, tolerance)
            checks.append(
                dict(
                    table="sensitivity",
                    group=parameter,
                    method=cell,
                    expected=(expected,),
                    actual=(actual,),
                    source="super_parameter.xlsx",
                    pass_=ok,
                )
            )
            if not ok:
                errors.append(f"sensitivity {parameter} {cell}: {actual} != {expected}")

    with (RESULTS / "pnp_multiseed.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        pnp_rows = {row["regularizer"]: row for row in csv.DictReader(handle)}
    for regularizer, expected in PNP_MULTISEED.items():
        row = pnp_rows[regularizer]
        actual = (
            float(row["psnr"]),
            float(row["ssim"]),
            float(row["lpips"]),
            float(row["kmse"]),
        )
        ok = (
            close(actual[0], expected[0], 0.006)
            and close(actual[1], expected[1], 0.0006)
            and close(actual[2], expected[2], 0.0006)
            and close(actual[3], expected[3], max(1e-12, expected[3] * 0.03))
        )
        checks.append(
            dict(
                table="regularizer",
                group="Gaussian multi-seed",
                method=regularizer,
                expected=expected,
                actual=actual,
                source="pnp_multiseed.csv",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"regularizer {regularizer}: {actual} != {expected}")

    with (RESULTS / "loop_multiseed_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        loop_rows = {row["variant"]: row for row in csv.DictReader(handle)}
    for variant, expected in LOOP_SCHEDULE.items():
        row = loop_rows[variant]
        actual = (
            float(row["psnr_mean"]),
            float(row["seed_mean_std"]),
            float(row["lpips_mean"]),
            float(row["kmse_mean"]),
        )
        ok = (
            close(actual[0], expected[0], 0.006)
            and close(actual[1], expected[1], 0.0006)
            and close(actual[2], expected[2], 0.0006)
            and close(actual[3], expected[3], max(1e-12, expected[3] * 0.03))
        )
        checks.append(
            dict(
                table="schedule",
                group="Gaussian multi-seed",
                method=variant,
                expected=expected,
                actual=actual,
                source="loop_multiseed_summary.csv",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"schedule {variant}: {actual} != {expected}")

    with (RESULTS / "proposal_tuning.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        proposal_rows = {row["variant"]: row for row in csv.DictReader(handle)}
    for variant, expected in PROPOSAL.items():
        row = proposal_rows[variant]
        actual = (
            float(row["psnr"]),
            float(row["ssim"]),
            float(row["lpips"]),
            float(row["kmse"]),
        )
        ok = all(
            close(actual_value, expected_value, tolerance)
            for actual_value, expected_value, tolerance in zip(
                actual,
                expected,
                (0.006, 0.0006, 0.0006, 0.003),
                strict=True,
            )
        )
        checks.append(
            dict(
                table="proposal",
                group="KernelWizard diagnostic",
                method=variant,
                expected=expected,
                actual=actual,
                source="proposal_tuning.csv",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"proposal {variant}: {actual} != {expected}")

    cross_pnp = json.loads((RESULTS / "full_pnp.json").read_text(encoding="utf-8"))
    actual = (
        float(cross_pnp["psnr"]),
        float(cross_pnp["ssim"]),
        float(cross_pnp["lpips"]),
        float(cross_pnp["kernel_mse"]),
        int(cross_pnp["success_10db"]),
    )
    ok = (
        close(actual[0], CROSS_PNP[0], 0.006)
        and close(actual[1], CROSS_PNP[1], 0.0006)
        and close(actual[2], CROSS_PNP[2], 0.0006)
        and close(actual[3], CROSS_PNP[3], max(1e-12, CROSS_PNP[3] * 0.03))
        and actual[4] == CROSS_PNP[4]
    )
    checks.append(
        dict(
            table="cross-kernel PnP",
            group="3 seeds",
            method="PnP",
            expected=CROSS_PNP,
            actual=actual,
            source="full_pnp.json",
            pass_=ok,
        )
    )
    if not ok:
        errors.append(f"cross-kernel PnP: {actual} != {CROSS_PNP}")

    motion = optimized["full_motion_200x4"]
    fixed_actual = {
        "Gaussian initial": (
            gaussian["initial"]["psnr"],
            gaussian["initial"]["ssim"],
            gaussian["initial"]["lpips"],
            gaussian["initial_kernel_mse"],
            100,
        ),
        "Gaussian + image": (
            gaussian["image_only"]["psnr"],
            gaussian["image_only"]["ssim"],
            gaussian["image_only"]["lpips"],
            gaussian["initial_kernel_mse"],
            100,
        ),
        "Gaussian + both": (
            gaussian["post_image"]["psnr"],
            gaussian["post_image"]["ssim"],
            gaussian["post_image"]["lpips"],
            gaussian["post_kernel_mse"],
            100,
        ),
        "Gaussian fusion": (
            gaussian["two_branch_equal_fusion"]["psnr"],
            gaussian["two_branch_equal_fusion"]["ssim"],
            gaussian["two_branch_equal_fusion"]["lpips"],
            gaussian["two_branch_equal_fusion"]["kernel_mse"],
            100,
        ),
        "Motion + both": (
            motion["post_image"]["psnr"],
            motion["post_image"]["ssim"],
            motion["post_image"]["lpips"],
            motion["post_kernel_mse"],
            796,
        ),
    }
    for task, expected in FIXED_PNP.items():
        actual = fixed_actual[task]
        ok = (
            close(actual[0], expected[0], 0.006)
            and close(actual[1], expected[1], 0.0006)
            and close(actual[2], expected[2], 0.0006)
            and close(actual[3], expected[3], max(1e-12, expected[3] * 0.03))
            and actual[4] == expected[4]
        )
        checks.append(
            dict(
                table="fixed-input PnP",
                group=task,
                method="corrected PnP",
                expected=expected,
                actual=actual,
                source="lcc_optimization.json",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"fixed-input PnP {task}: {actual} != {expected}")

    autotune = json.loads((RESULTS / "autotune_lrktau.json").read_text(encoding="utf-8"))
    search_rows = {row["name"]: row for row in autotune["search_runs"]}
    base = search_rows["seed42_paper200m100_pnp"]
    selected = search_rows["autotune_ei_0"]
    rank0 = next(
        row for row in autotune["validation_ranking"] if row["rank"] == 0
    )
    autotune_actual = (
        float(base["psnr"]),
        float(base["lpips"]),
        float(selected["psnr"]),
        float(selected["lpips"]),
        float(rank0["mean_delta_psnr"]),
        int(rank0["wins"]),
    )
    autotune_expected = (
        AUTOTUNE["base"][0],
        AUTOTUNE["base"][1],
        AUTOTUNE["selected"][0],
        AUTOTUNE["selected"][1],
        AUTOTUNE["selected"][2],
        AUTOTUNE["selected"][3],
    )
    ok = (
        close(autotune_actual[0], autotune_expected[0], 0.006)
        and close(autotune_actual[1], autotune_expected[1], 0.0006)
        and close(autotune_actual[2], autotune_expected[2], 0.006)
        and close(autotune_actual[3], autotune_expected[3], 0.0006)
        and close(autotune_actual[4], autotune_expected[4], 0.006)
        and autotune_actual[5] == autotune_expected[5]
    )
    checks.append(
        dict(
            table="autotune",
            group="Gaussian 4-image pilot",
            method="GP-EI",
            expected=autotune_expected,
            actual=autotune_actual,
            source="autotune_lrktau.json",
            pass_=ok,
        )
    )
    if not ok:
        errors.append(
            f"autotune: {autotune_actual} != {autotune_expected}"
        )

    for key, expected in PAIRED.items():
        if key[0] == "Gaussian":
            result = optimized["full_gaussian_100_seed123"][
                "two_branch_equal_fusion"
            ]["paired_image_vs_old_pnp"]
        else:
            result = optimized["full_motion_200x4"][
                "paired_image_vs_old_pnp"
            ]
        actual = (
            result["psnr"]["delta"],
            result["psnr"]["ci_low"],
            result["psnr"]["ci_high"],
            result["psnr"]["wins"],
            result["psnr"]["losses"],
            result["ssim"]["delta"],
            result["lpips"]["delta"],
        )
        ok = (
            close(actual[0], expected[0], 0.006)
            and close(actual[1], expected[1], 0.006)
            and close(actual[2], expected[2], 0.006)
            and actual[3] == expected[3]
            and actual[4] == expected[4]
            and close(actual[5], expected[5], 0.0006)
            and close(actual[6], expected[6], 0.0006)
        )
        checks.append(
            dict(
                table="paired",
                group=key[0],
                method=key[1],
                expected=expected,
                actual=actual,
                source="lcc_optimization.json",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"paired {key}: {actual} != {expected}")

    ensemble_summary = []
    for benchmark in ("gaussian", "motion"):
        payload = json.loads(
            (
                HERE
                / "remote_fixed_pnp_ensemble"
                / benchmark
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        for row in payload["summary"]:
            ensemble_summary.append({**row, "benchmark": benchmark.capitalize()})
    expected_ensemble = {
        ("Gaussian", 1): (25.162, 0.7303, 0.2237, 0.000, 0),
        ("Gaussian", 2): (25.668, 0.7604, 0.2147, 0.506, 99),
        ("Gaussian", 4): (25.971, 0.7773, 0.2124, 0.809, 100),
        ("Motion", 1): (20.082, 0.5683, 0.3522, 0.000, 0),
        ("Motion", 2): (20.860, 0.6105, 0.3421, 0.777, 200),
        ("Motion", 4): (21.539, 0.6411, 0.3377, 1.456, 199),
    }
    for row in ensemble_summary:
        expected = expected_ensemble[(row["benchmark"], int(row["samples"]))]
        actual = (
            float(row["psnr"]),
            float(row["ssim"]),
            float(row["lpips"]),
            float(row["delta_psnr"]),
            int(row["wins"]),
        )
        ok = (
            close(actual[0], expected[0], 0.001)
            and close(actual[1], expected[1], 0.001)
            and close(actual[2], expected[2], 0.0001)
            and close(actual[3], expected[3], 0.001)
            and actual[4] == expected[4]
        )
        checks.append(
            dict(
                table="ensemble",
                group=row["benchmark"],
                method=f"{row['samples']}-sample average",
                expected=expected,
                actual=actual,
                source=f"remote_fixed_pnp_ensemble/{row['benchmark'].lower()}/summary.json",
                pass_=ok,
            )
        )
        if not ok:
            errors.append(f"ensemble {row['benchmark']}: {actual} != {expected}")

    if errors:
        raise RuntimeError("\n".join(errors))

    output = {
        "source": {
            "comparison": str(RESULTS / "distribution_audit.csv"),
            "lcc": str(RESULTS / "ldcc_full_metrics.csv"),
            "nonreference": str(SOURCE_DATA / "NR_IQA_results.csv"),
            "ablation": str(SOURCE_DATA / "super_parameter.xlsx"),
        },
        "checks": checks,
        "passed": len(checks),
        "failed": 0,
    }
    (RESULTS / "table_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Table audit",
        "",
        f"Verified {len(checks)} numerical rows against retained artifacts.",
        "",
        "| Table | Group | Method | Printed | Source | Status |",
        "|---|---|---|---|---|---|",
    ]
    for item in checks:
        printed = ", ".join(
            "--" if value is None else f"{value:.6g}" for value in item["expected"]
        )
        report.append(
            f"| {item['table']} | {item['group']} | {item['method']} | "
            f"{printed} | `{item['source']}` | PASS |"
        )
    report.extend(
        [
            "",
            "All table rows use full available samples. Filtered values are "
            "reported separately in the failure-aware audit.",
        ]
    )
    (RESULTS / "TABLE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Verified {len(checks)} table rows; all source values agree.")


if __name__ == "__main__":
    main()
