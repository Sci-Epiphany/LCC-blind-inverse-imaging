"""Deterministic multi-fidelity search for the sensitive kernel-rate/noise pair.

Run on the remote GPU host. The search uses four tuning images and seed 42,
then validates the best candidates on three held-out seeds. The objective
combines fidelity and perceptual quality; Kernel MSE is a weak tie-breaker.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path("/home/gsn/PhD_thesis/DAPS")
PYTHON = Path("/home/gsn/anaconda3/envs/BlindDPS/bin/python")
RESULTS = ROOT / "autotune_lrktau"
LOGS = ROOT / "tuning_logs"
RESULT_ROOT = ROOT / "tuning_results"

LR_BOUNDS = (1.0e-9, 1.6e-8)
TAU_BOUNDS = (0.005, 0.040)
TUNING_SEED = 42
VALIDATION_SEEDS = (123, 2026, 7)
TUNING_IMAGES = 4


@dataclass(frozen=True)
class Candidate:
    lr_kernel: float
    tau: float


def to_unit(candidate: Candidate) -> np.ndarray:
    log_low, log_high = np.log10(LR_BOUNDS)
    return np.asarray(
        [
            (math.log10(candidate.lr_kernel) - log_low) / (log_high - log_low),
            (candidate.tau - TAU_BOUNDS[0]) / (TAU_BOUNDS[1] - TAU_BOUNDS[0]),
        ]
    )


def from_unit(point: np.ndarray) -> Candidate:
    log_low, log_high = np.log10(LR_BOUNDS)
    point = np.clip(point, 0.0, 1.0)
    return Candidate(
        lr_kernel=float(
            10.0 ** (log_low + point[0] * (log_high - log_low))
        ),
        tau=float(TAU_BOUNDS[0] + point[1] * (TAU_BOUNDS[1] - TAU_BOUNDS[0])),
    )


def metrics_from_dir(directory: Path) -> dict[str, float]:
    data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return {
        "psnr": mean(row[0] for row in data["image"]["psnr"]["sample"]),
        "ssim": mean(row[0] for row in data["image"]["ssim"]["sample"]),
        "lpips": mean(row[0] for row in data["image"]["lpips"]["sample"]),
        "kmse": mean(row[0] for row in data["kernel"]["kernel_mse"]["sample"]),
    }


def objective(metrics: dict[str, float]) -> float:
    return (
        metrics["psnr"]
        + 10.0 * metrics["ssim"]
        - 10.0 * metrics["lpips"]
        - 0.2 * math.log10(metrics["kmse"] / 1.0e-6)
    )


def run_candidate(name: str, seed: int, candidate: Candidate) -> dict[str, float]:
    out = RESULT_ROOT / name
    if (out / "metrics.json").is_file():
        return metrics_from_dir(out)
    command = [
        str(PYTHON),
        "posterior_sample.py",
        "+data=test-ffhq",
        "+kernel_data=gaussian64",
        "+model=ffhq256ddpm",
        "+kernel_model=kernel64ddpm",
        "+task=blind_gaussian_blur",
        "+sampler=edm_daps",
        "task_group=pixel",
        "save_dir=tuning_results",
        f"name={name}",
        f"seed={seed}",
        "num_runs=1",
        "save_samples=false",
        "batch_size=4",
        "data.start_id=0",
        f"data.end_id={TUNING_IMAGES}",
        "sampler.blind=True",
        "sampler.nonlinear=False",
        "sampler.annealing_scheduler_config.num_steps=200",
        "sampler.diffusion_scheduler_config.num_steps=5",
        "task.pixel.mcmc_sampler_config.num_steps=100",
        "task.pixel.mcmc_sampler_config.lr.image=5e-5",
        f"task.pixel.mcmc_sampler_config.lr.kernel={candidate.lr_kernel}",
        "task.pixel.mcmc_sampler_config.reg_terms.lamb=20",
        "task.pixel.mcmc_sampler_config.reg_terms.beta=2000000",
        f"task.pixel.mcmc_sampler_config.tau={candidate.tau}",
        "+task.pixel.mcmc_sampler_config.kernel_regularizer=pnp",
        "gpu=0",
    ]
    log_path = LOGS / f"{name}.log"
    print(
        f"RUN {name}: lr_k={candidate.lr_kernel:.4e}, tau={candidate.tau:.5f}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    return metrics_from_dir(out)


def rbf(left: np.ndarray, right: np.ndarray, length_scale: float = 0.22) -> np.ndarray:
    distance = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-distance / (2.0 * length_scale**2))


def propose(points: np.ndarray, scores: np.ndarray, count: int) -> list[np.ndarray]:
    best = float(scores.max())
    improvement = max(0.05, 0.05 * float(np.std(scores)))
    rng = np.random.default_rng(20261031)
    candidates = rng.uniform(0.0, 1.0, size=(2000, 2))
    if len(points) == 0:
        return list(candidates[:count])
    kernel = rbf(points, points) + np.eye(len(points)) * 1.0e-6
    alpha = np.linalg.solve(kernel, scores - scores.mean())
    mean_offset = float(scores.mean())
    cross = rbf(candidates, points)
    predicted = mean_offset + cross @ alpha
    solved = np.linalg.solve(kernel, cross.T)
    variance = np.maximum(1.0e-12, 1.0 - np.sum(cross * solved.T, axis=1))
    std = np.sqrt(variance)
    z = (predicted - best - improvement) / std
    normal_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    normal_pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    acquisition = (predicted - best - improvement) * normal_cdf + std * normal_pdf
    chosen: list[np.ndarray] = []
    for index in np.argsort(acquisition)[::-1]:
        point = candidates[index]
        if all(np.linalg.norm(point - previous) > 0.08 for previous in chosen):
            chosen.append(point)
        if len(chosen) == count:
            break
    return chosen


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    # Reuse two already evaluated points from the manuscript campaign.
    known = [
        (
            Candidate(4e-9, 0.02),
            metrics_from_dir(RESULT_ROOT / "seed42_paper200m100_pnp"),
        ),
        (
            Candidate(2e-9, 0.01),
            metrics_from_dir(RESULT_ROOT / "seed42_pnp_lrk2e9_tau001"),
        ),
    ]
    evaluated = [
        {
            "name": name,
            "seed": TUNING_SEED,
            "candidate": asdict(candidate),
            "metrics": metrics,
            "objective": objective(metrics),
        }
        for name, (candidate, metrics) in zip(
            ("cached_base", "cached_lrk2e9_tau001"), known
        )
    ]

    initial = [
        Candidate(8e-9, 0.02),
        Candidate(4e-9, 0.01),
        Candidate(8e-9, 0.035),
        Candidate(1.2e-8, 0.01),
    ]
    for index, candidate in enumerate(initial):
        metrics = run_candidate(f"autotune_r0_{index}", TUNING_SEED, candidate)
        evaluated.append(
            {
                "name": f"autotune_r0_{index}",
                "seed": TUNING_SEED,
                "candidate": asdict(candidate),
                "metrics": metrics,
                "objective": objective(metrics),
            }
        )

    points = np.stack([to_unit(Candidate(**row["candidate"])) for row in evaluated])
    scores = np.asarray([row["objective"] for row in evaluated])
    for index, point in enumerate(propose(points, scores, 3)):
        candidate = from_unit(point)
        name = f"autotune_ei_{index}"
        metrics = run_candidate(name, TUNING_SEED, candidate)
        evaluated.append(
            {
                "name": name,
                "seed": TUNING_SEED,
                "candidate": asdict(candidate),
                "metrics": metrics,
                "objective": objective(metrics),
            }
        )

    finalists = sorted(evaluated, key=lambda row: row["objective"], reverse=True)[:3]
    validation = []
    for rank, finalist in enumerate(finalists):
        candidate = Candidate(**finalist["candidate"])
        for seed in VALIDATION_SEEDS:
            name = f"autotune_validate_{rank}_seed{seed}"
            metrics = run_candidate(name, seed, candidate)
            baseline = metrics_from_dir(
                RESULT_ROOT / f"seed{seed}_paper200m100_pnp"
            )
            validation.append(
                {
                    "rank": rank,
                    "seed": seed,
                    "candidate": asdict(candidate),
                    "metrics": metrics,
                    "baseline": baseline,
                    "delta_psnr": metrics["psnr"] - baseline["psnr"],
                    "delta_objective": objective(metrics) - objective(baseline),
                }
            )

    ranking = []
    for rank, finalist in enumerate(finalists):
        rows = [row for row in validation if row["rank"] == rank]
        ranking.append(
            {
                "rank": rank,
                "candidate": finalist["candidate"],
                "tuning_objective": finalist["objective"],
                "validation_mean_delta_psnr": mean(row["delta_psnr"] for row in rows),
                "validation_mean_delta_objective": mean(
                    row["delta_objective"] for row in rows
                ),
                "validation_wins": sum(row["delta_objective"] > 0 for row in rows),
            }
        )
    ranking.sort(key=lambda row: row["validation_mean_delta_objective"], reverse=True)
    output = {
        "search_space": {
            "lr_kernel": LR_BOUNDS,
            "tau": TAU_BOUNDS,
            "tuning_images": TUNING_IMAGES,
            "tuning_seed": TUNING_SEED,
            "validation_seeds": VALIDATION_SEEDS,
        },
        "evaluated": evaluated,
        "validation": validation,
        "ranking": ranking,
        "selected": ranking[0],
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
