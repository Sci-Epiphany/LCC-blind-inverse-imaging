import numpy as np
import torch

from eval import Evaluator


class _IdentityMetric:
    name = "metric"

    def __call__(self, gt, measurement, estimate, reduction="mean"):
        value = estimate.flatten(1).mean(1)
        return value.mean() if reduction == "mean" else value


def test_multirun_display_reports_run_mean_not_best_run() -> None:
    evaluator = Evaluator((_IdentityMetric(),))
    gt = torch.zeros((1, 2, 1, 1))
    measurement = torch.zeros_like(gt)
    estimates = torch.tensor([
        [[[[0.2]]], [[[0.6]]]],
        [[[[0.8]]], [[[1.0]]]],
    ])

    result = evaluator.report(gt, measurement, estimates)

    assert np.allclose(result["metric"]["mean"], [0.5, 0.8])
    rendered = evaluator.display(result)
    assert "0.500000" in rendered
    assert "0.800000" in rendered
    assert "0.65" in rendered
