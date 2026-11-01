import torch

import forward_operator
from cores.mcmc_nl import MCMCNLSampler
from forward_operator import BlindNonlinearBlur


class _KernelWizardRecorder:
    def __init__(self) -> None:
        self.inputs = None

    def __call__(self, image, measurement):
        self.inputs = (image.detach().clone(), measurement.detach().clone())
        mean = torch.ones(1, 512, 2, 2)
        sigma = torch.zeros_like(mean)
        return mean, sigma


class _ZeroOperator:
    sigma = 1.0

    def gradient(self, state, measurement, return_loss=False):
        gradients = {key: torch.zeros_like(value) for key, value in state.items()}
        loss = torch.zeros(())
        return (gradients, loss) if return_loss else gradients


def test_kernelwizard_solution_uses_unit_interval_inputs() -> None:
    operator = object.__new__(BlindNonlinearBlur)
    operator.blur_model = _KernelWizardRecorder()
    operator.proposal_mode = "mean"
    operator.proposal_candidates = 1
    operator.proposal_noise_scale = 1.0

    image = torch.full((1, 3, 8, 8), -1.0)
    measurement = torch.full((1, 3, 8, 8), 1.0)
    operator.call_solution(image, measurement)

    for value in operator.blur_model.inputs:
        assert float(value.min()) >= 0.0
        assert float(value.max()) <= 1.0


def test_nonlinear_kernel_prior_is_attractive() -> None:
    sampler = MCMCNLSampler(
        num_steps=1,
        lr={"image": 0.0, "kernel": 0.0},
        tau=1.0,
        lr_min_ratio={"image": 1.0, "kernel": 1.0},
        prior_solver="gaussian",
        reg_terms={"lamb": 0.0, "beta": 1.0},
        kernel_prior_weight=1e5,
        image_noise_scale=1.0,
    )
    state = {
        "image": torch.zeros(1, 3, 8, 8),
        "kernel": torch.zeros(1, 512, 2, 2),
    }
    clean = {
        "image": torch.zeros(1, 3, 8, 8),
        "kernel": torch.ones(1, 512, 2, 2),
    }
    sampler.prepare_prior_score(clean["image"], state["image"], None, 1.0)
    score, _ = sampler.score_fn(
        state, clean, {}, state, _ZeroOperator(), None, 1.0
    )
    assert torch.all(score["kernel"] > 0)


def test_select_mode_chooses_observation_consistent_kernel(monkeypatch) -> None:
    operator = object.__new__(BlindNonlinearBlur)
    operator.blur_model = _KernelWizardRecorder()
    operator.proposal_mode = "select"
    operator.proposal_candidates = 3
    operator.proposal_noise_scale = 1.0

    def fake_randn_like(value):
        candidate = fake_randn_like.values.pop(0)
        return torch.full_like(value, candidate)

    fake_randn_like.values = [-1.0, 0.0, 1.0]
    monkeypatch.setattr(forward_operator.torch, "randn_like", fake_randn_like)
    operator.loss = lambda state, measurement: (
        (state["kernel"] - 1.0).pow(2).flatten(1).sum(-1)
    )

    kernel = operator.call_solution(
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
    )

    assert torch.allclose(kernel, torch.ones_like(kernel))


def test_image_and_kernel_thermal_noise_scales_are_independent() -> None:
    sampler = MCMCNLSampler(
        num_steps=1,
        lr={"image": 1.0, "kernel": 1.0},
        tau=1.0,
        lr_min_ratio={"image": 1.0, "kernel": 1.0},
        prior_solver="gaussian",
        reg_terms={"lamb": 0.0, "beta": 1.0},
        kernel_noise_scale=0.1,
        image_noise_scale=0.0,
    )
    state = {
        "image": torch.zeros(1, 3, 2, 2),
        "kernel": torch.zeros(1, 2, 2, 2),
    }
    score = {key: torch.zeros_like(value) for key, value in state.items()}
    epsilon = {key: torch.ones_like(value) for key, value in state.items()}

    updated = sampler.mc_update(
        state,
        score,
        {"image": 1.0, "kernel": 1.0},
        epsilon,
    )

    assert torch.count_nonzero(updated["image"]) == 0
    assert torch.allclose(
        updated["kernel"],
        torch.full_like(updated["kernel"], 0.1 * 2 ** 0.5),
    )
