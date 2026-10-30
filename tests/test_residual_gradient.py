from __future__ import annotations

import torch


def test_data_residual_jacobian_matches_energy_gradient() -> None:
    torch.manual_seed(7)
    x = torch.randn(3, 4, requires_grad=True)
    matrix = torch.randn(5, 4)
    observation = torch.randn(3, 5)
    beta = 0.7

    residual = (x @ matrix.T - observation) / (2**0.5 * beta)
    energy = 0.5 * residual.square().sum()
    jacobian_transpose_residual = (
        (x @ matrix.T - observation) @ matrix / (2 * beta**2)
    )
    autograd_gradient = torch.autograd.grad(energy, x)[0]

    assert torch.allclose(
        jacobian_transpose_residual, autograd_gradient, atol=1e-6, rtol=1e-6
    )


def test_signed_prior_residual_gradient_and_descent() -> None:
    torch.manual_seed(11)
    target = torch.randn(6)
    x = torch.randn(6, requires_grad=True)
    weight = 0.3

    residual = weight**0.5 * (x - target)
    energy = 0.5 * residual.square().sum()
    residual_direction = weight * (x - target)
    autograd_gradient = torch.autograd.grad(energy, x)[0]
    assert torch.allclose(residual_direction, autograd_gradient, atol=1e-6)

    with torch.no_grad():
        updated = x - 0.05 * residual_direction
        updated_energy = 0.5 * (weight**0.5 * (updated - target)).square().sum()
    assert updated_energy < energy
