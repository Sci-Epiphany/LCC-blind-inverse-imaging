from __future__ import annotations

import torch
import pytest

from cores.mcmc import regularization_step


def test_pnp_regularizer_invokes_denoiser() -> None:
    called = False

    def denoiser(batch: torch.Tensor) -> torch.Tensor:
        nonlocal called
        called = True
        assert batch.shape[1] == 2
        return batch[:, :1]

    kernel = torch.rand(2, 1, 4, 4)
    result = regularization_step(
        kernel,
        lamb=20.0,
        beta=2.0e6,
        denoiser=denoiser,
        how="pnp",
    )
    assert called
    assert result.shape == kernel.shape


def test_unknown_regularizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        regularization_step(
            torch.rand(1, 1, 4, 4),
            lamb=20.0,
            beta=2.0e6,
            how="unknown",
        )


def test_pnp_requires_denoiser() -> None:
    with pytest.raises(ValueError, match="requires"):
        regularization_step(
            torch.rand(1, 1, 4, 4),
            lamb=20.0,
            beta=2.0e6,
            how="pnp",
        )
