from __future__ import annotations

import torch

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
