from __future__ import annotations

import torch

from forward_operator.blind_blur_utils import fft_blur, normalize_kernel
from sampler import BlindDAPS, Nonlinear_DAPS


def test_kernel_normalization() -> None:
    kernel = torch.rand(2, 1, 33, 33)
    normalized = normalize_kernel(kernel)
    assert normalized.shape == kernel.shape
    assert torch.all(normalized >= 0)
    assert torch.allclose(normalized.sum((-2, -1)), torch.ones(2, 1), atol=1e-6)


def test_linear_operator_differentiable() -> None:
    image = torch.rand(2, 3, 64, 64, requires_grad=True)
    kernel = normalize_kernel(torch.rand(2, 1, 33, 33))
    blurred = fft_blur(image, kernel)
    assert blurred.shape == image.shape
    assert torch.isfinite(blurred).all()
    blurred.mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_reported_sampler_classes_exist() -> None:
    assert BlindDAPS.__name__ == "BlindDAPS"
    assert Nonlinear_DAPS.__name__ == "Nonlinear_DAPS"
