import torch

from forward_operator import BlindBlurOperator


def test_blind_blur_gradient_includes_kernel_normalization() -> None:
    torch.manual_seed(0)
    operator = BlindBlurOperator(
        kernel_size=7, intensity=1.0, device="cpu", sigma=0.01
    )
    image = torch.randn(1, 3, 16, 16)
    # Keep the finite-difference perturbations away from the ReLU boundary.
    kernel = 0.2 + torch.randn(1, 1, 7, 7) * 0.02
    measurement = torch.randn(1, 3, 16, 16)

    gradients = operator.gradient(
        {"image": image, "kernel": kernel}, measurement
    )

    direction = torch.randn_like(kernel)
    eps = 1e-3

    def loss_at(candidate: torch.Tensor) -> torch.Tensor:
        positive = torch.relu((candidate + 1.0) / 2.0)
        normalized = positive / positive.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-12)
        return operator.loss(
            {"image": image, "kernel": normalized}, measurement
        ).sum()

    numerical = (loss_at(kernel + eps * direction) - loss_at(kernel - eps * direction)) / (2 * eps)
    analytic = (gradients["kernel"] * direction).sum()
    torch.testing.assert_close(analytic, numerical, rtol=2e-3, atol=2e-2)
