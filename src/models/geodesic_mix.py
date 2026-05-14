"""Geodesic mixing on the unit hypersphere.

Ported from Oh et al., NeurIPS 2023 (github.com/changdaeoh/multimodal-mixup).
Convention: ``sph_inter(a, b, s)`` returns ``a`` at ``s=1`` and ``b`` at ``s=0``.
"""

import torch


def sph_inter(a: torch.Tensor, b: torch.Tensor, s, eps: float = 1e-6) -> torch.Tensor:
    """Slerp between L2-normalized rows of ``a`` and ``b``.

    ``s`` may be a scalar, a length-1 tensor, a length-B tensor, or a [B, 1]
    tensor. ``s=1`` returns ``a``, ``s=0`` returns ``b``.
    """
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    if not isinstance(s, torch.Tensor):
        s = torch.tensor(float(s), device=a.device, dtype=a.dtype)
    else:
        s = s.to(device=a.device, dtype=a.dtype)

    if s.dim() == 0:
        s = s.view(1, 1).expand(a.shape[0], 1)
    elif s.dim() == 1:
        if s.shape[0] == 1:
            s = s.view(1, 1).expand(a.shape[0], 1)
        else:
            s = s.view(-1, 1)

    w_a = torch.sin(s * theta) / (sin_theta + eps)
    w_b = torch.sin((1.0 - s) * theta) / (sin_theta + eps)
    mixed = w_a * a + w_b * b

    # Near-parallel rows would divide by ~0; fall back to a normalized lerp.
    near_parallel = (sin_theta.abs() < eps).expand_as(mixed)
    lerp = s * a + (1.0 - s) * b
    lerp = lerp / (lerp.norm(dim=-1, keepdim=True) + eps)
    mixed = torch.where(near_parallel, lerp, mixed)

    return mixed / (mixed.norm(dim=-1, keepdim=True) + eps)


def sample_beta_lambda(
    alpha: float,
    beta: float,
    device,
    per_sample: bool,
    B: int = 1,
) -> torch.Tensor:
    """Draw ``lambda ~ Beta(alpha, beta)`` as shape ``[B]`` or ``[1]``."""
    alpha_t = torch.tensor(float(alpha), device=device, dtype=torch.float32)
    beta_t = torch.tensor(float(beta), device=device, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    shape = (B,) if per_sample else (1,)
    return dist.sample(shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    D, B = 16, 4

    def _rand_unit(shape):
        x = torch.randn(*shape)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    a = _rand_unit((B, D))
    b = _rand_unit((B, D))

    assert torch.allclose(sph_inter(a, b, 1.0), a, atol=1e-4)
    assert torch.allclose(sph_inter(a, b, 0.0), b, atol=1e-4)

    m = sph_inter(a, b, torch.rand(B))
    assert torch.all((m.norm(dim=-1) - 1.0).abs() < 1e-4)

    assert torch.allclose(sph_inter(a, a, 0.37), a, atol=1e-3)

    assert sample_beta_lambda(0.2, 0.2, "cpu", per_sample=False, B=B).shape == (1,)
    assert sample_beta_lambda(0.2, 0.2, "cpu", per_sample=True, B=B).shape == (B,)

    print("geodesic_mix.py: all asserts passed.")
