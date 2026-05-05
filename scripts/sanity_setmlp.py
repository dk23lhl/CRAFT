"""SetMLPFusion architecture sanity check — all 4 checks must pass before training.

Usage (from repo root):
    python scripts/sanity_setmlp.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.fusion import SetMLPFusion


def test_permutation_invariance():
    """phi shared + mean pool => permuting views does not change the output."""
    torch.manual_seed(0)
    fusion = SetMLPFusion(embed_dim=64, num_views=6, hidden_dim=128).eval()
    x_list = [torch.randn(4, 64) for _ in range(6)]
    m = torch.ones(4, 6)

    perm = torch.randperm(6).tolist()
    with torch.no_grad():
        y1 = fusion(x_list, m)
        y2 = fusion([x_list[i] for i in perm], m[:, perm])
    diff = (y1 - y2).abs().max().item()
    assert diff < 1e-5, f"permutation invariance broken (max diff={diff:.2e})"
    print(f"[OK] permutation invariance (max diff={diff:.2e})")


def test_variable_length():
    """|O|=1 does not blow up or divide by zero; |O|=V also works."""
    torch.manual_seed(0)
    fusion = SetMLPFusion(embed_dim=64, num_views=6, hidden_dim=128).eval()
    x_list = [torch.randn(4, 64) for _ in range(6)]

    m_full = torch.ones(4, 6)
    m_one = torch.zeros(4, 6); m_one[:, 0] = 1.0
    m_half = torch.tensor([[1, 0, 1, 0, 1, 0]] * 4, dtype=torch.float32)

    with torch.no_grad():
        y_full = fusion(x_list, m_full)
        y_one = fusion(x_list, m_one)
        y_half = fusion(x_list, m_half)
    for name, y in [('full', y_full), ('one', y_one), ('half', y_half)]:
        assert torch.isfinite(y).all(), f"non-finite output at {name}"
    print("[OK] variable length |O| in {1, 3, 6} all finite")


def test_gradient_isolation():
    """Masked view positions must not receive gradients (C2's 'excluded from computation')."""
    torch.manual_seed(0)
    fusion = SetMLPFusion(embed_dim=64, num_views=4, hidden_dim=128).train()

    x_stack = torch.randn(2, 4, 64, requires_grad=True)
    x_split = [x_stack[:, v] for v in range(4)]
    m = torch.tensor([[1, 1, 0, 0],
                      [1, 0, 1, 0]], dtype=torch.float32)

    out = fusion(x_split, m).sum()
    out.backward()
    masked_grad = x_stack.grad * (1 - m).unsqueeze(-1)
    leak = masked_grad.abs().max().item()
    assert leak < 1e-7, f"gradient leaks into masked positions (max={leak:.2e})"
    print(f"[OK] gradient isolation (masked grad max={leak:.2e})")


def test_manual_match():
    """forward(x, ones) == norm(rho(mean(phi(x)))); verify manual computation matches forward."""
    torch.manual_seed(0)
    fusion = SetMLPFusion(embed_dim=64, num_views=6, hidden_dim=128).eval()
    x_list = [torch.randn(4, 64) for _ in range(6)]

    with torch.no_grad():
        y_auto = fusion(x_list, view_mask=None)
        tokens = torch.stack(x_list, dim=1)
        phi_out = fusion.phi(tokens)
        pooled = phi_out.mean(dim=1)
        y_manual = fusion.norm(fusion.rho(pooled))
    diff = (y_auto - y_manual).abs().max().item()
    assert diff < 1e-6, f"forward != manual (max diff={diff:.2e})"
    print(f"[OK] forward matches manual (max diff={diff:.2e})")


if __name__ == '__main__':
    test_permutation_invariance()
    test_variable_length()
    test_gradient_isolation()
    test_manual_match()
    print("\nAll 4 sanity checks passed. Safe to start training.")
