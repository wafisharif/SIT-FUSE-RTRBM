"""
Smoke test for RTRBM.forward() and RTRBM.reconstruct().
Both methods mirror their RBM counterparts in learnergy's rbm.py,
adapted for temporal sequences (batch, seq_len, n_visible).
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
from torch.utils.data import TensorDataset
import numpy as np
import torch
import logging
logging.disable(logging.CRITICAL)


def build_toy_dataset(batch=8, seq_len=5, n_visible=8):
    x = torch.rand(batch, seq_len, n_visible)
    targets = torch.zeros(batch)
    return TensorDataset(x, targets), x


def test_forward_output_shape():
    """forward() must return (batch, seq_len, n_hidden)."""
    model = RTRBM(n_visible=8, n_hidden=6)
    ds, x = build_toy_dataset()

    out = model.forward(x)

    assert out.shape == (8, 5, 6), (
        f"Expected (8, 5, 6), got {out.shape}"
    )
    assert torch.all((out >= 0) & (out <= 1)), "forward() outputs not in [0,1]"
    print(f"forward() output shape {out.shape}: PASS")


def test_forward_varies_across_timesteps():
    """Hidden states should differ across timesteps due to recurrence."""
    model = RTRBM(n_visible=8, n_hidden=6)
    _, x = build_toy_dataset()

    out = model.forward(x)

    # Not all timesteps identical -- recurrence must be doing something
    all_same = all(torch.allclose(
        out[:, 0, :], out[:, t, :]) for t in range(1, 5))
    assert not all_same, "All timesteps identical -- recurrence not working in forward()"
    print("forward() varies across timesteps (recurrence active): PASS")


def test_reconstruct_output_shape():
    """reconstruct() must return (mse, visible_probs) where
    visible_probs shape is (batch, seq_len, n_visible)."""
    model = RTRBM(n_visible=8, n_hidden=6)
    ds, x = build_toy_dataset()

    mse, vis_probs = model.reconstruct(ds)

    assert vis_probs.shape == (8, 5, 8), (
        f"Expected visible_probs shape (8, 5, 8), got {vis_probs.shape}"
    )
    assert torch.isfinite(mse), f"MSE not finite: {mse}"
    print(f"reconstruct() shape {vis_probs.shape}, mse={mse:.4f}: PASS")


def test_reconstruct_output_is_valid_probabilities():
    """reconstruct() visible_probs must be in [0,1] since they are
    P(v|h) values, not binary samples."""
    model = RTRBM(n_visible=8, n_hidden=6)
    ds, x = build_toy_dataset()

    mse, vis_probs = model.reconstruct(ds)

    assert torch.all((vis_probs >= 0) & (vis_probs <= 1)), (
        "visible_probs from reconstruct() are not valid probabilities"
    )
    print(f"reconstruct() returns valid probabilities in [0,1]: PASS")


if __name__ == "__main__":
    test_forward_output_shape()
    test_forward_varies_across_timesteps()
    test_reconstruct_output_shape()
    test_reconstruct_output_is_valid_probabilities()
    print("\nAll reconstruct/forward smoke tests passed.")
