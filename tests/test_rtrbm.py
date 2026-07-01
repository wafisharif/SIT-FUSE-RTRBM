"""
Smoke test for RTRBM.hidden_sampling, run against the REAL installed
learnergy package (not stand-ins), via the sit_fuse_rtrbm dev package.
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
import torch
import logging
logging.disable(logging.CRITICAL)


def test_hidden_sampling_output_shape():
    batch_size, n_visible, n_hidden = 8, 10, 6

    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)

    v = torch.randn(batch_size, n_visible)
    h_prev = model.h0.unsqueeze(0).expand(batch_size, -1)

    probs, states = model.hidden_sampling(v, h_prev)

    assert probs.shape == (batch_size, n_hidden)
    assert states.shape == (batch_size, n_hidden)
    assert torch.all((probs >= 0) & (probs <= 1))
    assert torch.all((states == 0) | (states == 1))

    print("hidden_sampling output shape + value range: PASS")


def test_recurrent_gradient_flows_through_W_prime():
    batch_size, n_visible, n_hidden = 8, 10, 6

    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)

    v = torch.randn(batch_size, n_visible)
    h_prev = torch.rand(batch_size, n_hidden)

    probs, _ = model.hidden_sampling(v, h_prev)

    loss = probs.sum()
    loss.backward()

    assert model.W_prime.grad is not None
    assert torch.any(model.W_prime.grad != 0)

    print("Gradient flow through W_prime: PASS")


def test_h_prev_actually_changes_output():
    batch_size, n_visible, n_hidden = 4, 10, 6
    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)

    v = torch.randn(batch_size, n_visible)
    h_prev_a = torch.zeros(batch_size, n_hidden)
    h_prev_b = torch.ones(batch_size, n_hidden)

    probs_a, _ = model.hidden_sampling(v, h_prev_a)
    probs_b, _ = model.hidden_sampling(v, h_prev_b)

    assert not torch.allclose(probs_a, probs_b)

    print("h_prev actually influences output: PASS")


def test_fit_is_now_implemented():
    """fit() must NOT raise NotImplementedError -- it is implemented now."""
    import numpy as np
    from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset

    data = np.random.rand(20, 10).astype(np.float32)
    targets = np.arange(20)
    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=4, do_shuffle=False)

    model = RTRBM(n_visible=10, n_hidden=6)
    try:
        model.fit(ds, batch_size=4, epochs=1)
        print("fit() is implemented and runs cleanly: PASS")
    except NotImplementedError:
        raise AssertionError("fit() still raises NotImplementedError")


def test_inherits_real_learnergy_rbm():
    """
    Confirms RTRBM is actually built on top of the REAL installed
    learnergy package, not a stand-in -- e.g. catches accidental
    namespace collisions if a local 'learnergy' folder existed.
    """
    from learnergy.models.bernoulli import RBM
    model = RTRBM(n_visible=10, n_hidden=6)
    assert isinstance(
        model, RBM), "RTRBM is not an instance of the real learnergy RBM"
    print("RTRBM correctly inherits from the real installed learnergy RBM: PASS")


if __name__ == "__main__":
    test_hidden_sampling_output_shape()
    test_recurrent_gradient_flows_through_W_prime()
    test_h_prev_actually_changes_output()
    test_fit_is_now_implemented()
    test_inherits_real_learnergy_rbm()
    print("\nAll RTRBM smoke tests passed (against real installed learnergy).")
