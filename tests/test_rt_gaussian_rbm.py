"""
Smoke test for RTGaussianRBM.
Key thing to verify: visible_sampling returns continuous values,
not binary 0/1 -- this is the whole point of the Gaussian extension.
Everything else (recurrence, h0, W_prime, fit, forward, reconstruct,
sample) is inherited from RTRBM and already tested there.
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
from sit_fuse_rtrbm.temporal.rt_gaussian_rbm import RTGaussianRBM
from torch.utils.data import TensorDataset
import numpy as np
import torch
import logging
logging.disable(logging.CRITICAL)


def test_visible_sampling_is_continuous():
    """The whole point of Gaussian visible units -- outputs must be
    continuous real values, not binary 0/1 like Bernoulli."""
    model = RTGaussianRBM(n_visible=8, n_hidden=6)
    h = torch.rand(4, 6)
    probs, states = model.visible_sampling(h)
    assert not torch.all((states == 0) | (states == 1)), (
        "visible_sampling returned binary values -- Gaussian override not working"
    )
    print(f"visible_sampling returns continuous values: PASS")
    print(f"  states range: [{states.min():.3f}, {states.max():.3f}]")


def test_energy_uses_quadratic_visible_term():
    """Gaussian energy uses 0.5*(v-a)^2, not -v*a like Bernoulli.
    Confirmed by checking values differ from Bernoulli RTRBM."""
    gauss_model = RTGaussianRBM(n_visible=8, n_hidden=6)
    bern_model = RTRBM(n_visible=8, n_hidden=6)

    v = torch.rand(4, 8)
    h_prev = torch.rand(4, 6)

    e_gauss = gauss_model.energy(v, h_prev)
    e_bern = bern_model.energy(v, h_prev)

    assert e_gauss.shape == (4,), f"Wrong energy shape: {e_gauss.shape}"
    assert not torch.allclose(e_gauss, e_bern), (
        "Gaussian and Bernoulli energies are identical -- "
        "quadratic term not being applied"
    )
    print("Gaussian energy differs from Bernoulli (quadratic term active): PASS")


def test_inherits_recurrence_from_rtrbm():
    """W_prime and h0 must still be present and functional."""
    model = RTGaussianRBM(n_visible=8, n_hidden=6)
    assert hasattr(model, 'W_prime'), "W_prime missing"
    assert hasattr(model, 'h0'), "h0 missing"

    v = torch.rand(4, 8)
    h_prev_a = torch.zeros(4, 6)
    h_prev_b = torch.ones(4, 6)
    probs_a, _ = model.hidden_sampling(v, h_prev_a)
    probs_b, _ = model.hidden_sampling(v, h_prev_b)
    assert not torch.allclose(probs_a, probs_b), (
        "Recurrence not working -- h_prev not influencing hidden sampling"
    )
    print("Recurrence (W_prime, h0) inherited and working: PASS")


def test_forward_returns_continuous_hidden_sequence():
    """forward() should return continuous hidden probabilities across
    the sequence -- same as RTRBM since hidden layer is unchanged."""
    model = RTGaussianRBM(n_visible=8, n_hidden=6)
    x = torch.rand(4, 5, 8)
    out = model.forward(x)
    assert out.shape == (4, 5, 6), f"Wrong shape: {out.shape}"
    assert torch.all((out >= 0) & (out <= 1)), "Hidden probs not in [0,1]"
    print(f"forward() shape {out.shape}, hidden probs in [0,1]: PASS")


def test_reconstruct_returns_continuous_visible():
    """reconstruct() visible probs should be continuous for Gaussian --
    this is the key improvement over Bernoulli RTRBM."""
    model = RTGaussianRBM(n_visible=8, n_hidden=6)
    x = torch.rand(4, 5, 8)
    targets = torch.zeros(4)
    ds = TensorDataset(x, targets)

    mse, vis_probs = model.reconstruct(ds)

    assert vis_probs.shape == (4, 5, 8)
    assert torch.isfinite(mse)
    # Key check: reconstructed visible values should NOT all be binary
    assert not torch.all((vis_probs == 0) | (vis_probs == 1)), (
        "reconstruct() returned binary values -- Gaussian visible "
        "override not flowing through to reconstruction"
    )
    print(f"reconstruct() returns continuous visibles, mse={mse:.4f}: PASS")


def test_fit_runs_on_continuous_data():
    """fit() should run cleanly on continuous-valued sequences."""
    torch.manual_seed(0)
    model = RTGaussianRBM(n_visible=8, n_hidden=6, learning_rate=0.01)
    x = torch.rand(8, 5, 8)
    targets = torch.zeros(8)
    ds = TensorDataset(x, targets)
    mse = model.fit(ds, batch_size=4, epochs=5)
    assert torch.isfinite(mse), f"fit() returned non-finite MSE: {mse}"
    print(f"fit() runs on continuous data, final mse={mse:.4f}: PASS")


if __name__ == "__main__":
    test_visible_sampling_is_continuous()
    test_energy_uses_quadratic_visible_term()
    test_inherits_recurrence_from_rtrbm()
    test_forward_returns_continuous_hidden_sequence()
    test_reconstruct_returns_continuous_visible()
    test_fit_runs_on_continuous_data()
    print("\nAll RTGaussianRBM smoke tests passed.")
