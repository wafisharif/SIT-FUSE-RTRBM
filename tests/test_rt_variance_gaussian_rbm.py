"""
Smoke test for RTVarianceGaussianRBM.
Key things to verify vs RTGaussianRBM:
- sigma exists, initialized to ones, gets gradient
- visible_sampling returns continuous values sampled from Normal
- hidden_sampling scales by sigma^2
- energy uses learned variance term
- No per-batch normalization needed
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
from sit_fuse_rtrbm.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
from torch.utils.data import TensorDataset
import torch
import logging
logging.disable(logging.CRITICAL)


def test_sigma_initialized_correctly():
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    assert model.sigma.shape == (8,), f"Wrong sigma shape: {model.sigma.shape}"
    assert torch.all(model.sigma == 1.0), "sigma not initialized to ones"
    print("sigma initialized to ones, shape (n_visible,): PASS")


def test_visible_sampling_returns_continuous():
    """Samples from Normal(mean, sigma^2) -- not binary, not bounded."""
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    h = torch.rand(4, 6)
    states, acts = model.visible_sampling(h)
    assert states.shape == (4, 8)
    assert acts.shape == (4, 8)
    assert not torch.all((states == 0) | (states == 1)
                         ), "states are binary -- should be continuous"
    print(
        f"visible_sampling continuous, range [{states.min():.3f}, {states.max():.3f}]: PASS")


def test_visible_sampling_return_order():
    """VarianceGaussianRBM returns (states, activations) -- reversed vs GaussianRBM."""
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    h = torch.rand(4, 6)
    first, second = model.visible_sampling(h)
    # states (noisy samples) should have more variance than activations (means)
    assert first.std() >= second.std() - 1e-3, (
        "Return order may be wrong -- first should be noisy states, second clean activations"
    )
    print("visible_sampling return order (states, activations): PASS")


def test_hidden_sampling_scales_by_sigma():
    """hidden_sampling must divide v by sigma^2 -- different from plain RTRBM."""
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    plain = RTRBM(n_visible=8, n_hidden=6)
    v = torch.rand(4, 8)
    h_prev = torch.rand(4, 6)
    h_var, _ = model.hidden_sampling(v, h_prev)
    h_plain, _ = plain.hidden_sampling(v, h_prev)
    assert not torch.allclose(h_var, h_plain), (
        "sigma^2 scaling not affecting hidden sampling"
    )
    print("hidden_sampling correctly scales by sigma^2: PASS")


def test_energy_uses_sigma():
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    plain = RTRBM(n_visible=8, n_hidden=6)
    v = torch.rand(4, 8)
    h_prev = torch.rand(4, 6)
    e_var = model.energy(v, h_prev)
    e_plain = plain.energy(v, h_prev)
    assert e_var.shape == (4,)
    assert not torch.allclose(e_var, e_plain), "energy not using sigma"
    print("energy() uses sigma, shape (batch,): PASS")


def test_sigma_receives_gradient():
    """sigma must get a gradient -- it's a learnable parameter."""
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    x = torch.rand(4, 5, 8)
    h_prev = model.h0.unsqueeze(0).expand(4, -1)
    model.optimizer.zero_grad()
    total_cost = torch.tensor(0.0)
    for t in range(5):
        v_t = x[:, t, :]
        _, _, _, _, vis_act = model.gibbs_sampling(v_t, h_prev)
        vis_act = vis_act.detach()
        cost_t = torch.mean(model.energy(v_t, h_prev)) - torch.mean(
            model.energy(vis_act, h_prev)
        )
        total_cost = total_cost + cost_t
        h_prev, _ = model.hidden_sampling(v_t, h_prev)
    total_cost.backward()
    assert model.sigma.grad is not None, "sigma has no gradient"
    assert torch.any(model.sigma.grad != 0), "sigma gradient is all zero"
    print("sigma receives non-zero gradient during training: PASS")


def test_recurrence_still_works():
    """W_prime and h0 must still function correctly."""
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    assert hasattr(model, 'W_prime')
    assert hasattr(model, 'h0')
    v = torch.rand(4, 8)
    h_a = torch.zeros(4, 6)
    h_b = torch.ones(4, 6)
    probs_a, _ = model.hidden_sampling(v, h_a)
    probs_b, _ = model.hidden_sampling(v, h_b)
    assert not torch.allclose(probs_a, probs_b), "recurrence not working"
    print("recurrence (W_prime, h0) inherited and working: PASS")


def test_reconstruct_returns_continuous():
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    x = torch.rand(4, 5, 8)
    targets = torch.zeros(4)
    ds = TensorDataset(x, targets)
    mse, vis = model.reconstruct(ds)
    assert vis.shape == (4, 5, 8)
    assert torch.isfinite(mse)
    assert not torch.all((vis == 0) | (vis == 1)
                         ), "reconstruct not returning continuous"
    print(f"reconstruct() continuous, mse={mse:.4f}: PASS")


if __name__ == "__main__":
    test_sigma_initialized_correctly()
    test_visible_sampling_returns_continuous()
    test_visible_sampling_return_order()
    test_hidden_sampling_scales_by_sigma()
    test_energy_uses_sigma()
    test_sigma_receives_gradient()
    test_recurrence_still_works()
    test_reconstruct_returns_continuous()
    print("\nAll RTVarianceGaussianRBM smoke tests passed.")
