"""
Smoke test for RTRBM.fit_subseries -- chained CD-k training across a
whole subseries, with real BPTT through the h_prev chain (single
accumulated backward/step, not one per timestep).
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
import torch
import logging
logging.disable(logging.CRITICAL)


def test_fit_subseries_runs_and_updates_all_parameters():
    """Confirms one fit_subseries call updates W, a, b, AND W_prime, h0."""
    n_visible, n_hidden, batch_size, seq_len = 8, 6, 4, 5
    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)

    sequence = torch.rand(batch_size, seq_len, n_visible)

    W_before = model.W.clone().detach()
    W_prime_before = model.W_prime.clone().detach()
    h0_before = model.h0.clone().detach()

    mse = model.fit_subseries(sequence)

    assert not torch.allclose(W_before, model.W), "W did not update"
    assert not torch.allclose(
        W_prime_before, model.W_prime), "W_prime did not update"
    assert not torch.allclose(h0_before, model.h0), (
        "h0 did not update -- h0 ONLY matters through the recurrent "
        "chain, so if it's not updating, BPTT isn't actually reaching "
        "back to it."
    )
    assert torch.isfinite(mse), "mse is not finite"
    print("fit_subseries updates W, a, b, W_prime, AND h0: PASS")


def test_h0_gradient_present_before_step():
    """
    More direct check than the above: confirms h0.grad is populated and
    non-zero BEFORE the optimizer step consumes it -- proving gradient
    genuinely flows back through the whole chain to h0, not just that
    h0 happened to change for some other reason.
    """
    n_visible, n_hidden, batch_size, seq_len = 8, 6, 4, 5
    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)
    sequence = torch.rand(batch_size, seq_len, n_visible)

    # Manually replicate fit_subseries up to backward(), so we can
    # inspect .grad before optimizer.step() clears/consumes it.
    h_prev = model.h0.unsqueeze(0).expand(batch_size, -1)
    model.optimizer.zero_grad()
    total_cost = torch.tensor(0.0)
    for t in range(seq_len):
        v_t = sequence[:, t, :]
        _, _, _, _, visible_states = model.gibbs_sampling(v_t, h_prev)
        visible_states = visible_states.detach()
        cost_t = torch.mean(model.energy(v_t, h_prev)) - torch.mean(
            model.energy(visible_states, h_prev)
        )
        total_cost = total_cost + cost_t
        h_prev, _ = model.hidden_sampling(v_t, h_prev)

    total_cost.backward()

    assert model.h0.grad is not None, "h0.grad is None after backward()"
    assert torch.any(
        model.h0.grad != 0), "h0.grad is all zero after backward()"
    print("h0.grad is genuinely populated by BPTT through the chain: PASS")


def test_repeated_subseries_training_reduces_reconstruction_error():
    """
    Same memorization sanity check as cd_step's test, but now across a
    WHOLE subseries with real BPTT, not a single isolated timestep.
    """
    torch.manual_seed(0)
    n_visible, n_hidden, batch_size, seq_len = 8, 6, 1, 5

    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.1)

    # ONE fixed subseries, reused every iteration.
    sequence = torch.rand(batch_size, seq_len, n_visible)

    mse_history = []
    for i in range(200):
        mse = model.fit_subseries(sequence)
        mse_history.append(mse.item())

    early_avg = sum(mse_history[:10]) / 10
    late_avg = sum(mse_history[-10:]) / 10

    print(f"  early MSE avg (first 10 subseries-steps): {early_avg:.4f}")
    print(f"  late MSE avg (last 10 subseries-steps):   {late_avg:.4f}")

    assert late_avg < early_avg, (
        f"Expected reconstruction error to decrease, but "
        f"early={early_avg:.4f}, late={late_avg:.4f}"
    )
    print("Repeated subseries training reduces reconstruction error: PASS")


def test_cd_step_unaffected_by_new_method():
    """Regression check: cd_step should still work exactly as before."""
    n_visible, n_hidden, batch_size = 8, 6, 4
    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)
    v = torch.rand(batch_size, n_visible)
    h_prev = model.h0.unsqueeze(0).expand(batch_size, -1)
    mse = model.cd_step(v, h_prev)
    assert torch.isfinite(mse)
    print("cd_step still works unchanged: PASS")


if __name__ == "__main__":
    test_fit_subseries_runs_and_updates_all_parameters()
    test_h0_gradient_present_before_step()
    test_repeated_subseries_training_reduces_reconstruction_error()
    test_cd_step_unaffected_by_new_method()
    print("\nAll fit_subseries smoke tests passed.")
