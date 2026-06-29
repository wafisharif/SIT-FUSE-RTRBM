"""
Smoke test for RTRBM.cd_step -- ONE timestep's worth of CD-k training.
The classic sanity check: repeatedly training on the SAME single example
should make reconstruction error (MSE) trend downward over iterations.
If it doesn't, something in energy()/gibbs_sampling()/hidden_sampling()
is wired together inconsistently.
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
import torch
import logging
logging.disable(logging.CRITICAL)


def test_energy_accounts_for_recurrent_bias():
    """
    Confirms energy() actually responds to h_prev (i.e. is NOT silently
    using the old parent-class behavior that ignores the recurrent term).
    """
    n_visible, n_hidden, batch_size = 8, 6, 4
    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)

    v = torch.rand(batch_size, n_visible)
    h_prev_a = torch.zeros(batch_size, n_hidden)
    h_prev_b = torch.ones(batch_size, n_hidden)

    energy_a = model.energy(v, h_prev_a)
    energy_b = model.energy(v, h_prev_b)

    assert not torch.allclose(energy_a, energy_b), (
        "energy() did not change when h_prev changed -- it's not "
        "accounting for the recurrent bias term."
    )
    print("energy() correctly responds to h_prev: PASS")


def test_cd_step_runs_and_updates_parameters():
    """Confirms one cd_step call actually changes the model's weights."""
    n_visible, n_hidden, batch_size = 8, 6, 4
    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)

    v = torch.rand(batch_size, n_visible)
    h_prev = model.h0.unsqueeze(0).expand(batch_size, -1)

    W_before = model.W.clone().detach()

    mse = model.cd_step(v, h_prev)

    assert not torch.allclose(W_before, model.W), (
        "cd_step ran but W did not change at all -- optimizer step had no effect."
    )
    assert torch.isfinite(mse), "MSE is not finite (NaN/Inf)"
    print("cd_step runs and actually updates parameters: PASS")


def test_w_prime_gradient_present_during_cd_step():
    """Confirms W_prime specifically gets a gradient during cd_step, not
    just W/a/b -- this is the whole point of overriding energy()."""
    n_visible, n_hidden, batch_size = 8, 6, 4
    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden)

    v = torch.rand(batch_size, n_visible)
    h_prev = torch.rand(batch_size, n_hidden)  # non-trivial, not h0/zeros

    W_prime_before = model.W_prime.clone().detach()

    model.cd_step(v, h_prev)

    assert not torch.allclose(W_prime_before, model.W_prime), (
        "W_prime did not change after cd_step -- the recurrent weights "
        "are not actually being trained."
    )
    print("W_prime is actually updated by cd_step: PASS")


def test_repeated_training_on_one_example_reduces_reconstruction_error():
    """
    THE key sanity check: hammering the model with the SAME single
    example over and over should make its reconstruction of that example
    get BETTER (lower MSE) over time. This is the standard way to sanity
    check an energy-based model's training mechanics before trusting it
    on real varied data.
    """
    torch.manual_seed(0)
    n_visible, n_hidden, batch_size = 8, 6, 1

    model = RTRBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.1)

    # ONE fixed example, reused every iteration.
    v = torch.rand(batch_size, n_visible)
    h_prev = model.h0.unsqueeze(0).expand(batch_size, -1)

    mse_history = []
    for i in range(200):
        mse = model.cd_step(v, h_prev)
        mse_history.append(mse.item())

    early_avg = sum(mse_history[:10]) / 10
    late_avg = sum(mse_history[-10:]) / 10

    print(f"  early MSE avg (first 10 steps): {early_avg:.4f}")
    print(f"  late MSE avg (last 10 steps):   {late_avg:.4f}")

    assert late_avg < early_avg, (
        f"Expected reconstruction error to decrease with repeated training "
        f"on one example, but early={early_avg:.4f}, late={late_avg:.4f}"
    )
    print("Repeated training on one example reduces reconstruction error: PASS")


if __name__ == "__main__":
    test_energy_accounts_for_recurrent_bias()
    test_cd_step_runs_and_updates_parameters()
    test_w_prime_gradient_present_during_cd_step()
    test_repeated_training_on_one_example_reduces_reconstruction_error()
    print("\nAll cd_step smoke tests passed.")
