"""
Forward-pass-only smoke test for RTRBM, run across a FULL toy sequence
(not a single isolated timestep). No training/backward() happens here --
this only confirms the recurrence loop itself runs correctly end-to-end:
h_prev correctly carries forward from one timestep to the next, t=0
correctly uses the learned h0, and shapes stay correct across the whole
sequence.

Corresponds to Work Plan Weeks 4-8 deliverable: "successful runs of
RTRBM... forward pass... on toy sequences."
"""
import numpy as np
import torch

from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset


def build_toy_dataset(n_timesteps=30, n_features=8, seq_len=10):
    """
    Toy data: smooth-ish synthetic signal (sum of a couple sine waves +
    small noise) per feature, so it's a bit more realistic than pure
    np.arange ramps used in earlier tests, but still small and fast.
    """
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    data = np.zeros((n_timesteps, n_features), dtype=np.float32)
    rng = np.random.default_rng(42)
    for f in range(n_features):
        freq = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, np.pi)
        data[:, f] = np.sin(freq * t + phase) + 0.05 * rng.standard_normal(n_timesteps)

    targets = np.arange(n_timesteps)

    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=seq_len, do_shuffle=False)
    return ds


def run_forward_pass_on_sequence(model, sequence):
    """
    Chains hidden_sampling across every timestep in ONE sequence.

    :param model: an RTRBM instance.
    :param sequence: tensor, shape (seq_len, n_visible) -- ONE sequence,
        not yet batched.
    :return: list of hidden probability tensors, one per timestep, each
        shape (n_hidden,).
    """
    seq_len = sequence.shape[0]

    # t=0: no real h_prev exists yet, so we use the model's learned
    # initial hidden state h0. Note h0 has shape (n_hidden,) and v at a
    # single timestep has shape (n_visible,) -- hidden_sampling expects a
    # BATCH dimension, so we unsqueeze both to (1, n_visible)/(1, n_hidden)
    # and squeeze back down afterward, for a batch size of 1.
    h_prev = model.h0.unsqueeze(0)  # shape (1, n_hidden)

    all_probs = []

    for t in range(seq_len):
        v_t = sequence[t].unsqueeze(0)  # shape (1, n_visible)

        probs, states = model.hidden_sampling(v_t, h_prev)

        all_probs.append(probs.squeeze(0))  # back to (n_hidden,)

        # THE key recurrence step: this timestep's mean-field PROBABILITY
        # (not the sampled binary state) becomes next timestep's h_prev.
        # This is deliberate -- see Sutskever/Hinton/Taylor 2008, and the
        # note in rtrbm.py's hidden_sampling docstring.
        h_prev = probs

    return all_probs


def test_forward_pass_runs_without_shape_errors():
    n_features, seq_len, n_hidden = 8, 10, 6

    ds = build_toy_dataset(n_features=n_features, seq_len=seq_len)
    model = RTRBM(n_visible=n_features, n_hidden=n_hidden)

    sequence, target = ds[0]
    sequence = torch.from_numpy(sequence)  # shape (seq_len, n_features)

    all_probs = run_forward_pass_on_sequence(model, sequence)

    assert len(all_probs) == seq_len, (
        f"Expected {seq_len} timestep outputs, got {len(all_probs)}"
    )
    for t, probs in enumerate(all_probs):
        assert probs.shape == (n_hidden,), (
            f"Timestep {t}: expected shape ({n_hidden},), got {probs.shape}"
        )

    print("Forward pass runs across full sequence, shapes correct: PASS")


def test_first_timestep_uses_h0():
    """
    Confirms t=0 actually used self.h0 as h_prev, not some other default
    (e.g. zeros), by manually recomputing what t=0's output SHOULD be and
    comparing.
    """
    n_features, seq_len, n_hidden = 8, 10, 6

    ds = build_toy_dataset(n_features=n_features, seq_len=seq_len)
    model = RTRBM(n_visible=n_features, n_hidden=n_hidden)

    sequence, target = ds[0]
    sequence = torch.from_numpy(sequence)

    all_probs = run_forward_pass_on_sequence(model, sequence)

    # Manually recompute t=0's expected output using h0 directly.
    v_0 = sequence[0].unsqueeze(0)
    h0_batched = model.h0.unsqueeze(0)
    expected_probs_0, _ = model.hidden_sampling(v_0, h0_batched)

    assert torch.allclose(all_probs[0], expected_probs_0.squeeze(0)), (
        "Timestep 0's output does not match manually recomputing with h0 "
        "-- the loop is not correctly using the learned initial hidden "
        "state at the start of a sequence."
    )

    print("First timestep correctly uses learned h0: PASS")


def test_hidden_probs_actually_change_across_sequence():
    """
    If the recurrence were silently broken (e.g. h_prev gets overwritten
    with something constant each step), every timestep's output would end
    up identical despite different v_t inputs feeding in -- this check
    would catch that.
    """
    n_features, seq_len, n_hidden = 8, 10, 6

    ds = build_toy_dataset(n_features=n_features, seq_len=seq_len)
    model = RTRBM(n_visible=n_features, n_hidden=n_hidden)

    sequence, target = ds[0]
    sequence = torch.from_numpy(sequence)

    all_probs = run_forward_pass_on_sequence(model, sequence)

    # Not every pair needs to differ, but they can't ALL be identical.
    all_same = all(
        torch.allclose(all_probs[0], p) for p in all_probs[1:]
    )
    assert not all_same, (
        "All timesteps produced IDENTICAL hidden probabilities -- "
        "suspicious, suggests the recurrence loop isn't actually wiring "
        "v_t through correctly."
    )

    print("Hidden probabilities vary across the sequence (not stuck): PASS")


def test_forward_pass_works_on_batch_of_sequences():
    """
    Same forward pass, but using ALL sequences from the dataset at once
    via batched hidden_sampling calls -- closer to how this will actually
    run during real training (whole batches, not one sequence at a time).
    """
    n_features, seq_len, n_hidden = 8, 10, 6

    ds = build_toy_dataset(n_features=n_features, seq_len=seq_len)
    model = RTRBM(n_visible=n_features, n_hidden=n_hidden)

    # Stack every sequence in the toy dataset into one batch:
    # shape (batch, seq_len, n_features)
    batch = torch.stack([torch.from_numpy(ds[i][0]) for i in range(len(ds))])
    batch_size = batch.shape[0]

    h_prev = model.h0.unsqueeze(0).expand(batch_size, -1)  # (batch, n_hidden)

    for t in range(seq_len):
        v_t = batch[:, t, :]  # (batch, n_features)
        probs, states = model.hidden_sampling(v_t, h_prev)

        assert probs.shape == (batch_size, n_hidden), (
            f"Timestep {t}: expected ({batch_size}, {n_hidden}), got {probs.shape}"
        )

        h_prev = probs

    print("Batched forward pass across full sequence: PASS")


if __name__ == "__main__":
    test_forward_pass_runs_without_shape_errors()
    test_first_timestep_uses_h0()
    test_hidden_probs_actually_change_across_sequence()
    test_forward_pass_works_on_batch_of_sequences()
    print("\nAll forward-pass smoke tests passed.")