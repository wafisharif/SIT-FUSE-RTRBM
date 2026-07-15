"""
Smoke test for RTRBM.sample().
Verifies the paper's sampling procedure generates valid sequences
from scratch with no input data required.
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
import torch
import logging
logging.disable(logging.CRITICAL)


def test_sample_output_shape():
    """sample() must return (n_samples, n_steps, n_visible)."""
    model = RTRBM(n_visible=8, n_hidden=6)
    samples = model.sample(n_samples=4, n_steps=10)
    assert samples.shape == (4, 10, 8), f"Wrong shape: {samples.shape}"
    print(f"sample() output shape {samples.shape}: PASS")


def test_sample_values_are_binary():
    """Bernoulli visible units -- sampled values must be 0 or 1."""
    model = RTRBM(n_visible=8, n_hidden=6)
    samples = model.sample(n_samples=4, n_steps=10)
    assert torch.all((samples == 0) | (samples == 1)), (
        "sample() returned non-binary values -- visible_sampling not working"
    )
    print("sample() values are binary (0 or 1): PASS")


def test_sample_generates_diverse_sequences():
    """Different samples in the same batch should not all be identical."""
    model = RTRBM(n_visible=8, n_hidden=6)
    samples = model.sample(n_samples=4, n_steps=10)
    all_same = torch.all(samples[0] == samples[1])
    assert not all_same, (
        "All generated samples identical -- model may be degenerate"
    )
    print("sample() generates diverse sequences across batch: PASS")


def test_sample_is_stochastic():
    """Two separate calls should produce different results."""
    model = RTRBM(n_visible=8, n_hidden=6)
    samples1 = model.sample(n_samples=4, n_steps=10)
    samples2 = model.sample(n_samples=4, n_steps=10)
    assert not torch.all(samples1 == samples2), (
        "Two calls produced identical results -- sampling not stochastic"
    )
    print("sample() is stochastic across calls: PASS")


def test_sample_requires_no_input_data():
    """sample() takes only n_samples and n_steps -- no dataset needed."""
    model = RTRBM(n_visible=8, n_hidden=6)
    try:
        samples = model.sample(n_samples=2, n_steps=5)
        assert samples.shape == (2, 5, 8)
        print("sample() works with no input data (generation only): PASS")
    except TypeError as e:
        raise AssertionError(f"sample() requires unexpected arguments: {e}")


if __name__ == "__main__":
    test_sample_output_shape()
    test_sample_values_are_binary()
    test_sample_generates_diverse_sequences()
    test_sample_is_stochastic()
    test_sample_requires_no_input_data()
    print("\nAll sample() smoke tests passed.")
