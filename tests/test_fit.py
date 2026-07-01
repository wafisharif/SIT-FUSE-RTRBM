"""
Smoke test for RTRBM.fit() -- the full outer training loop.
Wires together SFTemporalDataset + RTRBM.fit() end to end for the
first time, on a small toy dataset.
"""
from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
import torch
import numpy as np
import logging
logging.disable(logging.CRITICAL)


def build_toy_dataset(n_timesteps=50, n_features=8, seq_len=5):
    """Small synthetic dataset -- sine waves + noise."""
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    rng = np.random.default_rng(42)
    data = np.zeros((n_timesteps, n_features), dtype=np.float32)
    for f in range(n_features):
        freq = rng.uniform(0.5, 2.0)
        data[:, f] = np.sin(freq * t) + 0.05 * rng.standard_normal(n_timesteps)
    targets = np.arange(n_timesteps)
    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=seq_len, do_shuffle=True)
    return ds


def test_fit_runs_and_returns_finite_mse():
    """fit() should run without crashing and return a finite MSE."""
    ds = build_toy_dataset()
    model = RTRBM(n_visible=8, n_hidden=6)

    mse = model.fit(ds, batch_size=8, epochs=2)

    assert torch.isfinite(mse), f"fit() returned non-finite MSE: {mse}"
    print(f"fit() ran cleanly, final MSE: {mse:.4f}: PASS")


def test_fit_raises_no_longer():
    """fit() must NOT raise NotImplementedError anymore."""
    ds = build_toy_dataset()
    model = RTRBM(n_visible=8, n_hidden=6)
    try:
        model.fit(ds, batch_size=8, epochs=1)
        print("fit() no longer raises NotImplementedError: PASS")
    except NotImplementedError:
        raise AssertionError(
            "fit() still raises NotImplementedError -- not implemented yet")


def test_fit_populates_history():
    """self.dump() should write mse and time into model.history each epoch."""
    ds = build_toy_dataset()
    model = RTRBM(n_visible=8, n_hidden=6)

    model.fit(ds, batch_size=8, epochs=3)

    assert "mse" in model.history, "model.history has no 'mse' key"
    assert "time" in model.history, "model.history has no 'time' key"
    assert len(model.history["mse"]) == 3, (
        f"Expected 3 mse entries (one per epoch), got {len(model.history['mse'])}"
    )
    assert all(np.isfinite(v) for v in model.history["mse"]), (
        "Some mse history values are non-finite"
    )
    print(f"model.history populated correctly across 3 epochs: PASS")
    print(f"  mse per epoch: {[round(v, 4) for v in model.history['mse']]}")


def test_fit_mse_trends_downward_with_overfit():
    """
    With a tiny dataset, many epochs, and a small model, MSE should
    trend downward as the model overfits -- same memorization check as
    before, just now going through the full fit() pipeline instead of
    calling fit_subseries() directly.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # Very small dataset (one short recording, small seq_len) so the
    # model can actually memorize it in a reasonable number of epochs.
    ds = build_toy_dataset(n_timesteps=20, n_features=4, seq_len=3)
    model = RTRBM(n_visible=4, n_hidden=6, learning_rate=0.1)

    model.fit(ds, batch_size=4, epochs=50)

    early_mse = np.mean(model.history["mse"][:5])
    late_mse = np.mean(model.history["mse"][-5:])

    print(f"  early MSE (first 5 epochs): {early_mse:.4f}")
    print(f"  late MSE (last 5 epochs):   {late_mse:.4f}")

    assert late_mse < early_mse, (
        f"Expected MSE to decrease over training, but "
        f"early={early_mse:.4f}, late={late_mse:.4f}"
    )
    print("MSE trends downward over full fit() training run: PASS")


if __name__ == "__main__":
    test_fit_raises_no_longer()
    test_fit_runs_and_returns_finite_mse()
    test_fit_populates_history()
    test_fit_mse_trends_downward_with_overfit()
    print("\nAll fit() smoke tests passed.")
