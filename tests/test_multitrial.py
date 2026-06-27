"""
Smoke test for the multi-trial loader. Critical check: sequences must
NEVER cross a trial boundary, even though the data is fed as separate
files.
"""
import numpy as np
import tempfile
import os

from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset


def test_single_trial_path_unchanged():
    """Regression check: the original single-array path must still work
    identically after refactoring to share __window_single_trial__."""
    n_timesteps, n_features, seq_len = 20, 5, 4
    data = np.tile(np.arange(n_timesteps).reshape(-1, 1),
                   (1, n_features)).astype(np.float32)
    targets = np.arange(n_timesteps)

    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=seq_len, do_shuffle=False)

    assert len(ds) == n_timesteps - seq_len + 1
    sample, target = ds[0]
    assert np.array_equal(sample[:, 0], np.array([0, 1, 2, 3]))
    print("Single-trial path unchanged after refactor: PASS")


def test_multitrial_never_crosses_boundary():
    seq_len = 4

    # Trial 0: timesteps 100-109 (10 steps). Trial 1: timesteps 900-904
    # (5 steps). Deliberately distinct number ranges so we can immediately
    # spot if a sequence accidentally mixes values from both trials.
    trial0 = np.tile(np.arange(100, 110).reshape(-1, 1),
                     (1, 3)).astype(np.float32)
    trial1 = np.tile(np.arange(900, 905).reshape(-1, 1),
                     (1, 3)).astype(np.float32)

    tmpdir = tempfile.mkdtemp()
    f0 = os.path.join(tmpdir, "trial0.npy")
    f1 = os.path.join(tmpdir, "trial1.npy")
    np.save(f0, trial0)
    np.save(f1, trial1)

    ds = SFTemporalDataset()
    ds.read_and_preprocess_data([f0, f1], seq_len=seq_len, do_shuffle=False)

    # Expected sequence counts: trial0 -> 10-4+1=7, trial1 -> 5-4+1=2. Total 9.
    assert len(ds) == 9, f"Expected 9 total sequences, got {len(ds)}"

    # THE critical check: every single sequence's values must come from
    # ONLY the 100s range or ONLY the 900s range, never both.
    for i in range(len(ds)):
        sample, target = ds[i]
        col = sample[:, 0]
        all_low = np.all(col < 200)
        all_high = np.all(col >= 200)
        assert all_low or all_high, (
            f"Sequence {i} MIXES values across trials: {col} "
            f"-- a sequence crossed a trial boundary!"
        )

    print("Multi-trial loading never crosses a trial boundary: PASS")


def test_multitrial_target_traces_back_to_trial():
    seq_len = 4
    trial0 = np.tile(np.arange(10).reshape(-1, 1), (1, 3)).astype(np.float32)
    trial1 = np.tile(np.arange(5).reshape(-1, 1), (1, 3)).astype(np.float32)

    tmpdir = tempfile.mkdtemp()
    f0 = os.path.join(tmpdir, "trial0.npy")
    f1 = os.path.join(tmpdir, "trial1.npy")
    np.save(f0, trial0)
    np.save(f1, trial1)

    ds = SFTemporalDataset()
    ds.read_and_preprocess_data([f0, f1], seq_len=seq_len, do_shuffle=False)

    # First sequence should be tagged with trial_idx=0.
    _, target0 = ds[0]
    assert target0[0] == 0, f"Expected trial_idx 0, got {target0[0]}"

    # Last sequence should be tagged with trial_idx=1 (trial1's last window).
    _, target_last = ds[len(ds) - 1]
    assert target_last[0] == 1, f"Expected trial_idx 1, got {target_last[0]}"

    print("Target correctly traces sequences back to source trial: PASS")


def test_short_trial_is_skipped_with_warning():
    seq_len = 4
    too_short_trial = np.zeros((2, 3), dtype=np.float32)  # only 2 timesteps
    normal_trial = np.tile(np.arange(10).reshape(-1, 1),
                           (1, 3)).astype(np.float32)

    tmpdir = tempfile.mkdtemp()
    f0 = os.path.join(tmpdir, "tooshort.npy")
    f1 = os.path.join(tmpdir, "normal.npy")
    np.save(f0, too_short_trial)
    np.save(f1, normal_trial)

    ds = SFTemporalDataset()
    ds.read_and_preprocess_data([f0, f1], seq_len=seq_len, do_shuffle=False)

    # Only the normal trial's sequences should exist: 10-4+1=7
    assert len(
        ds) == 7, f"Expected 7 sequences (short trial skipped), got {len(ds)}"
    print("Too-short trial correctly skipped (with warning): PASS")


if __name__ == "__main__":
    test_single_trial_path_unchanged()
    test_multitrial_never_crosses_boundary()
    test_multitrial_target_traces_back_to_trial()
    test_short_trial_is_skipped_with_warning()
    print("\nAll multi-trial smoke tests passed.")
