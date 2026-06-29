"""
Smoke test for subset_training on sequences (random subsetting, resolved
per Nick 2026-06-29). K-means stratification remains explicitly deferred.
"""
import numpy as np

from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset

import logging
logging.disable(logging.CRITICAL)


def test_subset_training_reduces_count():
    n_timesteps, n_features, seq_len = 30, 5, 4
    data = np.tile(np.arange(n_timesteps).reshape(-1, 1),
                   (1, n_features)).astype(np.float32)
    targets = np.arange(n_timesteps)

    full_n_sequences = n_timesteps - seq_len + 1  # 27

    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=seq_len,
                       subset_training=10, do_shuffle=True)

    assert len(
        ds) == 10, f"Expected 10 sequences after subsetting, got {len(ds)}"
    print("subset_training correctly reduces sequence count: PASS")


def test_subset_training_default_disabled():
    n_timesteps, n_features, seq_len = 20, 5, 4
    data = np.tile(np.arange(n_timesteps).reshape(-1, 1),
                   (1, n_features)).astype(np.float32)
    targets = np.arange(n_timesteps)

    ds = SFTemporalDataset()
    # subset_training defaults to -1
    ds.init_from_array(data, targets, seq_len=seq_len, do_shuffle=False)

    expected = n_timesteps - seq_len + 1
    assert len(
        ds) == expected, "Default subset_training=-1 should not reduce sequences"
    print("Default (no subsetting) behaves as before: PASS")


def test_subset_training_requesting_more_than_available():
    n_timesteps, n_features, seq_len = 10, 5, 4
    data = np.tile(np.arange(n_timesteps).reshape(-1, 1),
                   (1, n_features)).astype(np.float32)
    targets = np.arange(n_timesteps)

    full_n_sequences = n_timesteps - seq_len + 1  # 7

    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=seq_len,
                       subset_training=100, do_shuffle=False)

    assert len(ds) == full_n_sequences, (
        f"Requesting more sequences than exist should just return all of them, "
        f"got {len(ds)} expected {full_n_sequences}"
    )
    print("Requesting subset_training larger than pool returns all sequences: PASS")


def test_kmeans_stratification_raises_not_implemented():
    n_timesteps, n_features, seq_len = 20, 5, 4
    data = np.tile(np.arange(n_timesteps).reshape(-1, 1),
                   (1, n_features)).astype(np.float32)
    targets = np.arange(n_timesteps)

    ds = SFTemporalDataset()
    try:
        ds.init_from_array(
            data, targets, seq_len=seq_len, subset_training=5,
            stratify_data={"kmeans": True}, do_shuffle=False
        )
        raise AssertionError(
            "Expected NotImplementedError for kmeans stratification")
    except NotImplementedError:
        print("K-means stratification correctly raises NotImplementedError: PASS")


if __name__ == "__main__":
    test_subset_training_reduces_count()
    test_subset_training_default_disabled()
    test_subset_training_requesting_more_than_available()
    test_kmeans_stratification_raises_not_implemented()
    print("\nAll subsetting smoke tests passed.")
