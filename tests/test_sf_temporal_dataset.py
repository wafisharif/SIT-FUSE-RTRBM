"""
Smoke test for SFTemporalDataset, via the installed sit_fuse_rtrbm package.
"""
import numpy as np

from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset

import logging
logging.disable(logging.CRITICAL)


def test_sequence_count_and_shape():
    n_timesteps, n_features, seq_len = 20, 5, 4
    data = np.tile(np.arange(n_timesteps).reshape(-1, 1),
                   (1, n_features)).astype(np.float32)
    targets = np.arange(n_timesteps)

    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=seq_len, do_shuffle=False)

    assert len(ds) == n_timesteps - seq_len + 1
    sample, target = ds[0]
    assert sample.shape == (seq_len, n_features)
    print("Sequence count + shape: PASS")


def test_sequence_ordering_is_preserved():
    n_timesteps, n_features, seq_len = 20, 5, 4
    data = np.tile(np.arange(n_timesteps).reshape(-1, 1),
                   (1, n_features)).astype(np.float32)
    targets = np.arange(n_timesteps)

    ds = SFTemporalDataset()
    ds.init_from_array(data, targets, seq_len=seq_len, do_shuffle=False)

    sample, target = ds[0]
    assert np.array_equal(sample[:, 0], np.array([0, 1, 2, 3]))
    assert target == 3
    print("Sequence ordering + target convention: PASS")


if __name__ == "__main__":
    test_sequence_count_and_shape()
    test_sequence_ordering_is_preserved()
    print("\nAll SFTemporalDataset smoke tests passed.")
