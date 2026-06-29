"""
Smoke test for the min-max scaling utility.
"""
import numpy as np

from sit_fuse_rtrbm.utils.scale import (
    fit_minmax_scale, apply_minmax_scale, minmax_scale
)

import logging
logging.disable(logging.CRITICAL)


def test_basic_scaling_range():
    """Scaled training data should land in [0, 1], with min->0, max->1."""
    data = np.array([[1.0, 100.0], [5.0, 200.0], [
                    10.0, 300.0]], dtype=np.float32)

    scaled, d_min, d_max = minmax_scale(data)

    assert np.allclose(scaled.min(axis=0), 0.0, atol=1e-6)
    assert np.allclose(scaled.max(axis=0), 1.0, atol=1e-6)
    assert scaled.shape == data.shape

    print("Basic scaling range [0, 1]: PASS")


def test_per_feature_independence():
    """Each feature/column must be scaled using ITS OWN min/max, not a
    global min/max across all features mixed together."""
    # Feature 0 ranges 0-10, feature 1 ranges 1000-2000 -- very different
    # scales. If features were accidentally mixed, feature 0 would barely
    # move off zero.
    data = np.array([[0.0, 1000.0], [10.0, 2000.0]], dtype=np.float32)

    scaled, d_min, d_max = minmax_scale(data)

    assert np.isclose(scaled[0, 0], 0.0)
    assert np.isclose(scaled[1, 0], 1.0)
    assert np.isclose(scaled[0, 1], 0.0)
    assert np.isclose(scaled[1, 1], 1.0)

    print("Per-feature independent scaling: PASS")


def test_constant_feature_does_not_divide_by_zero():
    """A feature with no variation at all (max == min) must not produce
    NaN/Inf -- this is exactly what EPSILON guards against."""
    data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], dtype=np.float32)

    scaled, d_min, d_max = minmax_scale(data)

    assert np.all(np.isfinite(scaled)), "Constant feature produced NaN/Inf"
    print("Constant feature (max==min) handled safely: PASS")


def test_fit_on_train_apply_to_val_no_leakage():
    """
    THE important real-world check: fit stats on a TRAIN split only, then
    apply those same stats to a separate VAL split. Val data should NOT
    influence the stats at all, and val values outside train's range
    should land outside [0, 1] (not clipped, not silently renormalized).
    """
    train_data = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    val_data = np.array([[-5.0], [25.0], [10.0]], dtype=np.float32)

    d_min, d_max = fit_minmax_scale(train_data)

    # Stats must come ONLY from train: min=0, max=20.
    assert np.isclose(d_min[0, 0], 0.0)
    assert np.isclose(d_max[0, 0], 20.0)

    scaled_val = apply_minmax_scale(val_data, d_min, d_max)

    # -5 is below train's min -> should scale to slightly below 0.
    assert scaled_val[0, 0] < 0.0
    # 25 is above train's max -> should scale to slightly above 1.
    assert scaled_val[1, 0] > 1.0
    # 10 is within train's range -> should scale to exactly 0.5.
    assert np.isclose(scaled_val[2, 0], 0.5)

    print("Fit-on-train, apply-to-val (no leakage, no clipping): PASS")


def test_works_on_3d_sequence_shaped_data():
    """
    SFTemporalDataset produces (batch, seq_len, n_features)-shaped data.
    Confirm scaling still works correctly with that shape, treating the
    LAST axis as features (the default).
    """
    # shape (batch=2, seq_len=3, n_features=2)
    data = np.array([
        [[0.0, 100.0], [5.0, 150.0], [10.0, 200.0]],
        [[2.0, 120.0], [4.0, 140.0], [6.0, 160.0]],
    ], dtype=np.float32)

    scaled, d_min, d_max = minmax_scale(data)

    assert scaled.shape == data.shape
    assert np.all(np.isfinite(scaled))
    # Feature 0's global min across BOTH batch elements and all timesteps
    # should be 0.0 (from batch 0, t=0), and max should be 10.0.
    assert np.isclose(d_min[..., 0].item(), 0.0)
    assert np.isclose(d_max[..., 0].item(), 10.0)

    print("Works correctly on 3D (batch, seq_len, n_features) data: PASS")


if __name__ == "__main__":
    test_basic_scaling_range()
    test_per_feature_independence()
    test_constant_feature_does_not_divide_by_zero()
    test_fit_on_train_apply_to_val_no_leakage()
    test_works_on_3d_sequence_shaped_data()
    print("\nAll minmax_scale smoke tests passed.")
