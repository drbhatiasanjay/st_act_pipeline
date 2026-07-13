"""
Unit tests for AnisotropicZarrLoader with real staged data.
Tests validate correct loading of Zarr v3 OME-NGFF stores with proper anisotropy and quantile normalization.
"""

import os

import numpy as np
import pytest

from src.data_loader import AnisotropicZarrLoader


class TestAnisotropicZarrLoaderReal:
    """Test AnisotropicZarrLoader with real staged Zarr v3 data."""

    @pytest.fixture
    def train_data_path(self):
        """Path to real training data."""
        path = "data/staging/train/44b6_0113de3b.zarr"
        assert os.path.exists(path), f"Test data not found at {path}"
        return path

    @pytest.fixture
    def test_data_path(self):
        """Path to real test data (no ground truth)."""
        path = "data/staging/test/44b6_0113de3b.zarr"
        # Test data may not exist in all environments, but we'll check if it does
        return path if os.path.exists(path) else None

    def test_loader_initialization_real_data(self, train_data_path):
        """Test that loader correctly initializes with real Zarr v3 OME-NGFF store."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        assert loader.dataset is not None, "Dataset should be initialized"
        assert loader.dataset.shape is not None, "Dataset should have shape"

    def test_real_data_shape_is_4d(self, train_data_path):
        """Test that loaded data has correct 4D shape (T, Z, Y, X)."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        t, z, y, x = loader.get_shape()
        assert t > 0 and z > 0 and y > 0 and x > 0, f"All dimensions must be > 0, got ({t}, {z}, {y}, {x})"
        assert loader.dataset.shape == (t, z, y, x), f"Shape mismatch: {loader.dataset.shape} != ({t}, {z}, {y}, {x})"

    def test_real_data_dtype_uint16(self, train_data_path):
        """Test that raw real data is uint16 as expected."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        assert loader.dataset.dtype == np.uint16, f"Expected uint16, got {loader.dataset.dtype}"

    def test_anisotropy_is_4_1_1(self, train_data_path):
        """Test that default anisotropy is correctly set to (4.0, 1.0, 1.0)."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        expected = np.array([4.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(loader.anisotropy_ratio, expected,
                                           err_msg="Anisotropy should be (4.0, 1.0, 1.0)")

    def test_load_first_timepoint(self, train_data_path):
        """Test that first timepoint can be loaded without errors."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        vol = loader.load_timepoint_block(0, normalize=False)
        assert vol is not None, "Loaded volume should not be None"
        assert vol.shape == loader.dataset.shape[1:], "Loaded volume should be 3D (Z, Y, X)"
        assert vol.dtype == np.uint16, f"Raw data should be uint16, got {vol.dtype}"

    def test_quantile_normalization_applied(self, train_data_path):
        """Test that quantile normalization is correctly applied when available."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)

        # Load raw data
        vol_raw = loader.load_timepoint_block(0, normalize=False)

        # Load with normalization
        vol_normalized = loader.load_timepoint_block(0, normalize=True)

        # If quantile params were found, normalized data should be float32 and in [0, 1]
        if loader._quantile_normalization_params is not None:
            assert vol_normalized.dtype == np.float32, f"Normalized data should be float32, got {vol_normalized.dtype}"
            assert np.min(vol_normalized) >= 0.0, f"Normalized min should be >= 0, got {np.min(vol_normalized)}"
            assert np.max(vol_normalized) <= 1.0, f"Normalized max should be <= 1, got {np.max(vol_normalized)}"
            # Normalized and raw should be different
            assert not np.allclose(vol_raw.astype(np.float32), vol_normalized), \
                "Normalization should change values"
        else:
            # If no quantile params, normalized should be same as raw
            assert vol_normalized.dtype == vol_raw.dtype, "Data types should match without quantile params"

    def test_quantile_params_extracted(self, train_data_path):
        """Test that quantile normalization parameters are extracted from metadata."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        # Real staged data should have image_statistics.quantiles in metadata
        assert loader._quantile_normalization_params is not None, \
            "Quantile parameters should be extracted from real data metadata"
        q_low, q_high = loader._quantile_normalization_params
        assert q_low < q_high, f"Quantile range invalid: q_low={q_low} >= q_high={q_high}"
        assert q_low >= 0, f"q_low should be non-negative, got {q_low}"

    def test_repeated_call_for_same_timepoint_does_not_redecompress(self, train_data_path):
        """REGRESSION-relevant: CompetitionDataset.__getitem__ requests (t, t+1)
        per item with shuffle=False consecutive access, so item i's t+1 is item
        i+1's t -- confirmed live in a real run's log as every timepoint being
        decompressed twice in a row (~30s each) with no caching at all. The
        single-slot cache must make a second identical call a no-op."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)

        class _CountingDatasetProxy:
            def __init__(self, real_dataset):
                self._real = real_dataset
                self.call_count = 0

            def __getitem__(self, key):
                self.call_count += 1
                return self._real[key]

            def __getattr__(self, name):
                return getattr(self._real, name)

        proxy = _CountingDatasetProxy(loader.dataset)
        loader.dataset = proxy

        first = loader.load_timepoint_block(0, normalize=True)
        second = loader.load_timepoint_block(0, normalize=True)

        assert proxy.call_count == 1, "second identical call must hit the cache, not re-decompress"
        assert np.array_equal(first, second)

    def test_different_timepoint_evicts_the_cache_and_redecompresses(self, train_data_path):
        """The cache must not silently return stale data for a different t."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        t_dim = loader.get_shape()[0]
        if t_dim < 2:
            pytest.skip("Need at least 2 timepoints for this test")

        class _CountingDatasetProxy:
            def __init__(self, real_dataset):
                self._real = real_dataset
                self.call_count = 0

            def __getitem__(self, key):
                self.call_count += 1
                return self._real[key]

            def __getattr__(self, name):
                return getattr(self._real, name)

        proxy = _CountingDatasetProxy(loader.dataset)
        loader.dataset = proxy

        loader.load_timepoint_block(0, normalize=True)
        loader.load_timepoint_block(1, normalize=True)

        assert proxy.call_count == 2, "a different timepoint must always trigger a real decompress"

    def test_load_multiple_timepoints(self, train_data_path):
        """Test loading multiple timepoints sequentially."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        t_dim, z, y, x = loader.get_shape()

        # Load first few timepoints
        num_to_load = min(3, t_dim)
        for t in range(num_to_load):
            vol = loader.load_timepoint_block(t, normalize=True)
            assert vol.shape == (z, y, x), f"Timepoint {t} shape mismatch"
            assert vol.size > 0, f"Timepoint {t} should have data"

    def test_normalized_values_in_expected_range(self, train_data_path):
        """Test that normalized data values are in expected [0, 1] range."""
        loader = AnisotropicZarrLoader(store_path=train_data_path, simulate=False)
        vol_normalized = loader.load_timepoint_block(0, normalize=True)

        if loader._quantile_normalization_params is not None:
            # Most values should be in [0, 1] with some edge cases
            valid_range = np.sum((vol_normalized >= 0.0) & (vol_normalized <= 1.0))
            total_voxels = vol_normalized.size
            validity_ratio = valid_range / total_voxels
            assert validity_ratio > 0.95, f"Most values should be in [0, 1], got {validity_ratio*100:.1f}% valid"

    def test_cannot_load_with_simulate_false_nonexistent(self):
        """Test that simulate=False raises error for non-existent path."""
        fake_path = "data/nonexistent_store.zarr"
        with pytest.raises(FileNotFoundError):
            AnisotropicZarrLoader(store_path=fake_path, simulate=False)

    def test_simulate_false_is_default(self, train_data_path):
        """Test that simulate=False is the default (requires real data)."""
        # Should work with real data
        loader = AnisotropicZarrLoader(store_path=train_data_path)  # No simulate parameter
        assert loader.simulate is False, "simulate should default to False"
        assert loader.dataset is not None, "Should load real data with simulate=False (default)"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
