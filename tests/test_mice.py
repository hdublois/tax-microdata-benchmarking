"""
Unit tests for the MICE (Multivariate Imputation by Chained Equations) class.

This module tests the core functionality of the MICE class including:
- Constructor parameter validation
- Basic imputation with synthetic data
- Monotone vs non-monotone mode behavior
- Reproducibility with random seeds
"""

import numpy as np
import pytest
from tmd.utils.mice import MICE

# =============================================================================
# 1. CONSTRUCTOR VALIDATION TESTS
# =============================================================================


class TestConstructorValidation:
    """Test that MICE constructor properly validates all parameters."""

    def test_valid_minimal_constructor(self):
        """Test constructor with minimal valid parameters."""
        mice = MICE(
            x_obs=100,
            x_var=5,
            x_idx=[0, 1],
            x_ign=[],
        )
        assert mice.n_obs == 100
        assert mice.n_var == 5
        assert mice.x_idx == [0, 1]
        assert mice.x_ign == []
        assert mice.n_iters == 10  # default

    def test_x_obs_validation(self):
        """Test x_obs parameter validation."""
        # Non-integer x_obs
        with pytest.raises(AssertionError, match="x_obs must be an integer"):
            MICE(x_obs=100.5, x_var=5, x_idx=[0], x_ign=[])

        # Non-positive x_obs
        with pytest.raises(AssertionError, match="x_obs <= 0"):
            MICE(x_obs=0, x_var=5, x_idx=[0], x_ign=[])

        with pytest.raises(AssertionError, match="x_obs <= 0"):
            MICE(x_obs=-10, x_var=5, x_idx=[0], x_ign=[])

    def test_x_var_validation(self):
        """Test x_var parameter validation."""
        # Non-integer x_var
        with pytest.raises(AssertionError, match="x_var must be an integer"):
            MICE(x_obs=100, x_var=5.5, x_idx=[0], x_ign=[])

        # Non-positive x_var
        with pytest.raises(AssertionError, match="x_var <= 0"):
            MICE(x_obs=100, x_var=0, x_idx=[0], x_ign=[])

    def test_x_idx_validation(self):
        """Test x_idx parameter validation."""
        # Non-list x_idx
        with pytest.raises(AssertionError, match="x_idx must be a list"):
            MICE(x_obs=100, x_var=5, x_idx=(0, 1), x_ign=[])

        # Empty x_idx
        with pytest.raises(AssertionError, match="len\\(x_idx\\) <= 0"):
            MICE(x_obs=100, x_var=5, x_idx=[], x_ign=[])

        # Duplicate indices in x_idx
        with pytest.raises(AssertionError, match="x_idx contains duplicates"):
            MICE(x_obs=100, x_var=5, x_idx=[0, 1, 0], x_ign=[])

        # Negative index in x_idx
        with pytest.raises(AssertionError, match="min\\(x_idx\\) < 0"):
            MICE(x_obs=100, x_var=5, x_idx=[-1, 0], x_ign=[])

        # Index >= x_var in x_idx
        with pytest.raises(AssertionError, match="max\\(x_idx\\) >= x_var=5"):
            MICE(x_obs=100, x_var=5, x_idx=[0, 5], x_ign=[])

    def test_x_ign_validation(self):
        """Test x_ign parameter validation."""
        # Non-list x_ign
        with pytest.raises(AssertionError, match="x_ign must be a list"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=(2, 3))

        # Duplicate indices in x_ign
        with pytest.raises(AssertionError, match="x_ign contains duplicates"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[2, 2])

        # x_ign contains index from x_idx
        with pytest.raises(
            AssertionError, match="x_ign contains index in x_idx"
        ):
            MICE(x_obs=100, x_var=5, x_idx=[0, 1], x_ign=[1, 2])

        # Negative index in x_ign
        with pytest.raises(AssertionError, match="min\\(x_ign\\) < 0"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[-1])

        # Index >= x_var in x_ign
        with pytest.raises(AssertionError, match="max\\(x_ign\\) >= x_var=5"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[5])

    def test_seed_validation(self):
        """Test seed parameter validation."""
        # Non-integer seed
        with pytest.raises(AssertionError, match="seed must be an integer"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[], seed=123.5)

        # Non-positive seed
        with pytest.raises(AssertionError, match="seed must be positive"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[], seed=0)

        # Seed too large
        with pytest.raises(
            AssertionError, match="seed must be no greater than 999999999"
        ):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[], seed=1_000_000_000)

    def test_min_node_split_size_validation(self):
        """Test min_node_split_size parameter validation."""
        # Non-integer
        with pytest.raises(
            AssertionError, match="min_node_split_size must be an integer"
        ):
            MICE(
                x_obs=100,
                x_var=5,
                x_idx=[0],
                x_ign=[],
                min_node_split_size=2.5,
            )

        # Too small
        with pytest.raises(
            AssertionError, match="min_node_split_size=1 must be 2 or larger"
        ):
            MICE(
                x_obs=100, x_var=5, x_idx=[0], x_ign=[], min_node_split_size=1
            )

    def test_min_leaf_node_size_validation(self):
        """Test min_leaf_node_size parameter validation."""
        # Non-integer
        with pytest.raises(
            AssertionError, match="min_leaf_node_size must be an integer"
        ):
            MICE(
                x_obs=100, x_var=5, x_idx=[0], x_ign=[], min_leaf_node_size=1.5
            )

        # Too small
        with pytest.raises(
            AssertionError, match="min_leaf_node_size=0 must be 1 or larger"
        ):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[], min_leaf_node_size=0)

    def test_iters_validation(self):
        """Test iters parameter validation."""
        # Non-integer
        with pytest.raises(AssertionError, match="iters must be an integer"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[], iters=5.5)

        # Too small
        with pytest.raises(AssertionError, match="iters must be 1 or larger"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[], iters=0)

    def test_monotone_validation(self):
        """Test monotone parameter validation and constraints."""
        # Non-boolean monotone
        with pytest.raises(AssertionError, match="monotone must be a boolean"):
            MICE(x_obs=100, x_var=5, x_idx=[0], x_ign=[], monotone=1)

        # When monotone=True, iters must be 1
        with pytest.raises(
            AssertionError, match="iters must be 1 when monotone is True"
        ):
            MICE(
                x_obs=100, x_var=5, x_idx=[0], x_ign=[], monotone=True, iters=5
            )

        # Valid monotone=True with iters=1
        mice = MICE(
            x_obs=100, x_var=5, x_idx=[0], x_ign=[], monotone=True, iters=1
        )
        assert mice.monotone is True

    def test_adjustment_parameters_require_monotone(self):
        """
        Test that adjustment parameters can only be used with monotone=True.
        """
        # shift requires monotone
        with pytest.raises(
            AssertionError, match="shift must be None if monotone is False"
        ):
            MICE(
                x_obs=100,
                x_var=5,
                x_idx=[0],
                x_ign=[],
                monotone=False,
                shift=[1.0],
            )

        # post_shift_min requires monotone
        with pytest.raises(
            AssertionError,
            match="post_shift_min must be None if monotone is False",
        ):
            MICE(
                x_obs=100,
                x_var=5,
                x_idx=[0],
                x_ign=[],
                monotone=False,
                post_shift_min=[0.0],
            )

        # scale requires monotone
        with pytest.raises(
            AssertionError, match="scale must be None if monotone is False"
        ):
            MICE(
                x_obs=100,
                x_var=5,
                x_idx=[0],
                x_ign=[],
                monotone=False,
                scale=[1.5],
            )

    def test_adjustment_parameter_lengths(self):
        """Test that adjustment parameters must match x_idx length."""
        # shift wrong length
        with pytest.raises(
            AssertionError, match="len\\(shift\\) != len\\(x_idx\\)"
        ):
            MICE(
                x_obs=100,
                x_var=5,
                x_idx=[0, 1],
                x_ign=[],
                monotone=True,
                iters=1,
                shift=[1.0],  # Should be length 2
            )

        # scale wrong length
        with pytest.raises(
            AssertionError, match="len\\(scale\\) != len\\(x_idx\\)"
        ):
            MICE(
                x_obs=100,
                x_var=5,
                x_idx=[0],
                x_ign=[],
                monotone=True,
                iters=1,
                scale=[1.0, 1.5],  # Should be length 1
            )

    def test_properties(self):
        """Test that read-only properties return correct values."""
        mice = MICE(
            x_obs=100,
            x_var=5,
            x_idx=[0, 1],
            x_ign=[],
            seed=42,
            iters=7,
            min_node_split_size=5,
            min_leaf_node_size=2,
        )
        assert mice.random_number_seed == 42
        assert mice.iterations == 7
        assert mice.min_node_split_size == 5
        assert mice.min_leaf_node_size == 2


# =============================================================================
# 2. BASIC IMPUTATION TESTS WITH SIMPLE SYNTHETIC DATA
# =============================================================================


class TestBasicImputation:
    """Test basic imputation functionality with simple synthetic datasets."""

    def test_simple_imputation_single_missing_value(self):
        """Test imputation with a single missing value."""
        # Create simple 3x2 dataset with one missing value
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]])

        mice = MICE(x_obs=3, x_var=2, x_idx=[1], x_ign=[], seed=123, iters=5)
        X_imputed = mice.impute(X)

        # Check that non-missing values are unchanged
        assert X_imputed[0, 0] == 1.0
        assert X_imputed[0, 1] == 2.0
        assert X_imputed[1, 0] == 3.0
        assert X_imputed[1, 1] == 4.0
        assert X_imputed[2, 0] == 5.0

        # Check that missing value was imputed (not NaN)
        assert not np.isnan(X_imputed[2, 1])
        # Should be either 2.0 or 4.0 (bootstrap from observed values)
        assert X_imputed[2, 1] in [2.0, 4.0]

    def test_imputation_preserves_non_missing_values(self):
        """Test that imputation does not change non-missing values."""
        np.random.seed(42)
        # Create 10x4 dataset with random missing pattern
        X = np.random.randn(10, 4)
        X_original = X.copy()

        # Introduce missing values in columns 1 and 2
        X[0:3, 1] = np.nan
        X[0:5, 2] = np.nan

        # Create mask BEFORE imputation (based on what was originally missing)
        non_missing_mask = ~np.isnan(X)

        mice = MICE(x_obs=10, x_var=4, x_idx=[1, 2], x_ign=[], seed=999)
        X_imputed = mice.impute(X)

        # Check all originally non-missing values are unchanged
        np.testing.assert_array_equal(
            X_imputed[non_missing_mask], X_original[non_missing_mask]
        )

    def test_all_missing_values_imputed(self):
        """Test that all NaN values are replaced with imputed values."""
        np.random.seed(42)
        X = np.random.randn(20, 5)

        # Introduce missing values in columns 0, 2, and 4
        X[0:5, 0] = np.nan
        X[0:8, 2] = np.nan
        X[0:3, 4] = np.nan

        mice = MICE(
            x_obs=20, x_var=5, x_idx=[0, 2, 4], x_ign=[], seed=777, iters=3
        )
        X_imputed = mice.impute(X)

        # Check that no NaN values remain
        assert not np.isnan(X_imputed).any()

    def test_imputation_with_ignored_variables(self):
        """Test imputation when some variables are ignored (x_ign)."""
        np.random.seed(42)
        X = np.random.randn(15, 5)

        # Variable 4 will be ignored in tree growing
        X[0:4, 1] = np.nan
        X[0:6, 3] = np.nan

        mice = MICE(
            x_obs=15,
            x_var=5,
            x_idx=[1, 3],
            x_ign=[4],  # Variable 4 not used as predictor
            seed=555,
        )
        X_imputed = mice.impute(X)

        # All missing values should be imputed
        assert not np.isnan(X_imputed).any()

        # Variable 4 should be unchanged (not in x_idx)
        np.testing.assert_array_equal(X_imputed[:, 4], X[:, 4])

    def test_get_ival_stats_shape(self):
        """Test that get_ival_stats returns correct shapes."""
        np.random.seed(42)
        X = np.random.randn(10, 4)
        X[0:3, 1] = np.nan
        X[0:5, 2] = np.nan

        n_iters = 5
        mice = MICE(
            x_obs=10, x_var=4, x_idx=[1, 2], x_ign=[], seed=123, iters=n_iters
        )
        X_imputed = mice.impute(X)
        assert isinstance(X_imputed, np.ndarray)

        mean, sdev, vmin, vmax = mice.get_ival_stats()

        # Each should have shape (x_var, iters+1)
        # Note: iters+1 because iteration 0 is initialization
        expected_shape = (4, n_iters + 1)
        assert mean.shape == expected_shape
        assert sdev.shape == expected_shape
        assert vmin.shape == expected_shape
        assert vmax.shape == expected_shape

        # Statistics should be non-zero for imputed variables at all iterations
        for idx in [1, 2]:
            for itr in range(1, n_iters + 1):
                assert mean[idx, itr] != 0  # Mean should be non-zero
                assert sdev[idx, itr] >= 0  # Std dev should be non-negative

    def test_impute_validates_input_shape(self):
        """Test that impute method validates X shape against constructor."""
        mice = MICE(x_obs=10, x_var=4, x_idx=[1], x_ign=[])

        # Wrong shape should raise assertion
        X_wrong_shape = np.random.randn(10, 5)  # 5 vars instead of 4
        with pytest.raises(AssertionError, match="unexpected X.shape"):
            mice.impute(X_wrong_shape)

        X_wrong_obs = np.random.randn(15, 4)  # 15 obs instead of 10
        with pytest.raises(AssertionError, match="unexpected X.shape"):
            mice.impute(X_wrong_obs)

    def test_impute_validates_missing_pattern(self):
        """Test that impute method validates missing pattern matches x_idx."""
        mice = MICE(x_obs=10, x_var=4, x_idx=[1, 2], x_ign=[])

        # Create data with missing values in wrong columns
        X = np.random.randn(10, 4)
        X[0:3, 0] = np.nan  # Column 0 has missing, but not in x_idx
        X[0:3, 1] = np.nan  # Column 1 has missing (correct)

        with pytest.raises(
            AssertionError, match="unexpected X missing pattern"
        ):
            mice.impute(X)


# =============================================================================
# 3. MONOTONE VS NON-MONOTONE MODE TESTS
# =============================================================================


class TestMonotoneMode:
    """Test differences between monotone and non-monotone modes."""

    def test_monotone_mode_basic(self):
        """Test basic monotone mode imputation."""
        np.random.seed(42)
        X = np.random.randn(20, 4)

        # Create monotone missing pattern: var 1 has 5 missing, var 2 has 10
        X[0:5, 1] = np.nan
        X[0:10, 2] = np.nan

        mice = MICE(
            x_obs=20,
            x_var=4,
            x_idx=[1, 2],  # Ordered by increasing missingness
            x_ign=[],
            monotone=True,
            iters=1,  # Must be 1 for monotone
            seed=123,
        )
        X_imputed = mice.impute(X)

        # All missing values should be imputed
        assert not np.isnan(X_imputed).any()

    def test_monotone_detects_non_monotone_pattern(self):
        """Test that monotone mode detects non-monotone missing patterns."""
        np.random.seed(42)
        X = np.random.randn(20, 4)

        # Create NON-monotone pattern: var 2 has fewer missing than var 1
        X[0:10, 1] = np.nan  # 10 missing
        X[0:5, 2] = np.nan  # 5 missing (violates monotone assumption)

        mice = MICE(
            x_obs=20,
            x_var=4,
            x_idx=[1, 2],  # Ordered incorrectly for actual pattern
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
        )

        with pytest.raises(
            AssertionError, match="non-monotone missing data pattern"
        ):
            mice.impute(X)

    def test_non_monotone_mode_performs_initialization(self):
        """Test that non-monotone mode performs bootstrap initialization."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        X[0:3, 1] = np.nan

        mice = MICE(
            x_obs=10,
            x_var=3,
            x_idx=[1],
            x_ign=[],
            monotone=False,
            iters=3,
            seed=123,
        )
        X_imputed = mice.impute(X)
        assert isinstance(X_imputed, np.ndarray)

        # Check that iteration 0 (initialization) has statistics
        mean, sdev, _, _ = mice.get_ival_stats()
        assert mean[1, 0] != 0  # Iteration 0 should have non-zero mean
        assert sdev[1, 0] >= 0  # Should have non-negative std dev

    def test_monotone_mode_skips_initialization(self):
        """Test that monotone mode skips bootstrap initialization."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        X[0:3, 1] = np.nan

        mice = MICE(
            x_obs=10,
            x_var=3,
            x_idx=[1],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
        )
        X_imputed = mice.impute(X)
        assert not np.isnan(X_imputed).any()

        # Check that iteration 0 (initialization) has zero statistics
        mean, sdev, _, _ = mice.get_ival_stats()
        assert mean[1, 0] == 0  # Iteration 0 should be zero (skipped)
        assert sdev[1, 0] == 0

    def test_monotone_mode_with_shift_adjustment(self):
        """Test monotone mode with shift adjustment parameter."""
        np.random.seed(42)
        X = np.random.randn(15, 3) * 10 + 50  # Mean around 50
        missing_idx = 1
        X[0:5, missing_idx] = np.nan

        # Apply two different shift adjustments
        mice5 = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            shift=[5.0],
        )
        X_imputed_shift5 = mice5.impute(X.copy())
        assert not np.isnan(X_imputed_shift5).any()
        mean5, _, _, _ = mice5.get_ival_stats()
        assert isinstance(mean5, np.ndarray)

        mice7 = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            shift=[7.0],
        )
        X_imputed_shift7 = mice7.impute(X.copy())
        assert not np.isnan(X_imputed_shift7).any()
        mean7, _, _, _ = mice7.get_ival_stats()
        assert isinstance(mean7, np.ndarray)

        # Imputed values should be shifted
        assert mean7[missing_idx, 1] > mean5[missing_idx, 1]

    def test_monotone_mode_with_scale_adjustment(self):
        """Test monotone mode with scale adjustment parameter."""
        np.random.seed(42)
        X = np.random.randn(15, 3) * 10 + 50  # Mean around 50
        missing_idx = 1
        X[0:5, missing_idx] = np.nan

        # Apply two different scale adjustments
        mice2 = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            scale=[2.0],
        )
        X_imputed_scale2 = mice2.impute(X.copy())
        assert not np.isnan(X_imputed_scale2).any()
        mean2, _, _, _ = mice2.get_ival_stats()
        assert isinstance(mean2, np.ndarray)

        mice3 = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            scale=[3.0],
        )
        X_imputed_scale3 = mice3.impute(X.copy())
        assert not np.isnan(X_imputed_scale3).any()
        mean3, _, _, _ = mice3.get_ival_stats()
        assert isinstance(mean3, np.ndarray)

        # Imputed values should be scaled
        assert mean3[missing_idx, 1] > mean2[missing_idx, 1]

    def test_monotone_mode_with_more_zeros_adjustment(self):
        """Test monotone mode with zero_below_abs adjustment parameter."""
        np.random.seed(42)
        X = np.random.randn(15, 3) * 10  # Mean around zero
        missing_idx = 1
        X[0:5, missing_idx] = np.nan

        # Apply two different zero_below_abs adjustments
        mice_no = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            zero_below_abs=[0.0],
        )
        X_imputed_no = mice_no.impute(X.copy())
        assert not np.isnan(X_imputed_no).any()
        imputed_no = X_imputed_no[:, missing_idx]
        num_zeros_no = np.count_nonzero(imputed_no == 0)

        mice_yes = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            zero_below_abs=[7.0],
        )
        X_imputed_yes = mice_yes.impute(X.copy())
        assert not np.isnan(X_imputed_yes).any()
        imputed_yes = X_imputed_yes[:, missing_idx]
        num_zeros_yes = np.count_nonzero(imputed_yes == 0)

        # Should be more imputed zeros in X_imputed_yes than in X_imputed_no
        assert num_zeros_yes > num_zeros_no

    def test_monotone_mode_with_fewer_zeros_adjustment(self):
        """Test monotone mode with convert_zero_prob adjustment parameter."""
        np.random.seed(42)
        X = np.random.randn(15, 3) * 10  # Mean around zero
        missing_idx = 1
        X[10:15, missing_idx] = 0.0  # mass point at zero
        X[0:5, missing_idx] = np.nan

        # Apply two different convert_zero_prob adjustments
        mice_no = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            convert_zero_prob=[0.0],
        )
        X_imputed_no = mice_no.impute(X.copy())
        assert not np.isnan(X_imputed_no).any()
        imputed_no = X_imputed_no[:, missing_idx]
        num_zeros_no = np.count_nonzero(imputed_no == 0)

        mice_yes = MICE(
            x_obs=15,
            x_var=3,
            x_idx=[missing_idx],
            x_ign=[],
            monotone=True,
            iters=1,
            seed=123,
            convert_zero_prob=[0.5],
        )
        X_imputed_yes = mice_yes.impute(X.copy())
        assert not np.isnan(X_imputed_yes).any()
        imputed_yes = X_imputed_yes[:, missing_idx]
        num_zeros_yes = np.count_nonzero(imputed_yes == 0)

        # Should be fewer imputed zeros in X_imputed_yes than in X_imputed_no
        assert num_zeros_yes < num_zeros_no


# =============================================================================
# 4. REPRODUCIBILITY TESTS
# =============================================================================


class TestReproducibility:
    """Test that MICE produces reproducible results with same seed."""

    def test_same_seed_produces_identical_results(self):
        """Test that same seed produces identical imputations."""
        np.random.seed(42)
        X1 = np.random.randn(30, 5)
        X1[0:8, 1] = np.nan
        X1[0:12, 3] = np.nan
        X2 = X1.copy()

        seed = 12345
        mice1 = MICE(
            x_obs=30, x_var=5, x_idx=[1, 3], x_ign=[], seed=seed, iters=5
        )
        mice2 = MICE(
            x_obs=30, x_var=5, x_idx=[1, 3], x_ign=[], seed=seed, iters=5
        )

        X1_imputed = mice1.impute(X1)
        X2_imputed = mice2.impute(X2)

        # Results should be identical
        np.testing.assert_array_equal(X1_imputed, X2_imputed)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different imputations."""
        np.random.seed(42)
        X1 = np.random.randn(30, 5)
        X1[0:8, 1] = np.nan
        X1[0:12, 3] = np.nan
        X2 = X1.copy()

        mice1 = MICE(
            x_obs=30, x_var=5, x_idx=[1, 3], x_ign=[], seed=111, iters=5
        )
        mice2 = MICE(
            x_obs=30, x_var=5, x_idx=[1, 3], x_ign=[], seed=999, iters=5
        )

        X1_imputed = mice1.impute(X1)
        X2_imputed = mice2.impute(X2)

        # Results should be different (at least somewhere)
        assert not np.array_equal(X1_imputed, X2_imputed)

        # But both should have no missing values
        assert not np.isnan(X1_imputed).any()
        assert not np.isnan(X2_imputed).any()

    def test_statistics_reproducible_with_same_seed(self):
        """Test that imputation statistics are reproducible."""
        np.random.seed(42)
        X = np.random.randn(25, 4)
        X[0:7, 1] = np.nan
        X[0:10, 2] = np.nan

        seed = 55555
        iters = 5

        # First run
        mice1 = MICE(
            x_obs=25, x_var=4, x_idx=[1, 2], x_ign=[], seed=seed, iters=iters
        )
        X1 = X.copy()
        mice1.impute(X1)
        mean1, sdev1, min1, max1 = mice1.get_ival_stats()

        # Second run with same seed
        mice2 = MICE(
            x_obs=25, x_var=4, x_idx=[1, 2], x_ign=[], seed=seed, iters=iters
        )
        X2 = X.copy()
        mice2.impute(X2)
        mean2, sdev2, min2, max2 = mice2.get_ival_stats()

        # Statistics should be identical
        np.testing.assert_array_equal(mean1, mean2)
        np.testing.assert_array_equal(sdev1, sdev2)
        np.testing.assert_array_equal(min1, min2)
        np.testing.assert_array_equal(max1, max2)

    def test_seed_increment_across_iterations(self):
        """Test that seed changes across iterations as documented."""
        np.random.seed(42)
        X = np.random.randn(20, 4)
        X[0:5, 1] = np.nan
        X[0:8, 2] = np.nan

        # Run with multiple iterations
        mice = MICE(
            x_obs=20, x_var=4, x_idx=[1, 2], x_ign=[], seed=100, iters=3
        )
        X_imputed = mice.impute(X)
        assert isinstance(X_imputed, np.ndarray)

        mean, sdev, _, _ = mice.get_ival_stats()

        # Statistics should differ across iterations (seed changes)
        # Iteration 1 vs iteration 2
        assert mean[1, 1] != mean[1, 2] or sdev[1, 1] != sdev[1, 2]
        # Iteration 2 vs iteration 3
        assert mean[1, 2] != mean[1, 3] or sdev[1, 2] != sdev[1, 3]

    def test_multiple_imputation_with_different_seeds(self):
        """
        Test multiple imputation scenario: m datasets with different seeds.
        This is the documented use case for multiple imputation.
        """
        np.random.seed(42)
        X_original = np.random.randn(20, 4)
        X_original[0:5, 1] = np.nan
        X_original[0:8, 2] = np.nan

        # Create m=3 imputed datasets
        m = 3
        base_seed = 1000
        imputed_datasets = []

        for i in range(m):
            X = X_original.copy()
            mice = MICE(
                x_obs=20,
                x_var=4,
                x_idx=[1, 2],
                x_ign=[],
                seed=base_seed + i,
                iters=5,
            )
            X_imputed = mice.impute(X)
            imputed_datasets.append(X_imputed)

        # All datasets should be fully imputed
        for ds in imputed_datasets:
            assert not np.isnan(ds).any()

        # Datasets should differ from each other
        assert not np.array_equal(imputed_datasets[0], imputed_datasets[1])
        assert not np.array_equal(imputed_datasets[1], imputed_datasets[2])

        # But non-missing values should be identical across all
        non_missing_mask = ~np.isnan(X_original)
        for ds in imputed_datasets:
            np.testing.assert_array_equal(
                ds[non_missing_mask], X_original[non_missing_mask]
            )
