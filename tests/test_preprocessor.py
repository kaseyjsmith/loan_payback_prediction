import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import Preprocessor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "preprocess"))

from preprocess import Preprocessor


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing."""
    return pd.DataFrame({
        'credit_score': [700, 650, 800, 720, 680],
        'interest_rate': [5.5, 7.2, 4.1, 6.0, 8.5],
        'debt_to_income_ratio': [0.3, 0.45, 0.2, 0.35, 0.5],
        'employment_status': ['Employed', 'Student', 'Employed', 'Unemployed', 'Employed'],
        'grade_subgrade': ['A1', 'B2', 'A3', 'C1', 'B1'],
        'loan_paid_back': [1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_test_data():
    """Create sample test data (no target) for testing.

    Note: Uses DIFFERENT categorical values than train data to test
    that OneHotEncoder handles this properly. Test data has A2 and B3,
    while train has A1, A3, B1, B2, C1.
    """
    return pd.DataFrame({
        'credit_score': [710, 690],
        'interest_rate': [5.8, 6.5],
        'debt_to_income_ratio': [0.32, 0.40],
        'employment_status': ['Employed', 'Student'],
        'grade_subgrade': ['A2', 'B3']  # Different from train: A2, B3 vs A1, A3, B1, B2, C1
    })


@pytest.fixture
def temp_files(tmp_path, sample_train_data, sample_test_data):
    """Create temporary CSV files for testing."""
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"

    sample_train_data.to_csv(train_file, index=False)
    sample_test_data.to_csv(test_file, index=False)

    return {
        'train': str(train_file),
        'test': str(test_file),
        'dir': str(tmp_path)
    }


class TestPreprocessorInit:
    """Test Preprocessor initialization."""

    def test_training_mode_initialization(self, temp_files):
        """Test that Preprocessor initializes correctly in training mode."""
        prep = Preprocessor(temp_files['train'], training=True)

        assert prep.training is True
        assert prep.y is not None
        assert len(prep.y) == 5
        assert 'loan_paid_back' not in prep.df.columns
        assert prep.scaler is not None

    def test_inference_mode_initialization_with_scaler(self, temp_files):
        """Test that Preprocessor initializes correctly in inference mode."""
        # First create and save a scaler and encoder
        train_prep = Preprocessor(temp_files['train'], training=True)
        train_prep.preprocess()
        scaler_path = temp_files['dir'] + "/scaler.pkl"
        encoder_path = temp_files['dir'] + "/encoder.pkl"
        joblib.dump(train_prep.scaler, scaler_path)
        joblib.dump(train_prep.encoder, encoder_path)

        # Now test inference mode
        test_prep = Preprocessor(
            temp_files['test'],
            training=False,
            scaler_path=scaler_path,
            encoder_path=encoder_path
        )

        assert test_prep.training is False
        assert test_prep.y is None
        assert 'loan_paid_back' not in test_prep.df.columns
        assert test_prep.scaler is not None
        assert test_prep.encoder is not None

    def test_inference_mode_without_scaler_raises_error(self, temp_files):
        """Test that inference mode without scaler_path raises ValueError."""
        with pytest.raises(ValueError, match="Must provide scaler_path"):
            Preprocessor(temp_files['test'], training=False)

    def test_inference_mode_without_encoder_raises_error(self, temp_files):
        """Test that inference mode without encoder_path raises ValueError."""
        scaler_path = temp_files['dir'] + "/scaler.pkl"
        joblib.dump(StandardScaler(), scaler_path)

        with pytest.raises(ValueError, match="Must provide encoder_path"):
            Preprocessor(temp_files['test'], training=False, scaler_path=scaler_path)

    def test_feature_columns_loaded_correctly(self, temp_files):
        """Test that only specified columns are loaded."""
        prep = Preprocessor(temp_files['train'], training=True)

        expected_cols = [
            'credit_score', 'interest_rate', 'debt_to_income_ratio',
            'employment_status', 'grade_subgrade'
        ]
        assert list(prep.df.columns) == expected_cols


class TestPreprocessorEncoding:
    """Test categorical encoding functionality."""

    def test_one_hot_encoding(self, temp_files):
        """Test that categorical variables are one-hot encoded."""
        prep = Preprocessor(temp_files['train'], training=True)
        prep.preprocess()

        # Check that original categorical columns are gone
        assert 'employment_status' not in prep.encoded.columns
        assert 'grade_subgrade' not in prep.encoded.columns

        # Check that one-hot encoded columns exist
        assert 'employment_status_Employed' in prep.encoded.columns
        assert 'employment_status_Student' in prep.encoded.columns
        assert 'employment_status_Unemployed' in prep.encoded.columns

        assert 'grade_subgrade_A1' in prep.encoded.columns
        assert 'grade_subgrade_B2' in prep.encoded.columns

    def test_one_hot_encoding_values(self, temp_files):
        """Test that one-hot encoding produces correct binary values."""
        prep = Preprocessor(temp_files['train'], training=True)
        prep.preprocess()

        # Check that all encoded columns contain only 0s and 1s
        encoded_cols = [col for col in prep.encoded.columns
                       if 'employment_status_' in col or 'grade_subgrade_' in col]

        for col in encoded_cols:
            assert prep.encoded[col].isin([0, 1]).all()


class TestPreprocessorScaling:
    """Test scaling functionality."""

    def test_training_mode_fits_scaler(self, temp_files):
        """Test that training mode fits the scaler."""
        prep = Preprocessor(temp_files['train'], training=True)
        prep.preprocess()

        # Check that scaler has learned parameters
        assert prep.scaler.mean_ is not None
        assert prep.scaler.scale_ is not None
        assert len(prep.scaler.mean_) == 3  # 3 continuous features

    def test_continuous_features_are_scaled(self, temp_files):
        """Test that continuous features are standardized."""
        prep = Preprocessor(temp_files['train'], training=True)
        prep.preprocess()

        # After standardization, mean should be ~0 and std ~1
        # Note: With small samples (n=5), std may deviate more from 1
        for col in prep.continuous_features:
            mean = prep.encoded[col].mean()
            std = prep.encoded[col].std()
            assert abs(mean) < 1e-10  # Close to 0
            assert abs(std - 1.0) < 0.2  # Relaxed tolerance for small sample

    def test_inference_mode_uses_pretrained_scaler(self, temp_files):
        """Test that inference mode uses pre-fitted scaler correctly."""
        # Train and save scaler and encoder
        train_prep = Preprocessor(temp_files['train'], training=True)
        train_prep.preprocess()
        scaler_path = temp_files['dir'] + "/scaler.pkl"
        encoder_path = temp_files['dir'] + "/encoder.pkl"
        joblib.dump(train_prep.scaler, scaler_path)
        joblib.dump(train_prep.encoder, encoder_path)

        # Use scaler and encoder on test data
        test_prep = Preprocessor(
            temp_files['test'],
            training=False,
            scaler_path=scaler_path,
            encoder_path=encoder_path
        )
        test_prep.preprocess()

        # Verify scaler parameters match
        assert np.allclose(test_prep.scaler.mean_, train_prep.scaler.mean_)
        assert np.allclose(test_prep.scaler.scale_, train_prep.scaler.scale_)


class TestPreprocessorSave:
    """Test save functionality."""

    def test_save_in_training_mode(self, temp_files, monkeypatch):
        """Test that save works in training mode."""
        # Monkeypatch proj_root to use temp directory
        import preprocess
        monkeypatch.setattr(preprocess, 'proj_root', temp_files['dir'] + '/')

        prep = Preprocessor(temp_files['train'], training=True)
        prep.preprocess()

        # Create scaled directory
        os.makedirs(temp_files['dir'] + '/data/scaled', exist_ok=True)
        prep.save()

        # Check that files were created
        scaler_file = f"{temp_files['dir']}/data/scaled/scaler_{prep.run_id}.pkl"
        encoder_file = f"{temp_files['dir']}/data/scaled/encoder_{prep.run_id}.pkl"
        encoded_file = f"{temp_files['dir']}/data/scaled/encoded_{prep.run_id}.pkl"

        assert os.path.exists(scaler_file)
        assert os.path.exists(encoder_file)
        assert os.path.exists(encoded_file)

    def test_save_in_inference_mode_raises_error(self, temp_files):
        """Test that save raises error in inference mode."""
        # Create a scaler and encoder first
        train_prep = Preprocessor(temp_files['train'], training=True)
        train_prep.preprocess()
        scaler_path = temp_files['dir'] + "/scaler.pkl"
        encoder_path = temp_files['dir'] + "/encoder.pkl"
        joblib.dump(train_prep.scaler, scaler_path)
        joblib.dump(train_prep.encoder, encoder_path)

        # Try to save in inference mode
        test_prep = Preprocessor(
            temp_files['test'],
            training=False,
            scaler_path=scaler_path,
            encoder_path=encoder_path
        )
        test_prep.preprocess()

        with pytest.raises(ValueError, match="Cannot save in inference mode"):
            test_prep.save()


class TestPreprocessorEndToEnd:
    """Test end-to-end preprocessing workflow."""

    def test_train_test_workflow(self, temp_files):
        """Test complete train/test preprocessing workflow.

        With OneHotEncoder, train and test will have consistent columns
        even if test data has different categorical values.
        """
        # Train preprocessing
        train_prep = Preprocessor(temp_files['train'], training=True)
        train_prep.preprocess()

        # Save scaler and encoder
        scaler_path = temp_files['dir'] + "/scaler.pkl"
        encoder_path = temp_files['dir'] + "/encoder.pkl"
        joblib.dump(train_prep.scaler, scaler_path)
        joblib.dump(train_prep.encoder, encoder_path)

        # Test preprocessing
        test_prep = Preprocessor(
            temp_files['test'],
            training=False,
            scaler_path=scaler_path,
            encoder_path=encoder_path
        )
        test_prep.preprocess()

        # Verify both have same columns (critical for neural networks!)
        train_cols = set(train_prep.encoded.columns)
        test_cols = set(test_prep.encoded.columns)
        assert train_cols == test_cols, "Train and test must have same columns"

        # Verify continuous features are present
        for col in train_prep.continuous_features:
            assert col in train_prep.encoded.columns
            assert col in test_prep.encoded.columns

        # Verify one-hot encoding happened
        assert any('employment_status_' in col for col in test_prep.encoded.columns)
        assert any('grade_subgrade_' in col for col in test_prep.encoded.columns)

    def test_preprocessed_data_shape(self, temp_files):
        """Test that preprocessed data has expected shape."""
        prep = Preprocessor(temp_files['train'], training=True)
        prep.preprocess()

        # Should have 5 rows
        assert prep.encoded.shape[0] == 5

        # Should have continuous features + one-hot encoded categoricals
        # 3 continuous + encoded employment_status (3) + encoded grade_subgrade (5)
        expected_cols = 3 + 3 + 5
        assert prep.encoded.shape[1] == expected_cols

    def test_single_row_inference(self, temp_files, tmp_path):
        """Test that single-row inference works correctly.

        This is critical for production: when you feed a single prediction
        request to your neural network, the preprocessor must produce the
        exact same number of features as during training.
        """
        # Train and save preprocessors
        train_prep = Preprocessor(temp_files['train'], training=True)
        train_prep.preprocess()
        scaler_path = temp_files['dir'] + "/scaler.pkl"
        encoder_path = temp_files['dir'] + "/encoder.pkl"
        joblib.dump(train_prep.scaler, scaler_path)
        joblib.dump(train_prep.encoder, encoder_path)

        # Create single-row test file
        single_row = pd.DataFrame({
            'credit_score': [750],
            'interest_rate': [5.0],
            'debt_to_income_ratio': [0.25],
            'employment_status': ['Employed'],
            'grade_subgrade': ['A1']
        })
        single_row_file = tmp_path / "single_row.csv"
        single_row.to_csv(single_row_file, index=False)

        # Process single row
        single_prep = Preprocessor(
            str(single_row_file),
            training=False,
            scaler_path=scaler_path,
            encoder_path=encoder_path
        )
        single_prep.preprocess()

        # Critical assertion: single row has SAME number of columns as training data
        assert single_prep.encoded.shape[1] == train_prep.encoded.shape[1], \
            "Single row must have same features as training data"

        # Verify it's actually 1 row
        assert single_prep.encoded.shape[0] == 1

        print(f"\nSingle row inference works!")
        print(f"  Training shape: {train_prep.encoded.shape}")
        print(f"  Single row shape: {single_prep.encoded.shape}")
        print(f"  Columns match: {single_prep.encoded.shape[1] == train_prep.encoded.shape[1]}")
