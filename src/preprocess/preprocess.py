# %%
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import joblib
from uuid import uuid4

# %%
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/kaggle/predicting_load_payback/"


# load the data
class Preprocessor:
    def __init__(
        self,
        filename,
        training=True,
        run_id: str = str(uuid4()),
        scaler_path=None,
        encoder_path=None,
    ):
        self.file = filename
        self.run_id = run_id
        self.training = training

        # Define features first
        self.continuous_features = [
            "credit_score",
            "interest_rate",
            "debt_to_income_ratio",
        ]
        self.categorical_features = ["employment_status", "grade_subgrade"]

        # Build column list based on mode
        feature_cols = self.continuous_features + self.categorical_features
        if training:
            self.included_columns = feature_cols + ["loan_paid_back"]
        else:
            self.included_columns = feature_cols

        # Load data
        self.df = pd.read_csv(filename, usecols=self.included_columns)

        # Extract target if training
        if training:
            self.y = self.df.pop("loan_paid_back")
        else:
            self.y = None

        # Handle scaler
        if training:
            self.scaler = StandardScaler()
        else:
            if scaler_path is None:
                raise ValueError("Must provide scaler_path for inference mode")
            self.scaler = joblib.load(scaler_path)

        # Handle encoder
        if training:
            self.encoder = OneHotEncoder(
                drop=None,  # Keep all categories
                sparse_output=False,  # Return dense array (not sparse matrix)
                handle_unknown="ignore",  # Handle unseen categories gracefully
            )
        else:
            if encoder_path is None:
                raise ValueError("Must provide encoder_path for inference mode")
            self.encoder = joblib.load(encoder_path)

    def encode_categoricals(self, df):
        """Encode categorical features using OneHotEncoder.

        Ensures consistent columns between training and inference.
        """
        if self.training:
            # Fit and transform on training data
            encoded_array = self.encoder.fit_transform(
                df[self.categorical_features]
            )
            # Get feature names
            feature_names = self.encoder.get_feature_names_out(
                self.categorical_features
            )
        else:
            # Transform only (uses fitted categories from training)
            encoded_array = self.encoder.transform(
                df[self.categorical_features]
            )
            feature_names = self.encoder.get_feature_names_out(
                self.categorical_features
            )

        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(
            encoded_array, columns=feature_names, index=df.index
        )

        # Combine with continuous features
        result = pd.concat(
            [
                df[self.continuous_features].reset_index(drop=True),
                encoded_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return result

    # scale the continuous_features
    def scale_continuous_features(self, df, continuous_features):
        """Scale continuous features. Fits on train, transforms on inference."""
        if self.training:
            df[continuous_features] = self.scaler.fit_transform(
                df[continuous_features]
            )
        else:
            df[continuous_features] = self.scaler.transform(
                df[continuous_features]
            )
        return df

    def preprocess(self):
        """Run the full preprocessing pipeline: encode then scale."""
        self.encoded = self.encode_categoricals(self.df)
        self.encoded = self.scale_continuous_features(
            self.encoded, self.continuous_features
        )

    def save(self, run_id=None):
        """Save scaler, encoder, and encoded data. Only valid in training mode."""
        if not self.training:
            raise ValueError("Cannot save in inference mode")

        # Save the scaler for future needs
        joblib.dump(
            self.scaler, proj_root + f"data/scaled/scaler_{self.run_id}.pkl"
        )
        # Save the encoder for future needs
        joblib.dump(
            self.encoder, proj_root + f"data/scaled/encoder_{self.run_id}.pkl"
        )
        # Save the encoded data for future needs
        joblib.dump(
            self.encoded,
            proj_root + f"data/scaled/encoded_data_{self.run_id}.pkl",
        )


# %%
if __name__ == "__main__":
    preprocessor = Preprocessor(filename=proj_root + "data/train.csv")
    preprocessor.preprocess()
    print("=== Encoded DataFrame")
    print(preprocessor.encoded)
    # preprocessor.encoded.to_csv(
    #     proj_root + "data/scaled/scaled_and_encoded_data.csv"
    # )
    preprocessor.save()
