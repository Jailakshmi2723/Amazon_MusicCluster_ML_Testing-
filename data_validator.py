
import pandas as pd
from pathlib import Path
import numpy as np

FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]


def validate_dataset(file_path):

    file_path = Path(file_path)

    # Check whether dataset exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load dataset
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    # Check for missing features
    missing_features = set(FEATURES) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}"
        )

    # Check numerical data types and values
    for feature in FEATURES:

        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(
                f"Feature '{feature}' must contain numeric values"
            )

    if not np.isfinite(df[feature]).all():
        raise ValueError(
        f"Feature '{feature}' contains invalid values"
    )

    return df
