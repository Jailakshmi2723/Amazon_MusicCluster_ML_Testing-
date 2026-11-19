
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_music.csv"

FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


def test_scaled_features_have_zero_mean():

    df = pd.read_csv(DATASET_PATH)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    means = scaled_data.mean(axis=0)

    print("\nScaled feature means:")
    print(means)

    assert np.allclose(means, 0, atol=1e-6)


def test_scaled_features_have_unit_variance():

    df = pd.read_csv(DATASET_PATH)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    variances = scaled_data.var(axis=0)

    print("\nScaled feature variances:")
    print(variances)

    assert np.allclose(variances, 1, atol=1e-6)


def test_scaled_data_contains_no_invalid_values():

    df = pd.read_csv(DATASET_PATH)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    assert np.isfinite(scaled_data).all()
