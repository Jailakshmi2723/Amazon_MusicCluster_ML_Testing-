

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "cleaned_music.csv"
)

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


def test_dataset_loads_successfully():
    df = pd.read_csv(DATASET_PATH)

    assert not df.empty


def test_dataset_has_expected_columns():
    df = pd.read_csv(DATASET_PATH)

    for column in FEATURES:
        assert column in df.columns


def test_dataset_has_10_features():
    assert len(FEATURES) == 10


def test_dataset_features_are_numeric():
    df = pd.read_csv(DATASET_PATH)

    assert all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in FEATURES
    )


def test_dataset_has_no_missing_values():
    df = pd.read_csv(DATASET_PATH)

    assert not df[FEATURES].isnull().any().any()

