
import pandas as pd
from pathlib import Path
import pytest


FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]


def test_missing_feature_detection():

    df = pd.DataFrame({
        "duration_ms": [200000, 180000],
        "danceability": [0.7, 0.5],
        "energy": [0.8, 0.6]
    })

    missing_features = set(FEATURES) - set(df.columns)

    print("\nMissing features:")
    print(missing_features)

    assert len(missing_features) == 7


def test_invalid_feature_data_type():

    df = pd.DataFrame({
        "duration_ms": [200000, 180000],
        "danceability": [0.7, 0.5],
        "energy": [0.8, "high"]
    })

    print("\nEnergy column dtype:", df["energy"].dtype)
    print("Energy column values:", df["energy"].tolist())

    assert not pd.api.types.is_numeric_dtype(df["energy"])
def test_missing_dataset_file():

    invalid_path = Path("data/non_existing_music.csv")

    print("\nChecking dataset path:", invalid_path)

    assert not invalid_path.exists()
def test_missing_file_raises_error():

    invalid_path = Path("data/non_existing_music.csv")

    with pytest.raises(FileNotFoundError):
        pd.read_csv(invalid_path)