
import sys
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from data_validator import validate_dataset


# Test 1: Valid dataset
def test_valid_dataset():

    dataset_path = PROJECT_ROOT / "data" / "cleaned_music.csv"

    df = validate_dataset(dataset_path)

    print("\nValidated dataset shape:", df.shape)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 95837


# Test 2: Reject missing features
def test_validator_rejects_missing_features(tmp_path):

    df = pd.DataFrame({
        "duration_ms": [200000],
        "danceability": [0.7],
        "energy": [0.8]
    })

    invalid_file = tmp_path / "missing_features.csv"
    df.to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="Missing features"):
        validate_dataset(invalid_file)


# Test 3: Reject non-numeric values
def test_validator_rejects_non_numeric_values(tmp_path):

    invalid_df = pd.DataFrame({
        "duration_ms": [200000],
        "danceability": [0.7],
        "energy": ["high"],
        "loudness": [-5.0],
        "speechiness": [0.05],
        "acousticness": [0.2],
        "instrumentalness": [0.0],
        "liveness": [0.1],
        "valence": [0.8],
        "tempo": [120.0]
    })

    invalid_file = tmp_path / "invalid_music.csv"
    invalid_df.to_csv(invalid_file, index=False)

    with pytest.raises(TypeError, match="must contain numeric values"):
        validate_dataset(invalid_file)


# Test 4: Reject missing values
def test_validator_rejects_missing_values(tmp_path):

    invalid_df = pd.DataFrame({
        "duration_ms": [200000],
        "danceability": [0.7],
        "energy": [0.8],
        "loudness": [-5.0],
        "speechiness": [0.05],
        "acousticness": [0.2],
        "instrumentalness": [0.0],
        "liveness": [0.1],
        "valence": [0.8],
        "tempo": [float("nan")]
    })

    invalid_file = tmp_path / "missing_values.csv"
    invalid_df.to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="contains invalid values"):
        validate_dataset(invalid_file)
def test_validator_rejects_infinite_values(tmp_path):

    invalid_df = pd.DataFrame({
        "duration_ms": [200000],
        "danceability": [0.7],
        "energy": [0.8],
        "loudness": [-5.0],
        "speechiness": [0.05],
        "acousticness": [0.2],
        "instrumentalness": [0.0],
        "liveness": [0.1],
        "valence": [0.8],
        "tempo": [float("inf")]
    })

    invalid_file = tmp_path / "infinite_values.csv"
    invalid_df.to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="contains invalid values"):
        validate_dataset(invalid_file)
    
def test_validator_rejects_empty_dataset(tmp_path):

    empty_df = pd.DataFrame(columns=[
        "duration_ms", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ])

    empty_file = tmp_path / "empty_music.csv"

    empty_df.to_csv(empty_file, index=False)

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset(empty_file)