import sys
import pandas as pd
import numpy as np
import pytest

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_validator import validate_dataset

DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_music.csv"

FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]


def run_ml_pipeline(dataset_path=DATASET_PATH):
    df = validate_dataset(dataset_path)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)

    return df, scaled_data, labels, model
def test_complete_ml_pipeline():

    df, scaled_data, labels, model = run_ml_pipeline()

    print("\nOriginal dataset shape:", df.shape)
    print("Scaled data shape:", scaled_data.shape)
    print("Number of cluster assignments:", len(labels))
    print("Unique clusters:", np.unique(labels))

    assert len(df) == 95837
    assert scaled_data.shape == (95837, 10)
    assert len(labels) == 95837
    assert len(np.unique(labels)) == 5
def test_pipeline_outputs_are_valid():

    df, scaled_data, labels, model = run_ml_pipeline()

    print("\nChecking scaled data...")
    print("All scaled values finite:", np.isfinite(scaled_data).all())

    print("Checking cluster labels...")
    print("All labels valid:", np.isin(labels, [0, 1, 2, 3, 4]).all())

    assert np.isfinite(scaled_data).all()
    assert np.isin(labels, [0, 1, 2, 3, 4]).all()
    assert not pd.isna(labels).any()
def test_pipeline_rejects_missing_dataset(tmp_path):

    missing_file = tmp_path / "missing_music.csv"

    with pytest.raises(FileNotFoundError):
        run_ml_pipeline(missing_file)