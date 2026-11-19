import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
def train_model():

    df = pd.read_csv(DATASET_PATH)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_data)

    return labels
def test_kmeans_reproducibility():

    labels_run1 = train_model()
    labels_run2 = train_model()

    print("\nFirst run cluster counts:")
    print(pd.Series(labels_run1).value_counts().sort_index())

    print("\nSecond run cluster counts:")
    print(pd.Series(labels_run2).value_counts().sort_index())

    assert np.array_equal(labels_run1, labels_run2)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_music.csv"

FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]