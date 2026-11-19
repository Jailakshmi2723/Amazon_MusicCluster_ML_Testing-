import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
def test_all_records_are_assigned_to_clusters():

    scaled_data, labels, model = train_kmeans()

    df = pd.read_csv(DATASET_PATH)

    print("\nTotal records:", len(df))
    print("Assigned records:", len(labels))

    assert len(labels) == len(df)
    assert not np.isnan(labels).any()

def test_cluster_distribution():

    scaled_data, labels, model = train_kmeans()

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    print("\nCluster distribution:")
    print(cluster_counts)

    assert len(cluster_counts) == 5
    assert (cluster_counts > 0).all()

def test_clustering_evaluation_metrics():

    scaled_data, labels, model = train_kmeans()

    silhouette = silhouette_score(
    scaled_data,
    labels,
    sample_size=5000,
    random_state=42
)
    db_index = davies_bouldin_score(scaled_data, labels)

    print("\nSilhouette Score:", silhouette)
    print("Davies-Bouldin Index:", db_index)

    assert np.isfinite(silhouette)
    assert np.isfinite(db_index)

    assert -1 <= silhouette <= 1
    assert db_index >= 0
def train_kmeans():

    df = pd.read_csv(DATASET_PATH)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_data)

    return scaled_data, labels, model
def test_kmeans_creates_five_clusters():

    scaled_data, labels, model = train_kmeans()

    unique_clusters = np.unique(labels)

    print("\nNumber of clusters:", len(unique_clusters))
    print("Cluster labels:", unique_clusters)

    assert len(unique_clusters) == 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_music.csv"

FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]