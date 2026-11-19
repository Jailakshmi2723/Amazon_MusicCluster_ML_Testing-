import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def apply_pca():

    df = pd.read_csv(DATASET_PATH)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    return pca, pca_data
def test_pca_creates_two_components():

    pca, pca_data = apply_pca()

    print("\nPCA output shape:", pca_data.shape)
    print("Number of components:", pca.n_components_)

    assert pca_data.shape[1] == 2
    assert pca.n_components_ == 2
def test_pca_explained_variance():

    pca, pca_data = apply_pca()

    explained_variance = sum(pca.explained_variance_ratio_)

    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)

    print("Total explained variance:", explained_variance)
    print("Percentage:", explained_variance * 100)

    assert 0 < explained_variance <= 1
def test_pca_output_contains_no_invalid_values():

    pca, pca_data = apply_pca()

    print("\nChecking PCA output for invalid values")

    assert np.isfinite(pca_data).all()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_music.csv"

FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]