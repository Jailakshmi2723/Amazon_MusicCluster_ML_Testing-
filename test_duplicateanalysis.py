
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_music.csv"


def test_analyse_duplicate_records():

    df = pd.read_csv(DATASET_PATH)

    duplicates = df[df.duplicated(keep=False)]

    print("\nDuplicate rows:", len(duplicates))

    print("\nSample duplicate records:")
    print(duplicates.head(10))

    print("\nDuplicate feature groups:")
    print(
        duplicates.groupby(
            df.columns.tolist()
        ).size().head(10)
    )

    assert len(duplicates) >= 0

def test_count_unique_duplicate_groups():

    df = pd.read_csv(DATASET_PATH)

    duplicate_groups = df[
        df.duplicated(keep=False)
    ].drop_duplicates()

    print(
        "\nUnique duplicated feature combinations:",
        len(duplicate_groups)
    )

    print(
        "\nTotal duplicate rows:",
        df.duplicated(keep=False).sum()
    )

    assert len(duplicate_groups) > 0
