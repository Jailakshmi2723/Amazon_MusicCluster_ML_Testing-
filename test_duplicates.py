
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_music.csv"


def test_report_duplicate_records():

    df = pd.read_csv(DATASET_PATH)

    duplicate_count = df.duplicated().sum()

    print("\nTotal records:", len(df))
    print("Duplicate records:", duplicate_count)

    assert duplicate_count >= 0


def test_report_duplicate_percentage():

    df = pd.read_csv(DATASET_PATH)

    duplicate_count = df.duplicated().sum()

    duplicate_percentage = (
        duplicate_count / len(df)
    ) * 100

    print(
        "\nDuplicate percentage:",
        round(duplicate_percentage, 2),
        "%"
    )

    assert 0 <= duplicate_percentage <= 100
