# Amazon Music – Machine Learning Testing Framework

## Project Overview

This project demonstrates a Python-based testing framework developed to validate an academic Machine Learning project involving music clustering and recommendation exploration.

The original ML workflow uses audio features from an Amazon Music dataset to explore music similarities through K-Means clustering and Principal Component Analysis (PCA).

The testing framework focuses on validating dataset quality, preprocessing, clustering outputs, PCA results, model reproducibility, and pipeline behaviour.

## Project Objectives

- Validate dataset structure and data quality.
- Identify and analyse duplicate feature records.
- Verify feature scaling and preprocessing.
- Validate K-Means clustering outputs.
- Verify PCA dimensionality reduction.
- Test model reproducibility.
- Validate invalid data handling.
- Perform end-to-end ML pipeline integration testing.

## Machine Learning Workflow

1. Dataset loading and validation.
2. Feature selection using 10 numerical audio features.
3. Missing-value and invalid-data validation.
4. Feature scaling using StandardScaler.
5. K-Means clustering with 5 clusters.
6. PCA transformation into 2 dimensions.
7. Model evaluation using Silhouette Score and Davies–Bouldin Index.

## Technologies Used

- Python 3.14
- Pandas
- NumPy
- Scikit-learn
- Pytest
- Pytest-Cov
- VS Code

## Project Structure

```text
Amazon_Music_ML_Testing/
├── data/
│   ├── cleaned_music.csv
│   └── clustered_music.csv
├── tests/
│   ├── test_dataset.py
│   ├── test_duplicateanalysis.py
│   ├── test_duplicates.py
│   ├── test_evaluation.py
│   ├── test_invalid_data.py
│   ├── test_ml_pipeline.py
│   ├── test_model_reproducibility.py
│   ├── test_pca.py
│   └── test_preprocessing.py
├── data_validator.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Test Coverage

The framework currently contains 33 passing automated tests.

| Test Area | Tests |
|---|---:|
| Dataset validation | 5 |
| Duplicate analysis | 4 |
| Feature scaling | 3 |
| K-Means clustering | 4 |
| PCA validation | 3 |
| Model reproducibility | 1 |
| Invalid data testing | 4 |
| Data validator functional testing | 6 |
| ML pipeline integration | 3 |
| **Total** | **33** |

### Coverage Results

- Application module: `data_validator.py`
- Statements: 20
- Missing statements: 0
- Code coverage: 100%
- Test execution: 33 passed

*Coverage applies to the measured data validator module, not the entire original ML notebook.*

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Amazon_Music_ML_Testing
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Execute all tests

```bash
python -m pytest tests/ -v
```

### 6. Generate coverage report

```bash
python -m pytest tests/ --cov=data_validator --cov-report=term-missing --cov-report=html
```

## Key Results

- Dataset contains 95,837 records.
- Ten numerical audio features are used.
- K-Means clustering identifies five clusters.
- Original notebook Silhouette Score: 0.1864.
- Original notebook Davies–Bouldin Index: 1.6888.
- First two PCA components explain approximately 45.90% of the variance.
- All 33 automated tests currently pass.

## Scope and Limitations

This is a self-developed ML testing framework built around an academic ML clustering project.

The integration pipeline recreates the preprocessing and clustering workflow for testing. It does not load the original notebook's fitted model or scaler artifacts.

The current framework focuses on functional validation and reproducibility in the local environment. It is not a production ML monitoring or model-serving system.

## Future Enhancements

- Add model performance and data drift testing.
- Add boundary-value and property-based testing.
- Integrate automated test execution with GitHub Actions.
- Expand model validation and recommendation testing.

## Author

Machine Learning Testing | QA Automation | Python | Pytest