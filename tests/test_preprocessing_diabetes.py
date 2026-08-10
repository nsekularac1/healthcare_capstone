from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.diabetes import prepare_diabetes_data


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "diabetes"
    / "diabetic_data.csv"
)


def test_prepare_diabetes_data_basic():
    raw_df = pd.read_csv(DATA_PATH)

    prepared = prepare_diabetes_data(raw_df)

    assert len(prepared.X) == len(prepared.y)
    assert len(prepared.X) == len(prepared.groups)

    assert prepared.X.shape[0] > 0
    assert prepared.X.shape[1] > 0

    assert prepared.y.isna().sum() == 0
    assert prepared.groups.isna().sum() == 0

    assert "patient_nbr" not in prepared.X.columns
    assert "encounter_id" not in prepared.X.columns

    assert set(prepared.y.unique()).issubset({0, 1})
