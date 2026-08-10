from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pamap2 import (
    prepare_pamap2_data,
    split_and_scale_pamap2,
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pamap2"
)


def test_prepare_pamap2_data_basic():
    prepared = prepare_pamap2_data(DATA_DIR)

    assert prepared.X.ndim == 3
    assert prepared.X.shape[1] == 100
    assert prepared.X.shape[2] == 19

    assert len(prepared.X) == len(prepared.y)
    assert len(prepared.X) == len(prepared.groups)

    assert np.isnan(prepared.X).sum() == 0
    assert len(prepared.class_names) == 12


def test_pamap2_subject_split_has_no_overlap():
    prepared = prepare_pamap2_data(DATA_DIR)

    split = split_and_scale_pamap2(
        prepared,
        test_size=0.22,
        random_state=42,
    )

    train_subjects = set(split.groups_train)
    test_subjects = set(split.groups_test)

    assert train_subjects.isdisjoint(test_subjects)
    assert np.isnan(split.X_train).sum() == 0
    assert np.isnan(split.X_test).sum() == 0
