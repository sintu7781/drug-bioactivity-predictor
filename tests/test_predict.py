from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.predict import BioactivityPredictor

FEATURE_COUNT = 2056


@pytest.fixture
def test_model_path(tmp_path: Path) -> Path:
    """
    Create a lightweight Random Forest model artifact for testing.

    The production model is intentionally not committed to Git because
    it is stored separately on Hugging Face.
    """

    rng = np.random.default_rng(42)

    X = rng.random((40, FEATURE_COUNT), dtype=np.float32)
    y = np.array([0, 1] * 20)

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X, y)

    artifact = {
        "model": model,
        "model_name": "random_forest_test",
        "feature_count": FEATURE_COUNT,
        "target": "EGFR",
        "target_chembl_id": "CHEMBL203",
        "activity_type": "IC50",
        "activity_unit": "nM",
        "active_threshold_nM": 1000.0,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "descriptor_count": 8,
    }

    model_path = tmp_path / "bioactivity_model.joblib"

    joblib.dump(
        artifact,
        model_path,
    )

    return model_path


@pytest.fixture
def predictor(
    test_model_path: Path,
) -> BioactivityPredictor:
    """Return a predictor backed by the temporary test model."""

    return BioactivityPredictor(
        model_path=test_model_path,
    )


def test_model_loads(
    predictor: BioactivityPredictor,
) -> None:
    """The predictor should load the model artifact."""

    assert predictor.model_name == "random_forest_test"
    assert predictor.feature_count == FEATURE_COUNT


def test_predict_returns_valid_result(
    predictor: BioactivityPredictor,
) -> None:
    """A valid SMILES should produce a prediction."""

    result = predictor.predict("CCO")

    assert isinstance(result, dict)

    assert result["prediction"] in {
        "ACTIVE",
        "INACTIVE",
    }

    assert result["prediction_label"] in {
        0,
        1,
    }


def test_probabilities_sum_to_one(
    predictor: BioactivityPredictor,
) -> None:
    """Prediction probabilities should sum to approximately one."""

    result = predictor.predict("CCO")

    total = (
        result["active_probability"]
        + result["inactive_probability"]
    )

    assert total == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_canonical_smiles_exists(
    predictor: BioactivityPredictor,
) -> None:
    """Canonical SMILES should be returned."""

    result = predictor.predict("CCO")

    assert result["canonical_smiles"]
    assert isinstance(
        result["canonical_smiles"],
        str,
    )


def test_invalid_smiles(
    predictor: BioactivityPredictor,
) -> None:
    """Invalid SMILES should raise ValueError."""

    with pytest.raises(ValueError):
        predictor.predict(
            "this-is-not-valid-smiles"
        )


def test_empty_smiles(
    predictor: BioactivityPredictor,
) -> None:
    """Empty SMILES should raise ValueError."""

    with pytest.raises(ValueError):
        predictor.predict("")


def test_descriptors(
    predictor: BioactivityPredictor,
) -> None:
    """Prediction should contain molecular descriptors."""

    result = predictor.predict("CCO")

    descriptors = result["descriptors"]

    expected = {
        "MolWt",
        "LogP",
        "HBD",
        "HBA",
        "RotatableBonds",
        "TPSA",
        "RingCount",
        "HeavyAtomCount",
    }

    assert set(descriptors.keys()) == expected