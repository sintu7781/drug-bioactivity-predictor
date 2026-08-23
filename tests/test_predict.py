from pathlib import Path

import pytest

from src.predict import BioactivityPredictor

MODEL_PATH = Path(
    "models/bioactivity_model.joblib"
)


@pytest.fixture
def predictor():
    return BioactivityPredictor(
        MODEL_PATH
    )


def test_model_loads(predictor):

    assert predictor.model_name == (
        "random_forest"
    )

    assert predictor.feature_count == (
        2056
    )


def test_valid_smiles(predictor):

    result = predictor.predict(
        "CCO"
    )

    assert result["prediction"] in {
        "ACTIVE",
        "INACTIVE",
    }


def test_probabilities_sum_to_one(
    predictor,
):

    result = predictor.predict(
        "CCO"
    )

    total = (
        result["active_probability"]
        + result["inactive_probability"]
    )

    assert total == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_canonical_smiles_exists(
    predictor,
):

    result = predictor.predict(
        "C(C)O"
    )

    assert result[
        "canonical_smiles"
    ] == "CCO"


def test_invalid_smiles(
    predictor,
):

    with pytest.raises(
        ValueError
    ):

        predictor.predict(
            "this-is-not-a-smiles"
        )


def test_empty_smiles(
    predictor,
):

    with pytest.raises(
        ValueError
    ):

        predictor.predict(
            ""
        )


def test_descriptors(
    predictor,
):

    result = predictor.predict(
        "CCO"
    )

    descriptors = result[
        "descriptors"
    ]

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

    assert set(
        descriptors.keys()
    ) == expected