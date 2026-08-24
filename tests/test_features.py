import numpy as np
import pytest

from src.features import (
    calculate_descriptors,
    calculate_morgan_fingerprint,
    featurize_smiles,
    smiles_to_mol,
)


def test_valid_smiles():

    molecule = smiles_to_mol(
        "CCO"
    )

    assert molecule is not None


def test_invalid_smiles():

    with pytest.raises(
        ValueError
    ):
        smiles_to_mol(
            "not-a-valid-smiles"
        )


def test_descriptors():

    molecule = smiles_to_mol(
        "CCO"
    )

    descriptors = (
        calculate_descriptors(
            molecule
        )
    )

    assert descriptors["MolWt"] > 0
    assert "LogP" in descriptors
    assert "HBD" in descriptors
    assert "HBA" in descriptors


def test_morgan_fingerprint():

    molecule = smiles_to_mol(
        "CCO"
    )

    fingerprint = (
        calculate_morgan_fingerprint(
            molecule
        )
    )

    assert isinstance(
        fingerprint,
        np.ndarray,
    )

    assert fingerprint.shape == (2048,)

def test_full_feature_vector():
    
    vector = featurize_smiles(
        "CCO"
    )
    
    assert vector.shape == (
        2056,
    )
    
    assert np.isfinite(
        vector
    ).all()