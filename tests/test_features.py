import numpy as np

from src.features import (
    calculate_descriptors,
    caculate_morgan_fingerprint,
    smiles_to_mol,
)


def test_valid_smiles():
    
    molecule = smiles_to_mol(
        "CCO"
    )
    
    assert molecule is not None
    
def test_invalid_smiles():
    
    try:
        smiles_to_mol(
            "invalid_smiles"
        )
        
        assert False
    
    except ValueError:
        assert True
        
    
def test_descriptors():
    
    molecule = smiles_to_mol(
        "CCO"
    )
    
    descriptors = calculate_descriptors(
        molecule
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
        caculate_morgan_fingerprint(
            molecule
        )
    )
    
    assert isinstance(
        fingerprint,
        np.ndarray,
    )
    
    assert fingerprint.shape == (
        2048,
    )