from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import (
    Crippen, 
    Descriptors, 
    Lipinski, 
    rdFingerprintGenerator
)


DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "HBD",
    "HBA",
    "RotatableBonds",
    "TPSA",
    "RingCount",
    "HeavyAtomCount",
]


MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048


MORGAN_GENERATOR = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_N_BITS,
    )
)


def smiles_to_mol(smiles: str):
    
    if not isinstance(
        smiles,
        str,
    ):
        
        raise ValueError(
            "SMILES must be a string."
        )
        
    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    return molecule


def calculate_descriptors(
    molecule
) -> dict[str, float]:
    
    return {
        "MolWt": float(
            Descriptors.MolWt(molecule)
        ),
        "LogP": float(
            Crippen.MolLogP(molecule)
        ),
        "HBD": float(
            Lipinski.NumHDonors(molecule)
        ),
        "HBA": float(
            Lipinski.NumHAcceptors(molecule)
        ),
        "RotatableBonds": float(
            Lipinski.NumRotatableBonds(
                molecule
            )
        ),
        "TPSA": float(
            Descriptors.TPSA(molecule)
        ),
        "RingCount": float(
            Lipinski.RingCount(molecule)
        ),
        "HeavyAtomCount": float(
            Lipinski.HeavyAtomCount(
                molecule
            )
        ),
    }



def caculate_morgan_fingerprint(
    molecule,
) -> np.ndarray:
    
    fingerprint = (
        MORGAN_GENERATOR.GetFingerprint(
            molecule
        )
    )

    array = np.zeros(
        MORGAN_N_BITS,
        dtype=np.uint8
    )

    DataStructs.ConvertToNumpyArray(
        fingerprint,
        array,
    )

    return array


def featurize_smiles(
    smiles: str
) -> np.ndarray:

    molecule = smiles_to_mol(
        smiles
    )

    descriptors = calculate_descriptors(
        molecule
    )

    descriptor_vector = np.array(
        [
            descriptors[name] 
            for name in DESCRIPTOR_NAMES
        ],
        dtype=np.float32,
    )

    fingerprint_vector = (
        caculate_morgan_fingerprint(
            molecule
        ).astype(np.float32))

    return np.concatenate(
        [
            descriptor_vector,
            fingerprint_vector,
        ]
    )
    

def feature_names() -> list[str]:
    
    descriptor_features = (
        DESCRIPTOR_NAMES
    )
    
    fingerprint_features = [
        f"morgan_{i}"
        for i in range(
            MORGAN_N_BITS
        )
    ]
    
    return  (
        descriptor_features
        + fingerprint_features
    )
