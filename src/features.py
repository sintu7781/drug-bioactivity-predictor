from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator

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


def smiles_to_mol(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return molecule


def calculate_descriptors(molecule) -> dict[str, float]:
    return {
        "MolWt": Descriptors.MolWt(molecule),
        "LogP": Crippen.MolLogP(molecule),
        "HBD": Lipinski.NumHDonors(molecule),
        "HBA": Lipinski.NumHAcceptors(molecule),
        "RotatableBonds": Lipinski.NumRotatableBonds(molecule),
        "TPSA": Descriptors.TPSA(molecule),
        "RingCount": Lipinski.RingCount(molecule),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(molecule),
    }


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def caculate_morgan_fingerprint(
    molecule,
) -> np.ndarray:
    fingerprint = MORGAN_GENERATOR.GetFingerprint(molecule)

    array = np.zeros((2048,), dtype=np.uint8)

    # RDKit writes the bit vector into this array.
    from rdkit import DataStructs

    DataStructs.ConvertToNumpyArray(
        fingerprint,
        array,
    )

    return array


def featurize_smiles(smiles: str) -> np.ndarray:

    molecule = smiles_to_mol(smiles)

    descriptors = calculate_descriptors(molecule)

    descriptor_vector = np.array(
        [descriptors[name] for name in DESCRIPTOR_NAMES],
        dtype=np.float32,
    )

    fingerprint_vector = caculate_morgan_fingerprint(molecule).astype(np.float32)

    return np.concatenate(
        [
            descriptor_vector,
            fingerprint_vector,
        ]
    )
