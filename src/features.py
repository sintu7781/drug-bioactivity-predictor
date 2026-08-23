from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
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
    rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_N_BITS,
    )
)


def smiles_to_mol(
    smiles: str,
) -> Chem.Mol:
    """
    Convert a SMILES string into an RDKit molecule.

    Raises:
        ValueError: If the SMILES is invalid.
    """

    if not isinstance(
        smiles,
        str,
    ):
        raise TypeError(
            "SMILES must be a string."
        )

    smiles = smiles.strip()

    if not smiles:
        raise ValueError(
            "SMILES cannot be empty."
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
    molecule: Chem.Mol,
) -> dict[str, float]:
    """
    Calculate the eight molecular descriptors
    used by the trained model.
    """

    return {
        "MolWt": float(
            Descriptors.MolWt(
                molecule
            )
        ),
        "LogP": float(
            Crippen.MolLogP(
                molecule
            )
        ),
        "HBD": float(
            Lipinski.NumHDonors(
                molecule
            )
        ),
        "HBA": float(
            Lipinski.NumHAcceptors(
                molecule
            )
        ),
        "RotatableBonds": float(
            Lipinski.NumRotatableBonds(
                molecule
            )
        ),
        "TPSA": float(
            Descriptors.TPSA(
                molecule
            )
        ),
        "RingCount": float(
            Lipinski.RingCount(
                molecule
            )
        ),
        "HeavyAtomCount": float(
            Lipinski.HeavyAtomCount(
                molecule
            )
        ),
    }


def generate_morgan_fingerprint(
    molecule: Chem.Mol,
) -> np.ndarray:
    """
    Generate the 2048-bit Morgan fingerprint
    used by the trained model.

    Configuration:
        radius = 2
        bits = 2048
    """

    fingerprint = (
        MORGAN_GENERATOR.GetFingerprint(
            molecule
        )
    )

    array = np.zeros(
        MORGAN_N_BITS,
        dtype=np.uint8,
    )

    DataStructs.ConvertToNumpyArray(
        fingerprint,
        array,
    )

    return array


def featurize_smiles(
    smiles: str,
) -> np.ndarray:
    """
    Convert SMILES into the complete
    2056-dimensional model feature vector.

    8 molecular descriptors
    +
    2048 Morgan fingerprint bits
    =
    2056 features
    """

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
        generate_morgan_fingerprint(
            molecule
        ).astype(
            np.float32
        )
    )

    features = np.concatenate(
        [
            descriptor_vector,
            fingerprint_vector,
        ]
    )

    if features.shape[0] != 2056:
        raise RuntimeError(
            "Unexpected feature count: "
            f"{features.shape[0]}. "
            "Expected 2056."
        )

    return features


def feature_names() -> list[str]:
    """
    Return names for all 2056 features.
    """

    descriptor_features = (
        DESCRIPTOR_NAMES.copy()
    )

    fingerprint_features = [
        f"morgan_{i}"
        for i in range(
            MORGAN_N_BITS
        )
    ]

    return (
        descriptor_features
        + fingerprint_features
    )