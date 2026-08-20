from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .config import ACTIVE_THRESHOLD_NM

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value",
    "standard_units",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False

    molecule = Chem.MolFromSmiles(smiles)

    return molecule is not None


def curate_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    data = df.copy()

    # Keep only IC50 measurements in nM.
    data = data[data["standard_units"].astype(str).str.lower().eq("nm")].copy()

    # Numeric activity values.
    data["standard_value"] = pd.to_numeric(
        data["standard_value"],
        errors="coerce",
    )

    data = data.dropna(subset=["standard_value", "canonical_smiles"])

    # Positive IC50 values only.
    data = data[data["standard_value"] > 0]

    # Validate SMILES.
    data["valid_smiles"] = data["canonical_smiles"].apply(validate_smiles)

    data = data[data["valid_smiles"]].copy()

    # Normalize SMILES using RDKit.
    data["canonical_smiles"] = data["canonical_smiles"].apply(
        lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    )

    # Remove duplicate molecule/activity pairs.
    data = data.drop_duplicates(
        subset=[
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
        ]
    )

    # If multiple measurements exists for the same molecule, use median IC50.
    data = data.groupby(
        ["molecule_chembl_id", "canonical_smiles"],
        as_index=False,
    )["standard_value"].median()

    data["activity_label"] = (data["standard_value"] <= ACTIVE_THRESHOLD_NM).astype(int)

    data["activity_class"] = data["activity_label"].map(
        {
            0: "INACTIVE",
            1: "ACTIVE",
        }
    )

    LOGGER.info("Curated dataset shape: %s", data.shape)

    return data


def save_curated_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_path, index=False)
