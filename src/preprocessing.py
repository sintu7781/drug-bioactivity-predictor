from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem

from .config import ACTIVE_THRESHOLD_NM

LOGGER = logging.getLogger(__name__)

def canonicalize_smiles(
    smiles: str,
) -> str | None:
    
    if not isinstance(smiles, str):
        return None
    
    smiles = smiles.strip()
    
    if not smiles:
        return None
    
    molecule = Chem.MolFromSmiles(
        smiles
    )
    
    if molecule is None:
        return None
    
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
    )

def curate_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    
    initial_count = len(df)
    
    data = df.copy()
    
    # Only nM.
    data = data[
        data["standard_units"]
        .astype(str)
        .str.lower()
        .eq("nM".lower())
    ].copy()
    
    data["standard_value"] = pd.to_numeric(
        data["standard_value"],
        errors="coerce",
    )
    
    missing_activity = int(
        data["standard_value"]
        .isna()
        .sum()
    )
    
    data = data.dropna(
        subset=[
            "standard_value",
        ]
    )
    
    data = data[
        data["standard_units"] > 0
    ].copy()
    
    # Structure handling.
    if "canonical_smiles" not in data.columns:
        
        raise ValueError(
            "canonical_smiles is missing. "
            "Retrieve molecule structures "
            "from ChEMBL before curattion."
        )
        
    before_smiles = len(data)
    
    data["canonical_smiles"] = (
        data["canonical_smiles"]
        .apply(canonicalize_smiles)
    )
    
    invalid_smiles = int(
        data["canonical_smiles"]
        .isna()
        .sum()
    )
    
    data = data.dropna(
        subset=[
            "canonical_smiles",
        ]
    )
    
    # Remove duplicate measurements.
    before_duplicates = len(data)
    
    data = data.drop_duplicates(
        subset=[
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
        ]
    )
    
    duplicate_measurements = (
        before_duplicates - len(data)
    )
    
    # Aggregate multiple measurements
    # for the same molecule.
    data = (
        data.groupby(
            [
                "molecule_chembl_id",
                "canonical_smiles",
            ],
            as_index=False,
        )
        .agg(
            standard_value=(
                "standard_value",
                "median",
            )
        )
    )
    
    # Binary classification.
    data["activity_label"] = (
        data["standard_value"]
        <= ACTIVE_THRESHOLD_NM
    ).astype(int)
    
    data["activity_class"] = (
        data["activity_label"]
        .map(
            {
                0: "INACTIVE",
                1: "ACTIVE",
            }
        )
    )
    
    data["pIC50"] = (
        9.0
        - __import__("numpy")
        .log10(
            data["standard_value"]
        )
    )
    
    quality = {
        "raw_recrds": initial_count,
        "after_ic50_filter": int(
            before_smiles
        ),
        "missing_activity_removed": (
            missing_activity
        ),
        "invalid_smiles_removed": (
            invalid_smiles
        ),
        "duplicate_measurements_removed": (
            duplicate_measurements
        ),
        "final_compounds": len(data),
        "active_compounds": int(
            data["activity_label"].sum()
        ),
        "inactive_compounds": int(
            (data["activity_label"] == 0)
            .sum()
        ),
    }
    
    LOGGER.info(
        "Curation quality: %s",
        quality,
    )
    
    return data, quality
