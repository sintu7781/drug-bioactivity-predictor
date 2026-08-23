from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from .config import ACTIVE_THRESHOLD_NM

LOGGER = logging.getLogger(__name__)


def canonicalize_smiles(
    smiles: Any,
) -> str | None:
    """
    Validate and canonicalize a SMILES string.

    Returns:
        Canonical SMILES or None if invalid.
    """

    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
    )


def curate_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean ChEMBL activity data and create
    binary bioactivity labels.

    Classification rule:

        IC50 <= ACTIVE_THRESHOLD_NM
            -> ACTIVE (1)

        IC50 > ACTIVE_THRESHOLD_NM
            -> INACTIVE (0)
    """

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    required_columns = {
        "molecule_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
        "canonical_smiles",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data = df.copy()

    initial_count = len(data)

    # ----------------------------------------------------
    # 1. Standardize column types
    # ----------------------------------------------------

    data["standard_type"] = (
        data["standard_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    data["standard_units"] = (
        data["standard_units"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    data["molecule_chembl_id"] = (
        data["molecule_chembl_id"]
        .astype("string")
        .str.strip()
    )

    # ----------------------------------------------------
    # 2. Keep only IC50 measurements
    # ----------------------------------------------------

    data = data[
        data["standard_type"].eq(
            "IC50"
        )
    ].copy()

    after_ic50_filter = len(data)

    # ----------------------------------------------------
    # 3. Keep only nM measurements
    # ----------------------------------------------------

    data = data[
        data["standard_units"].eq(
            "nm"
        )
    ].copy()

    after_unit_filter = len(data)

    # ----------------------------------------------------
    # 4. Convert activity value to numeric
    # ----------------------------------------------------

    data["standard_value"] = (
        pd.to_numeric(
            data["standard_value"],
            errors="coerce",
        )
    )

    missing_activity = int(
        data["standard_value"]
        .isna()
        .sum()
    )

    data = data.dropna(
        subset=[
            "standard_value"
        ]
    ).copy()

    # Only positive IC50 values make sense.
    data = data[
        data["standard_value"] > 0
    ].copy()

    positive_activity_count = len(
        data
    )

    # ----------------------------------------------------
    # 5. Validate / canonicalize SMILES
    # ----------------------------------------------------

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
            "canonical_smiles"
        ]
    ).copy()

    # ----------------------------------------------------
    # 6. Remove duplicate raw measurements
    # ----------------------------------------------------

    before_duplicates = len(data)

    data = data.drop_duplicates(
        subset=[
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
        ]
    ).copy()

    duplicate_measurements = (
        before_duplicates
        - len(data)
    )

    # ----------------------------------------------------
    # 7. Aggregate multiple measurements
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # 8. Create binary activity labels
    # ----------------------------------------------------

    data["activity_label"] = (
        data["standard_value"]
        <= ACTIVE_THRESHOLD_NM
    ).astype("int8")

    data["activity_class"] = (
        data["activity_label"]
        .map(
            {
                0: "INACTIVE",
                1: "ACTIVE",
            }
        )
    )

    # ----------------------------------------------------
    # 9. Calculate pIC50
    # ----------------------------------------------------

    data["pIC50"] = (
        9.0
        - np.log10(
            data["standard_value"]
        )
    )

    # ----------------------------------------------------
    # 10. Sort and reset index
    # ----------------------------------------------------

    data = (
        data.sort_values(
            "molecule_chembl_id"
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------
    # 11. Keep only fields needed downstream
    # ----------------------------------------------------

    data = data[
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "pIC50",
            "activity_label",
            "activity_class",
        ]
    ].copy()

    # ----------------------------------------------------
    # 12. Quality report
    # ----------------------------------------------------
    
    active_count = int(
        (
            data["activity_label"]
            == 1
        ).sum()
    )

    inactive_count = int(
        (
            data["activity_label"]
            == 0
        ).sum()
    )

    final_count = len(data)

    quality = {
        "raw_records": initial_count,
        "after_ic50_filter": (
            after_ic50_filter
        ),
        "after_unit_filter": (
            after_unit_filter
        ),
        "missing_activity_removed": (
            missing_activity
        ),
        "positive_activity_records": (
            positive_activity_count
        ),
        "invalid_smiles_removed": (
            invalid_smiles
        ),
        "duplicate_measurements_removed": (
            duplicate_measurements
        ),
        "final_compounds": final_count,
        "active_compounds": active_count,
        "inactive_compounds": inactive_count,
        "active_fraction": (
            active_count / final_count
            if final_count > 0
            else 0.0
        ),
        "inactive_fraction": (
            inactive_count / final_count
            if final_count > 0
            else 0.0
        ),
        "active_threshold_nM": (
            ACTIVE_THRESHOLD_NM
        ),
    }

    LOGGER.info(
        "Curation completed: %s",
        quality,
    )

    if final_count == 0:
        raise ValueError(
            "No compounds remained after "
            "data curation."
        )

    return data, quality