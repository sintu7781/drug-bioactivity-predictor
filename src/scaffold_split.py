from __future__ import annotations

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import (
    MurckoScaffold,
)


def scaffold_from_smiles(
    smiles: str,
) -> str:

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule
    )


def scaffold_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):

    data = df.copy()

    data["scaffold"] = (
        data["canonical_smiles"]
        .apply(
            scaffold_from_smiles
        )
    )

    scaffold_groups = (
        data.groupby(
            "scaffold"
        )
        .indices
    )

    groups = list(
        scaffold_groups.items()
    )

    rng = np.random.default_rng(
        random_state
    )

    rng.shuffle(groups)

    target_test_count = int(
        len(data) * test_size
    )

    test_indices = []

    for _, indices in groups:

        if (
            len(test_indices)
            >= target_test_count
        ):
            break

        test_indices.extend(
            indices
        )

    test_indices = set(
        test_indices
    )

    all_indices = set(
        range(len(data))
    )

    train_indices = (
        all_indices
        - test_indices
    )

    return (
        sorted(train_indices),
        sorted(test_indices),
    )