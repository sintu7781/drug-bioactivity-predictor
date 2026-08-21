from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    CURATED_DATA_FILE,
    FEATURE_FILE,
    MODEL_METADATA_FILE,
)
from .features import (
    featurize_smiles,
)


LOGGER = logging.getLogger(__name__)


def main() -> None:

    df = pd.read_csv(
        CURATED_DATA_FILE
    )

    features = []
    valid_indices = []

    for index, row in df.iterrows():
        
        try:
            
            vector = featurize_smiles(
                row["canonical_smiles"]
            )

            features.append(
                vector
            )
            valid_indices.append(
                index
            )

        except ValueError as exc:
            
            LOGGER.warning(
                f"Skipping %s: %s",
                index,
                exc,
            )
            
    if not features:
        
        raise RuntimeError(
            "No molecules could be featurized."
        )

    valid_df = (
        df.loc[
            valid_indices
            ]
        .reset_index(drop=True)
    )

    X = np.vstack(
        features
    )

    y = valid_df[
        "activity_label"
    ].to_numpy(
        dtype=np.int64
    )

    FEATURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    np.savez_compressed(
        FEATURE_FILE,
        X=X,
        y=y,
    )

    valid_df.to_csv(
        MODEL_METADATA_FILE,
        index=False,
    )

    print(
        f"Feature matrix: {X.shape}"
    )
    print(
        f"Target vector: {y.shape}"
    )


if __name__ == "__main__":
    main()
