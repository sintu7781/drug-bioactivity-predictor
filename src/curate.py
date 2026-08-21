from pathlib import Path

import pandas as pd

from .config import (
    CURATED_DATA_FILE,
    DATA_QUALITY_FILE,
    RAW_DATA_DIR,
)
from .logging_config import configure_logging
from .preprocessing import curate_data


def main() -> None:
    
    configure_logging(
        Path("logs/curation.log")
    )

    input_file = (
        RAW_DATA_DIR /
        "egfr_ic50_with_smiles.csv"
    )
    
    df = pd.read_csv(
        input_file
    )
    
    curated, quality = curate_data(
        df
    )

    CURATED_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    curated.to_csv(
        CURATED_DATA_FILE,
        index=False,
    )
    
    import json
    
    DATA_QUALITY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    DATA_QUALITY_FILE.write_text(
        json.dumps(
            quality,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved curated dataset: "
        f"{CURATED_DATA_FILE}"
    )
    
    print(
        json.dumps(
            quality,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
