from pathlib import Path

import pandas as pd

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .logging_config import configure_logging
from .preprocessing import curate_data, save_curated_data


def main() -> None:
    configure_logging(Path("logs/curation.log"))

    input_path = RAW_DATA_DIR / "egfr_ic50_raw.csv"
    output_path = PROCESSED_DATA_DIR / "egfr_ic50_curated.csv"

    df = pd.read_csv(input_path)

    curated = curate_data(df)

    save_curated_data(curated, output_path)

    print(f"Saved curated dataset to: {output_path}")
    print(f"Rows: {len(curated)}")
    print(curated["activity_class"].value_counts())


if __name__ == "__main__":
    main()
