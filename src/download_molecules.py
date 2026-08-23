from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .config import (
    RAW_ACTIVITY_FILE,
    RAW_DATA_DIR,
)


LOGGER = logging.getLogger(__name__)

MOLECULE_URL = (
    "https://www.ebi.ac.uk/"
    "chembl/api/data/molecule/"
)


def fetch_molecule(
    session: requests.Session,
    chembl_id: str,
) -> str | None:
    
    response = session.get(
        f"{MOLECULE_URL}{chembl_id}.json",
        timeout=(15, 60)
    )
    
    if not response.ok:
        LOGGER.warning(
            "Could not retrieve molecule %s: %s",
            chembl_id,
            response.status_code,
        )
        return None
    
    payload = response.json()
    
    structures = payload.get(
        "molecule_structures"
    )
    
    if not structures:
        return None
    
    return structures.get(
        "canonical_smiles"
    )
    
    
def main() -> None:
    
    from .download_data import (
        create_session
    )
    
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    df = pd.read_csv(
        RAW_ACTIVITY_FILE,
        low_memory=False,
    )
    
    if (
        "molecule_chembl_id"
        not in df.columns
    ):
        raise RuntimeError(
            "molecule_chembl_id is missing."
        )
    
    unique_ids = (
        df["molecule_chembl_id"]
        .dropna()
        .astype(str)
        .unique()
    )
    
    session = create_session()
    
    cache: dict[str, str | None] = {}
    
    for index, chembl_id in enumerate(
        unique_ids,
        start=1
    ):
        
        if chembl_id not in cache:
            
            cache[chembl_id] = (
                fetch_molecule(
                    session,
                    chembl_id,
                )
            )
            
            time.sleep(0.1)
            
        if index % 100 == 0:
            
            LOGGER.info(
                "Fetched structures: %d / %d",
                index,
                len(unique_ids),
            )
        
    df["canonical_smiles"] = (
        df["molecule_chembl_id"]
        .map(cache)
    )
    
    output = (
        RAW_DATA_DIR /
        "egfr_ic50_with_smiles.csv"
    )
    
    df.to_csv(
        output,
        index=False,
    )
    
    LOGGER.info(
        "Saved structure-enriched data to %s",
        output,
    )
    

if __name__ == "__main__":
    
    from .logging_config import (
        configure_logging
    )
    
    configure_logging(
        Path("logs/molecules.log")
    )
    
    main()