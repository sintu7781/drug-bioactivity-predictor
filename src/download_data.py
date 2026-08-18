import logging
from pathlib import Path

import pandas as pd
import requests

from config import RAW_DATA_DIR, TARGET_CHEMBL_ID

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

def fetch_activities(
    target_chembl_id: str,
    output_path: Path,
    page_size: int = 1000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    offset = 0
    
    while True:
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type": "IC50",
            "limit": page_size,
            "offset": offset,
        }
        
        LOGGER.info("Fetching records: offset=%s", offset)
        
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        
        payload = response.json()
        activities = payload.get("activities", [])
        
        if not activities:
            break
        
        rows.extend(activities)
        
        page_meta = payload.get("page_meta", {})
        total_count = page_meta.get("total_count", len(rows))
        
        LOGGER.info(
            "Fetched %s / %s records",
            len(rows),
            total_count,
        )
        
        if len(rows) >= total_count:
            break
        
        offset += page_size
        
    df = pd.DataFrame(rows)
    
    if df.empty:
        raise RuntimeError("No IC50 activity data was returned.")
    
    df.to_csv(output_path, index=False)
    
    LOGGER.info(
        "Saved %s records to %s",
        len(df),
        output_path,
    )
        
if __name__ == "__main__":
    from logging_config import configure_logging
    
    configure_logging(Path("logs/download.log"))
    
    output = RAW_DATA_DIR / "egfr_ic50_raw.csv"
    
    fetch_activities(
        target_chembl_id=TARGET_CHEMBL_ID,
        output_path=output,
    )