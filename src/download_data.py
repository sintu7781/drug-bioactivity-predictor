from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RAW_ACTIVITY_FILE, TARGET_CHEMBL_ID
from .logging_config import configure_logging

LOGGER = logging.getLogger(__name__)

BASE_URL = (
    "https://www.ebi.ac.uk/"
    "chembl/api/data/activity"
)


def create_session() -> requests.Session:
    
    session = requests.Session()
    
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET"
        ],
        raise_on_status=False,
    )
    
    adapter = HTTPAdapter(
        max_retries=retry,
    )
    
    session.mount(
        "https://",
        adapter,
    )
    
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "drug-bioactivity-predictor/1.0 "
                "(educational research project)"
            ),
        }
    )
    
    return session


def fetch_page(
    session: requests.Session,
    target_chembl_id: str,
    offset: int = 0,
    limit: int = 100,
) -> dict:
    
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": "IC50",
        "limit": limit,
        "offset": offset,
    }
    
    LOGGER.info(
        "Fetching ChEMBL activity data: "
        "offset=%s limit=%s",
        offset,
        limit,
    )
    
    respone = session.get(
        BASE_URL,
        params=params,
        timeout=(15, 120),
    )
    
    LOGGER.info(
        "ChEMBL response status: %d",
        respone.status_code,
    )
    
    if not respone.ok:
        
        LOGGER.error(
            "ChEMBL response body:\n%s",
            respone.text[:3000],
        )
        
        respone.raise_for_status()
        
    return respone.json()

def fetch_activities(
    target_chembl_id: str,
    output_path: Path,
    page_size: int = 100,
) -> pd.DataFrame:
    
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    
    session = create_session()

    rows: list[dict] = []
    offset = 0

    while True:
        
        payload = fetch_page(
            session=session,
            target_chembl_id=target_chembl_id,
            offset=offset,
            limit=page_size,
        )
        
        activities = payload.get(
            "activities",
            [],
        )
        
        if not activities:
            break
        
        rows.extend(activities)
        
        page_meta = payload.get(
            "page_meta",
            {},
        )
        
        total_count = page_meta.get(
            "total_count"
        )
        
        LOGGER.info(
            "Collected %d records%s",
            len(rows),
            (
                f" / {total_count}"
                if total_count is not None
                else ""
            ),
        )
        
        if (
            total_count is not None
            and len(rows) >= total_count
        ):
            break
        
        if len(activities) < page_size:
            break

        offset += page_size
        
        # Be polite to the public service
        time.sleep(0.2)
        
    if not rows:
        
        raise RuntimeError(
            "No activity records returned for "
            f"{target_chembl_id}."
        )

    df = pd.DataFrame(rows)

    LOGGER.info(
        "Raw dataframe shape: %s",
        df.shape,
    )
    
    LOGGER.info(
        "Columns returned:\n%s",
        df.columns.to_list(),
    )
    
    required = {
        "molecule_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
    }
    
    missing = required.difference(
        df.columns
    )
    
    if missing:
        
        raise RuntimeError(
            "ChEMBL response is missing "
            f"required fields: {sorted(missing)}"
        )

    df.to_csv(
        output_path, 
        index=False,
    )

    LOGGER.info(
        "Raw dataset saved to %s",
        output_path,
    )
    
    return df
    
def main() -> None:
    
    configure_logging(
        Path("logs/download.log")
        )

    fetch_activities(
        target_chembl_id=TARGET_CHEMBL_ID,
        output_path=RAW_ACTIVITY_FILE,
    )


if __name__ == "__main__":
    main()