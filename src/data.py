import json
from pathlib import Path

from dotenv import load_dotenv
from edgar import Company, set_identity
import os

load_dotenv()
set_identity(os.environ["EDGAR_IDENTITY"])

TICKERS = ["BLK", "IVZ", "GOOGL", "IBM", "M", "W", "MS", "GS"]
OUT = Path("data/raw/filings")


def download(n_years: int = 2):
    OUT.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        for filing in Company(t).get_filings(form="10-K").head(n_years):
            meta = {"ticker": t, "filing_date": str(filing.filing_date),
                    "accession": filing.accession_no}
            text = filing.text()
            name = f"{t}_{meta['filing_date']}"
            (OUT / f"{name}.txt").write_text(text)
            (OUT / f"{name}.json").write_text(json.dumps(meta))
            print(f"saved {name}: {len(text):,} characters")


if __name__ == "__main__":
    download()
