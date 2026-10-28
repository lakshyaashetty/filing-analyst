"""Search the filings for a phrase and show context. Usage: python -m src.find "phrase" [TICKER]"""
import re
import sys
from pathlib import Path

term = sys.argv[1]
ticker = sys.argv[2] if len(sys.argv) > 2 else ""
for f in sorted(Path("data/raw/filings").glob(f"{ticker}*.txt")):
    text = " ".join(f.read_text().split())
    for m in list(re.finditer(re.escape(term), text, re.I))[:3]:
        print(f"--- {f.stem}\n...{text[max(0, m.start() - 250):m.end() + 250]}...\n")
