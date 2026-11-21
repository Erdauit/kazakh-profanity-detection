"""Step 0 - download the full KazSAnDRA corpus (source of the course dataset).

KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes,
ISSAI (Yeshpanov & Varol, 2024), https://github.com/IS2AI/KazSAnDRA
The score-classification files (imbalanced train + valid + test) together
contain all 180k reviews with their 1-5 star score and domain.

Run:  python scripts/00_download_data.py
"""
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "data/raw/kazsandra"
BASE = "https://raw.githubusercontent.com/IS2AI/KazSAnDRA/main/dataset/"
FILES = ["06_sc_train_ib.zip", "09_sc_valid.zip", "10_sc_test.zip"]

DST.mkdir(parents=True, exist_ok=True)
for name in FILES:
    zpath = DST / name
    if not zpath.exists():
        print("downloading", name)
        urllib.request.urlretrieve(BASE + name, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DST)
print("done:", sorted(p.name for p in DST.glob("*.csv")))
