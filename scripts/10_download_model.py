"""Download the published model so predict.py works without retraining.

  python scripts/10_download_model.py          # from https://huggingface.co/Yerdauit/kazakh-profanity-kazroberta
  python predict.py "Мына ойын керемет екен"
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Yerdauit/kazakh-profanity-kazroberta")
ap.add_argument("--out", default="models/kazroberta")
args = ap.parse_args()
path = snapshot_download(repo_id=args.repo, local_dir=ROOT / args.out)
print("model saved to", path)
