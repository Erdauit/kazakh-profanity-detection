"""Step 9 - publish the trained model folder to the Hugging Face Hub.

The weights (~1.1 GB) are too large for GitHub. Log in once with your own
token (`hf auth login`), then:

  python scripts/09_upload_model.py --repo <your-hf-username>/kazakh-profanity-kazroberta

Uploads everything needed for inference: model.safetensors, config.json,
tokenizer files, inference_config.json (threshold, max_len, preprocessing),
history.json and a short model card. Anyone can then run
  python predict.py --model <downloaded folder>
or use scripts/10_download_model.py.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--model", default="models/kazroberta")
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

folder = ROOT / args.model
cfg = json.loads((folder / "inference_config.json").read_text(encoding="utf-8"))
metrics_path = ROOT / "results/metrics.json"
m = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
t = m.get("test", {})
card = f"""---
language: [kk, ru]
license: mit
base_model: {cfg['model_name']}
pipeline_tag: text-classification
tags: [profanity, content-moderation, kazakh]
---
# Kazakh profanity classifier (fine-tuned)

Binary classifier for Kazakh / mixed Kazakh-Russian product and app reviews:
`1` = contains profanity (obscene words or vulgar insults), `0` = no profanity.
Decision threshold (chosen on validation): **{cfg['threshold']}** — see `inference_config.json`.

Test (n={t.get('n', '?')}): precision {t.get('precision', '?')}, recall {t.get('recall', '?')}, F1 {t.get('f1', '?')}.

Use the repository's `predict.py` / `src/infer.py`, which apply the same preprocessing
(lower-case, collapse repeated characters, anonymisation) and sliding windows for long texts.
"""
(folder / "README.md").write_text(card, encoding="utf-8")

api = HfApi()
api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model",
                  allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "README.md"])
print(f"uploaded -> https://huggingface.co/{args.repo}")
