"""Step 9 - publish the trained model folder to the Hugging Face Hub.

The weights (~330 MB) are too large for GitHub. Log in once with your own
token (`hf auth login`), then:

  python scripts/09_upload_model.py --repo <your-hf-username>/kazakh-profanity-kazroberta

Uploads everything needed for inference: model.safetensors, config.json,
tokenizer files, inference_config.json (threshold, max_len, preprocessing),
moderation.json (two-threshold policy), history.json and a model card.
Anyone can then run scripts/10_download_model.py and predict.py.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--model", default="models/kazroberta")
ap.add_argument("--github", default="https://github.com/Erdauit/kazakh-profanity-detection")
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

folder = ROOT / args.model
cfg = json.loads((folder / "inference_config.json").read_text(encoding="utf-8"))
mod_path = folder / "moderation.json"
mod = json.loads(mod_path.read_text(encoding="utf-8")) if mod_path.exists() else None
m = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
t, to, ce = m["test"], m["test_obfuscated"], m["clean_eval"]
cm = t["confusion_matrix"]
market = m["clean_eval_by_domain"].get("market", {})

mod_md = ""
if mod:
    mt = mod["test"]
    mod_md = f"""
## Two-threshold moderation (from `moderation.json`)

| probability | action | on test |
|---|---|---|
| ≥ {mod['t_hide']} | auto-hide | {mt['profane_auto_hidden_%']}% of profane, {mt['clean_auto_hidden_(wrongly)_%']}% of clean comments |
| {mod['t_review']} – {mod['t_hide']} | send to a human | {mt['profane_to_human_%']}% of profane, {mt['clean_to_human_%']}% of clean |
| < {mod['t_review']} | publish | {mt['profane_published_(missed)_%']}% of profane slip through |

Rule: `{mod['rule']}` (both chosen on validation).
"""

card = f"""---
language: [kk, ru]
license: apache-2.0
base_model: {cfg['model_name']}
pipeline_tag: text-classification
tags: [profanity, toxicity, content-moderation, kazakh]
---
# Kazakh profanity classifier (Kaz-RoBERTa, fine-tuned)

Binary classifier for Kazakh and mixed Kazakh-Russian product / app reviews:
**`1` = contains profanity** (Kazakh obscene words, Russian mat inside Kazakh text,
vulgar insults aimed at someone), **`0` = no profanity** (including negative but clean reviews).

- Code, data, labeling rules and full evaluation: {args.github}
- Base model: [`{cfg['model_name']}`](https://huggingface.co/{cfg['model_name']})
  ({cfg['n_params'] / 1e6:.0f}M parameters), fine-tuned {cfg['train']['epochs']} epochs, best epoch {cfg['best_epoch']}
- Decision threshold: **{cfg['threshold']}** (max F1 on validation; stored in `inference_config.json`)

## Results (held-out test, n = {t['n']}, {t['n_profane']} profane)

| metric (class 1) | value |
|---|---|
| precision | {t['precision']} |
| recall | {t['recall']} |
| F1 | {t['f1']} |
| confusion matrix | TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']} · TP {cm['tp']} |
| F1 on obfuscated test (`*` masks, Latin script, keyboard swaps, stretching) | {to['f1']} |
| false-positive rate on {ce['n']} ordinary reviews | {ce['false_positive_rate']} (Kaspi product reviews: {market.get('confusion_matrix', {}).get('fp', '?')} / {market.get('n', '?')}) |
{mod_md}
## Usage

Recommended: clone the GitHub repository, then

```bash
python scripts/10_download_model.py
python predict.py "Мына ойын керемет екен"
```

`predict.py` applies the training preprocessing (anonymisation, lower-case, collapse 3+ repeated
characters), sliding windows over long texts, the threshold and the moderation policy, and shows
which words drove the prediction.

Minimal use with `transformers` only (short texts, without anonymisation):

```python
import re, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "{args.repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

text = re.sub(r"(.)\\1{{2,}}", r"\\1\\1", "Қосымша өте жақсы, рахмет!".lower())
with torch.no_grad():
    p = torch.softmax(model(**tok(text, truncation=True, max_length={cfg['max_len']}, return_tensors="pt")).logits, -1)[0, 1].item()
print(round(p, 3), int(p >= {cfg['threshold']}))
```

## Limitations

- Trained on product / app / place / book reviews (KazSAnDRA); other domains (social media, chats) are untested.
- Weaker on digit / symbol masking (`қ0т@қ`) and Latin-script Kazakh than on plain Cyrillic.
- Sometimes reacts to Russian colloquial words in otherwise clean comments; uncertain cases should go to a human.
- Labels for the extended data are semi-automatic (lexicon + model agreement, disagreements reviewed by hand).
"""
(folder / "README.md").write_text(card, encoding="utf-8")

api = HfApi()
api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model",
                  allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "README.md"])
print(f"uploaded -> https://huggingface.co/{args.repo}")
