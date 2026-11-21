"""Load a saved model and predict. Used by evaluation, predict.py and the notebook.

Long comments: the text is tokenised once and cut into overlapping windows of
`max_len` tokens (stride = max_len // 2). Each window is scored and the
comment gets the MAX probability, so a swear word at the end of a long review
is not lost by truncation.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import preprocess


class ProfanityClassifier:
    def __init__(self, model_dir, device=None):
        self.model_dir = Path(model_dir)
        self.cfg = json.loads((self.model_dir / "inference_config.json").read_text(encoding="utf-8"))
        self.threshold = float(self.cfg["threshold"])
        self.max_len = int(self.cfg["max_len"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, dtype=torch.float32).to(self.device).eval()

    def _windows(self, text):
        ids = self.tok(preprocess(text), add_special_tokens=False)["input_ids"]
        body = self.max_len - 2                      # room for <s> and </s>
        if len(ids) <= body:
            chunks = [ids]
        else:
            stride = body // 2
            chunks = [ids[i:i + body] for i in range(0, len(ids) - stride, stride)]
        # <s> ... </s>, exactly what tok(text, truncation=True) adds during training
        return [[self.tok.cls_token_id] + c + [self.tok.sep_token_id] for c in chunks]

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        texts = [texts] if isinstance(texts, str) else list(texts)
        owners, windows = [], []
        for i, t in enumerate(texts):
            for w in self._windows(t):
                owners.append(i)
                windows.append(w)
        probs = np.zeros(len(windows), dtype=np.float32)
        for s in range(0, len(windows), batch_size):
            batch = self.tok.pad({"input_ids": windows[s:s + batch_size]}, return_tensors="pt").to(self.device)
            logits = self.model(**batch).logits.float()
            probs[s:s + batch_size] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        out = np.zeros(len(texts), dtype=np.float32)
        np.maximum.at(out, np.array(owners), probs)
        return out

    def predict(self, texts):
        p = self.predict_proba(texts)
        return [{"label": int(x >= self.threshold), "probability": round(float(x), 4)} for x in p]


def load_classifier(model_dir, device=None):
    """Pick the right wrapper from inference_config.json ("arch")."""
    cfg = json.loads((Path(model_dir) / "inference_config.json").read_text(encoding="utf-8"))
    if cfg.get("arch") == "charcnn":
        from charcnn import CharCNNClassifier
        return CharCNNClassifier(model_dir, device)
    return ProfanityClassifier(model_dir, device)
