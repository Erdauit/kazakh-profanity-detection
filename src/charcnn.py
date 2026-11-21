"""Character-level CNN trained from scratch (no pre-training).

Text -> preprocess() -> sequence of characters -> ids from a character
vocabulary built on the TRAIN split only (chars seen < 2 times -> <unk>).
Architecture (Kim-style CNN over characters):
  Embedding(V, 64) -> Conv1d with kernel widths 2,3,4,5,6 (128 filters each, ReLU)
  -> global max-pool over time -> concat (640) -> dropout 0.3 -> Linear(640, 2)
A kernel of width k is a learned detector of k-character patterns such as
"қот", "сук", "*к", so the model works directly on spellings, masks and
stretched letters and has no unknown words except unseen characters.
Long texts: windows of `max_chars` with 50% overlap, max probability.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preprocess import preprocess

PAD, UNK = 0, 1


def build_vocab(texts, min_freq=2):
    cnt = Counter(ch for t in texts for ch in preprocess(t))
    chars = sorted(c for c, n in cnt.items() if n >= min_freq)
    return {c: i + 2 for i, c in enumerate(chars)}


def encode(text, vocab, max_chars):
    ids = [vocab.get(ch, UNK) for ch in preprocess(text)] or [UNK]
    if len(ids) <= max_chars:
        return [ids]
    step = max_chars // 2
    return [ids[i:i + max_chars] for i in range(0, len(ids) - step, step)]


class CharCNN(nn.Module):
    def __init__(self, vocab_size, emb=64, filters=128, widths=(2, 3, 4, 5, 6), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, w, padding=w // 2) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(widths), 2)

    def forward(self, x):                                   # x: (batch, chars)
        mask = (x != PAD).unsqueeze(1)                       # (batch, 1, chars)
        e = self.emb(x).transpose(1, 2)                      # (batch, emb, chars)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(e))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, -1e4)                   # ignore padding in the max
            pooled.append(h.max(dim=2).values)
        return self.out(self.drop(torch.cat(pooled, dim=1)))


def pad_batch(seqs):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs])


class CharCNNClassifier:
    """Same interface as infer.ProfanityClassifier."""

    def __init__(self, model_dir, device=None):
        d = Path(model_dir)
        self.cfg = json.loads((d / "inference_config.json").read_text(encoding="utf-8"))
        self.threshold = float(self.cfg["threshold"])
        self.max_chars = int(self.cfg["max_chars"])
        self.vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CharCNN(len(self.vocab) + 2, **self.cfg["hparams"]).to(self.device)
        self.model.load_state_dict(torch.load(d / "model.pt", map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=256):
        texts = [texts] if isinstance(texts, str) else list(texts)
        owners, wins = [], []
        for i, t in enumerate(texts):
            for w in encode(t, self.vocab, self.max_chars):
                owners.append(i)
                wins.append(w)
        probs = np.zeros(len(wins), dtype=np.float32)
        for s in range(0, len(wins), batch_size):
            x = pad_batch(wins[s:s + batch_size]).to(self.device)
            probs[s:s + batch_size] = torch.softmax(self.model(x), -1)[:, 1].cpu().numpy()
        out = np.zeros(len(texts), dtype=np.float32)
        np.maximum.at(out, np.array(owners), probs)
        return out

    def predict(self, texts):
        p = self.predict_proba(texts)
        return [{"label": int(x >= self.threshold), "probability": round(float(x), 4)} for x in p]
