"""Step 5b - train the character-level CNN from scratch (architecture comparison).

Same data (train + synthetic), same val-based selection and threshold rule
as scripts/05_train.py; only the architecture differs.

Run:  python scripts/05b_train_charcnn.py [--wandb]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charcnn import CharCNN, build_vocab, encode, pad_batch  # noqa: E402
from preprocess import PREPROCESS_VERSION  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", default="charcnn")
ap.add_argument("--no-synthetic", action="store_true")
ap.add_argument("--max-chars", type=int, default=400)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--weight-decay", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--wandb", action="store_true")
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
P, OUT = ROOT / "data/processed", ROOT / "models" / args.run_name
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(P / "train.csv")
if not args.no_synthetic:
    train = pd.concat([train, pd.read_csv(P / "train_synthetic.csv")], ignore_index=True)
val = pd.read_csv(P / "val.csv")

vocab = build_vocab(train.text)                    # fitted on TRAIN only
hp = {"emb": 64, "filters": 128, "widths": [2, 3, 4, 5, 6], "dropout": 0.3}
model = CharCNN(len(vocab) + 2, **hp).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"train {len(train)} rows, vocab {len(vocab)} chars, params {n_params:,}, device {device}")

# training uses the first window only (98%+ of texts fit in 400 chars)
Xtr = [encode(t, vocab, args.max_chars)[0] for t in train.text]
ytr = train.label.to_numpy()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
steps = (len(Xtr) + args.batch_size - 1) // args.batch_size
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps * args.epochs, pct_start=0.1)
loss_fn = torch.nn.CrossEntropyLoss()


@torch.no_grad()
def predict_val():
    model.eval()
    owners, wins = [], []
    for i, t in enumerate(val.text):
        for w in encode(t, vocab, args.max_chars):
            owners.append(i); wins.append(w)
    ps = np.zeros(len(wins), dtype=np.float32)
    for s in range(0, len(wins), 512):
        ps[s:s + 512] = torch.softmax(model(pad_batch(wins[s:s + 512]).to(device)), -1)[:, 1].cpu().numpy()
    out = np.zeros(len(val), dtype=np.float32)
    np.maximum.at(out, np.array(owners), ps)
    model.train()
    return out


run = None
if args.wandb:
    import wandb
    run = wandb.init(project="kazakh-profanity", name=args.run_name, config={**vars(args), **hp, "n_params": n_params})

history, best, t0 = [], {"f1": -1}, time.time()
rng = np.random.default_rng(args.seed)
model.train()
for epoch in range(1, args.epochs + 1):
    order, total = rng.permutation(len(Xtr)), 0.0
    for s in range(0, len(order), args.batch_size):
        idx = order[s:s + args.batch_size]
        x = pad_batch([Xtr[i] for i in idx]).to(device)
        y = torch.tensor(ytr[idx]).to(device)
        loss = loss_fn(model(x), y)                 # cross-entropy
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        total += loss.item()
    p, y = predict_val(), val.label.to_numpy()
    f1 = f1_score(y, p >= 0.5)
    pr, rc, th = precision_recall_curve(y, p)
    f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    i = np.flatnonzero(f1s >= f1s.max() - 1e-9)[-1]
    rec = {"epoch": epoch, "train_loss": round(total / steps, 4), "val_f1@0.5": round(float(f1), 4),
           "val_best_thr": round(float(th[i]), 4), "val_f1@best_thr": round(float(f1s[i]), 4),
           "val_pr_auc": round(float(average_precision_score(y, p)), 4), "minutes": round((time.time() - t0) / 60, 2)}
    history.append(rec)
    print(rec, flush=True)
    if run:
        run.log({"epoch": epoch, "train/epoch_loss": rec["train_loss"], "val/f1@0.5": f1, "val/pr_auc": rec["val_pr_auc"]})
    if f1 > best["f1"]:
        best = {"f1": float(f1), "epoch": epoch, "threshold": float(th[i]), "f1_at_threshold": float(f1s[i])}
        torch.save(model.state_dict(), OUT / "model.pt")

(OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
cfg = {"arch": "charcnn", "model_name": "char-CNN (from scratch)", "run_name": args.run_name,
       "max_chars": args.max_chars, "hparams": hp, "threshold": round(best["threshold"], 4),
       "threshold_rule": "max F1 on val (highest thr among ties)", "best_epoch": best["epoch"],
       "val_f1@0.5": round(best["f1"], 4), "val_f1@threshold": round(best["f1_at_threshold"], 4),
       "preprocess_version": PREPROCESS_VERSION, "vocab_size": len(vocab), "n_params": n_params,
       "train_minutes": history[-1]["minutes"],
       "train": {"synthetic": not args.no_synthetic, "n_train": len(train), "batch_size": args.batch_size,
                 "lr": args.lr, "epochs": args.epochs, "weight_decay": args.weight_decay, "seed": args.seed,
                 "optimizer": "AdamW", "scheduler": "one-cycle", "loss": "cross-entropy"}}
(OUT / "inference_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
if run:
    run.summary.update({"best_epoch": best["epoch"], "best_val_f1@0.5": best["f1"]})
    run.finish()
print("saved best epoch", best["epoch"], "->", OUT.relative_to(ROOT), "| threshold", round(best["threshold"], 4))
