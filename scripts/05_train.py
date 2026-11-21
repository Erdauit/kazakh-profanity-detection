"""Step 5 - fine-tune XLM-RoBERTa-base for binary profanity classification.

Objective: cross-entropy over 2 classes (softmax head on the <s> token).
Model selection uses the VALIDATION split only: the epoch with the best
val F1 is kept, and the decision threshold is chosen on val
(max F1; ties -> the higher threshold, i.e. fewer false accusations).
The test split is not touched here (see 06_evaluate.py).

Run:
  python scripts/05_train.py                      # default config (with synthetic rows)
  python scripts/05_train.py --no-synthetic --run-name xlmr_nosynth
  python scripts/05_train.py --wandb              # + log to Weights & Biases (run `wandb login` once)

wandb logs only hyper-parameters and metrics (loss, lr, val F1 / PR-AUC,
threshold), never review texts.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import PREPROCESS_VERSION, preprocess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
ap.add_argument("--run-name", default="xlmr_base")
ap.add_argument("--no-synthetic", action="store_true")
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--batch-size", type=int, default=32, help="effective batch size")
ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step (saves GPU memory)")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--warmup", type=float, default=0.1)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16",
                help="mixed precision on GPU (bf16 for DeBERTa, which overflows in fp16)")
ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
ap.add_argument("--wandb-project", default="kazakh-profanity")
ap.add_argument("--log-every", type=int, default=25)
args = ap.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
P = ROOT / "data/processed"
OUT = ROOT / "models" / args.run_name
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(P / "train.csv")
if not args.no_synthetic:
    train = pd.concat([train, pd.read_csv(P / "train_synthetic.csv")], ignore_index=True)
val = pd.read_csv(P / "val.csv")
print(f"train rows: {len(train)} (synthetic: {int(train.is_synthetic.sum())}), val rows: {len(val)}, device: {device}")

tok = AutoTokenizer.from_pretrained(args.model_name)


def encode(df):
    enc = tok([preprocess(t) for t in df.text], truncation=True, max_length=args.max_len)
    return [{"input_ids": ids, "attention_mask": m, "labels": int(y)}
            for ids, m, y in zip(enc["input_ids"], enc["attention_mask"], df.label)]


def collate(batch):
    # copy: never mutate the stored examples (they are reused every epoch)
    labels = torch.tensor([b["labels"] for b in batch])
    out = tok.pad([{k: v for k, v in b.items() if k != "labels"} for b in batch], return_tensors="pt")
    out["labels"] = labels
    return out


g = torch.Generator().manual_seed(args.seed)
train_dl = DataLoader(encode(train), batch_size=args.batch_size // args.grad_accum, shuffle=True, collate_fn=collate, generator=g)
val_dl = DataLoader(encode(val), batch_size=128, shuffle=False, collate_fn=collate)

# fp32 master weights: transformers v5 keeps the checkpoint dtype (mDeBERTa ships fp16 -> NaN with AdamW)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, dtype=torch.float32).to(device)
no_decay = ("bias", "LayerNorm.weight", "layer_norm")
params = [
    {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": args.weight_decay},
    {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
]
opt = torch.optim.AdamW(params, lr=args.lr)
total = math.ceil(len(train_dl) / args.grad_accum) * args.epochs   # optimizer steps
sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
AMP_DTYPE = torch.bfloat16 if args.precision == "bf16" else torch.float16
scaler = torch.amp.GradScaler(enabled=device == "cuda" and args.precision == "fp16")

run = None
if args.wandb:
    import wandb
    n_params = sum(p.numel() for p in model.parameters())
    run = wandb.init(project=args.wandb_project, name=args.run_name, config={"n_params": n_params,
        **vars(args), "n_train": len(train), "n_val": len(val), "n_synthetic": int(train.is_synthetic.sum()),
        "preprocess_version": PREPROCESS_VERSION, "steps_per_epoch": len(train_dl), "total_steps": total,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")


@torch.no_grad()
def val_probs():
    model.eval()
    ps = []
    for b in val_dl:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device, dtype=AMP_DTYPE, enabled=device == "cuda"):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        ps.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(ps)


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    i = np.flatnonzero(f1 >= f1.max() - 1e-9)[-1]       # highest threshold among the best
    return float(thr[i]), float(f1[i])


history, best = [], {"f1": -1}
model.train()
t0 = time.time()
gstep = 0
for epoch in range(1, args.epochs + 1):
    running = 0.0
    for step, b in enumerate(train_dl, 1):
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device, dtype=AMP_DTYPE, enabled=device == "cuda"):
            loss = model(**b).loss                          # cross-entropy
        scaler.scale(loss / args.grad_accum).backward()
        running += loss.item()
        if step % args.grad_accum and step != len(train_dl):
            continue                                        # accumulate gradients
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1
        if gstep % args.log_every == 0:
            el = time.time() - t0
            eta = el / gstep * (total - gstep)
            print(f"epoch {epoch} step {step}/{len(train_dl)} loss {running / step:.4f} "
                  f"| {gstep / el:.1f} it/s | elapsed {el / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
            if run:
                run.log({"train/loss": loss.item(), "train/loss_avg": running / step,
                         "train/lr": sched.get_last_lr()[0], "train/it_per_s": gstep / el}, step=gstep)
    p = val_probs()
    y = val.label.to_numpy()
    f1_05 = f1_score(y, p >= 0.5)
    thr, f1_thr = best_threshold(y, p)
    rec = {"epoch": epoch, "train_loss": round(running / len(train_dl), 4), "val_f1@0.5": round(float(f1_05), 4),
           "val_best_thr": round(thr, 4), "val_f1@best_thr": round(f1_thr, 4),
           "val_pr_auc": round(float(average_precision_score(y, p)), 4), "minutes": round((time.time() - t0) / 60, 1)}
    history.append(rec)
    print(rec, flush=True)
    if run:
        run.log({"epoch": epoch, "val/f1@0.5": f1_05, "val/f1@best_thr": f1_thr, "val/best_thr": thr,
                 "val/pr_auc": rec["val_pr_auc"], "train/epoch_loss": rec["train_loss"]}, step=gstep)
    if f1_05 > best["f1"]:
        best = {"f1": f1_05, "epoch": epoch, "threshold": thr, "f1_at_threshold": f1_thr}
        model.save_pretrained(OUT, safe_serialization=True)
        tok.save_pretrained(OUT)

cfg = {
    "arch": "transformer", "model_name": args.model_name, "run_name": args.run_name, "max_len": args.max_len,
    "threshold": round(best["threshold"], 4), "threshold_rule": "max F1 on val (highest thr among ties)",
    "best_epoch": best["epoch"], "val_f1@0.5": round(best["f1"], 4), "val_f1@threshold": round(best["f1_at_threshold"], 4),
    "preprocess_version": PREPROCESS_VERSION, "long_text": "sliding windows, stride max_len//2, max prob",
    "train": {"synthetic": not args.no_synthetic, "n_train": len(train), "batch_size": args.batch_size,
              "grad_accum": args.grad_accum, "lr": args.lr,
              "epochs": args.epochs, "warmup": args.warmup, "weight_decay": args.weight_decay, "seed": args.seed,
              "precision": args.precision if device == "cuda" else "fp32", "optimizer": "AdamW", "scheduler": "linear warmup/decay",
              "grad_clip": 1.0, "loss": "cross-entropy"},
    "n_params": sum(p.numel() for p in model.parameters()),
    "train_minutes": history[-1]["minutes"],
    "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
}
(OUT / "inference_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
print("saved best epoch", best["epoch"], "->", OUT.relative_to(ROOT), "| threshold", round(best["threshold"], 4))
if run:
    run.summary.update({"best_epoch": best["epoch"], "best_val_f1@0.5": best["f1"],
                        "threshold": best["threshold"], "val_f1@threshold": best["f1_at_threshold"]})
    run.finish()
