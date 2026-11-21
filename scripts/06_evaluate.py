"""Step 6 - evaluate a saved model.

  python scripts/06_evaluate.py --model models/xlmr_base --split val    # model selection
  python scripts/06_evaluate.py --model models/xlmr_base --split test   # FINAL, run once

--split val : prints val + val_obfuscated metrics (used to choose the run).
--split test: held-out test + test_obfuscated + clean_eval (ordinary reviews);
              writes results/ (metrics.json, confusion matrix, predictions, errors).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from augment import obfuscate_for_eval  # noqa: E402
from infer import load_classifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/kazroberta")
ap.add_argument("--split", choices=["val", "test"], default="val")
ap.add_argument("--out", default="results")
args = ap.parse_args()

P = ROOT / "data/processed"
clf = load_classifier(ROOT / args.model)
thr = clf.threshold


def metrics(y, p):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "n_profane": int(y.sum()), "threshold": thr,
            "precision": round(precision_score(y, yhat, zero_division=0), 4),
            "recall": round(recall_score(y, yhat, zero_division=0), 4),
            "f1": round(f1_score(y, yhat, zero_division=0), 4),
            "accuracy": round(accuracy_score(y, yhat), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "false_positive_rate": round(fp / max(1, fp + tn), 4)}


df = pd.read_csv(P / f"{args.split}.csv")
y = df.label.to_numpy()
p = clf.predict_proba(df.text)
obf = obfuscate_for_eval(df, seed=0)
p_obf = clf.predict_proba(obf.text)
res = {args.split: metrics(y, p), f"{args.split}_obfuscated": metrics(y, p_obf)}
res["selection_score"] = round((res[args.split]["f1"] + res[f"{args.split}_obfuscated"]["f1"]) / 2, 4)

if args.split == "val":
    print(json.dumps({"model": args.model, **res}, indent=2, ensure_ascii=False))
    sys.exit()

# ----------------------------------------------------------------- final test report
out = ROOT / args.out
out.mkdir(exist_ok=True)
df["prob"], df["pred"] = p.round(4), (p >= thr).astype(int)
df["error"] = np.select([(df.pred == 1) & (df.label == 0), (df.pred == 0) & (df.label == 1)], ["FP", "FN"], "")
res["test_by_subset"] = {k: metrics(g.label.to_numpy(), g.prob.to_numpy()) for k, g in df.groupby("subset")}
res["test_by_label_source"] = {k: metrics(g.label.to_numpy(), g.prob.to_numpy()) for k, g in df.groupby("label_source")}

ce = pd.read_csv(P / "clean_eval.csv")
ce["prob"] = clf.predict_proba(ce.text).round(4)
ce["pred"] = (ce.prob >= thr).astype(int)
res["clean_eval"] = metrics(ce.label.to_numpy(), ce.prob.to_numpy())
res["clean_eval_by_domain"] = {k: metrics(g.label.to_numpy(), g.prob.to_numpy()) for k, g in ce.groupby("domain")}
res["model_config"] = clf.cfg

(out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
df[["id", "text", "label", "prob", "pred", "error", "subset", "label_source", "domain"]].to_csv(
    out / "test_predictions.csv", index=False, encoding="utf-8")
df[df.error != ""].assign(conf=lambda d: (d.prob - 0.5).abs()).sort_values("conf", ascending=False).drop(columns="conf") \
    [["id", "error", "prob", "label", "subset", "text"]].to_csv(out / "test_errors.csv", index=False, encoding="utf-8")
ce[(ce.pred == 1) & (ce.label == 0)].sort_values("prob", ascending=False).to_csv(
    out / "clean_eval_false_positives.csv", index=False, encoding="utf-8")
obf.assign(prob=p_obf.round(4), pred=(p_obf >= thr).astype(int))[["id", "text", "label", "prob", "pred"]].to_csv(
    out / "test_obfuscated_predictions.csv", index=False, encoding="utf-8")

cm = confusion_matrix(y, df.pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["0 clean", "1 profane"])
ax.set_yticks([0, 1], ["0 clean", "1 profane"])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"Test (n={len(y)}), threshold={thr:.2f}")
fig.tight_layout()
fig.savefig(out / "confusion_matrix.png", dpi=150)

print(json.dumps({k: res[k] for k in ["test", "test_obfuscated", "clean_eval", "selection_score"]}, indent=2))
print("by subset:", {k: (v["n"], v["f1"], v["false_positive_rate"]) for k, v in res["test_by_subset"].items()})
