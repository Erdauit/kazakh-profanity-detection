"""Step 7 - from a classifier to a moderation policy (two thresholds).

A single threshold forces one trade-off between
  false positive = an honest review is hidden (angry customer, lost feedback)
  false negative = profanity is published (brand / platform harm).
Real moderation can send uncertain cases to a human. We pick, ON VALIDATION:
  t_hide   = lowest threshold whose val precision >= --hide-precision (default 0.98)
             -> above it the comment is hidden automatically
  t_review = highest threshold whose val recall >= --review-recall (default 0.98)
             -> between t_review and t_hide a moderator decides
             -> below t_review the comment is published
and then report what this policy does on test and on clean_eval
(ordinary reviews). Saves models/<run>/moderation.json (used by predict.py),
results/moderation.json and results/pr_curve.png.

Run:  python scripts/07_moderation.py --model models/xlmr_base
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import load_classifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/kazroberta")
ap.add_argument("--hide-precision", type=float, default=0.98)
ap.add_argument("--review-recall", type=float, default=0.98)
ap.add_argument("--out", default="results")
args = ap.parse_args()

P = ROOT / "data/processed"
clf = load_classifier(ROOT / args.model)
val, test, ce = (pd.read_csv(P / f) for f in ("val.csv", "test.csv", "clean_eval.csv"))
pv, pt, pc = (clf.predict_proba(d.text) for d in (val, test, ce))

prec, rec, thr = precision_recall_curve(val.label, pv)
prec, rec = prec[:-1], rec[:-1]
ok_hide = np.flatnonzero(prec >= args.hide_precision)
t_hide = float(thr[ok_hide[0]]) if len(ok_hide) else float(thr[-1])
ok_rev = np.flatnonzero(rec >= args.review_recall)
t_review = float(thr[ok_rev[-1]]) if len(ok_rev) else float(thr[0])
t_review = min(t_review, t_hide)


def policy(y, p):
    y = np.asarray(y)
    hide, review = p >= t_hide, (p >= t_review) & (p < t_hide)
    publish = p < t_review
    prof, clean = y == 1, y == 0
    pct = lambda m, base: round(100 * m[base].mean(), 2) if base.any() else None  # noqa: E731
    return {
        "n": int(len(y)), "n_profane": int(prof.sum()),
        "share_auto_hidden_%": round(100 * hide.mean(), 2),
        "share_to_human_%": round(100 * review.mean(), 2),
        "share_published_%": round(100 * publish.mean(), 2),
        "profane_auto_hidden_%": pct(hide, prof), "profane_to_human_%": pct(review, prof),
        "profane_published_(missed)_%": pct(publish, prof),
        "clean_auto_hidden_(wrongly)_%": pct(hide, clean), "clean_to_human_%": pct(review, clean),
        "clean_published_%": pct(publish, clean),
    }


res = {"t_hide": round(t_hide, 4), "t_review": round(t_review, 4), "single_threshold": clf.threshold,
       "rule": f"t_hide: val precision >= {args.hide_precision}; t_review: val recall >= {args.review_recall}",
       "val": policy(val.label, pv), "test": policy(test.label, pt), "clean_eval": policy(ce.label, pc),
       "test_pr_auc": round(float(average_precision_score(test.label, pt)), 4)}
(ROOT / args.model / "moderation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
out = ROOT / args.out
out.mkdir(exist_ok=True)
(out / "moderation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(5.2, 4.2))
for name, y, p in (("val", val.label, pv), ("test", test.label, pt)):
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})")
for t, lab, mk in ((t_hide, "t_hide", "o"), (t_review, "t_review", "s"), (clf.threshold, "F1 threshold", "^")):
    yhat = pt >= t
    r = (yhat & (test.label == 1)).sum() / max(1, (test.label == 1).sum())
    pr_ = (yhat & (test.label == 1)).sum() / max(1, yhat.sum())
    ax.scatter([r], [pr_], marker=mk, s=60, zorder=3, label=f"{lab}={t:.3f} (test)")
ax.set_xlabel("recall (profane caught)")
ax.set_ylabel("precision (flagged that are profane)")
ax.set_ylim(0.5, 1.01)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
ax.set_title("Precision-recall and moderation thresholds")
fig.tight_layout()
fig.savefig(out / "pr_curve.png", dpi=150)
print(json.dumps(res, indent=2))
