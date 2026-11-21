"""Step 1 - find probably-wrong labels in the original dataset.

Two independent signals are combined:
  1. lexicon conflict  - label 0 but an obscene word matched, or label 1 and
                         nothing from the lexicon matched;
  2. model conflict    - 5-fold out-of-fold probability from a char n-gram
                         TF-IDF + logistic regression disagrees strongly with
                         the label (confident-learning style).

Output: data/interim/audit_candidates.csv - every flagged row with both
signals. These rows were then read by a human and the decisions were written
to data/label_corrections.csv (that file, not this script, changes labels).

Run:  python scripts/01_audit_labels.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lexicon import matched_spans, normalize, strong_hits, weak_hits  # noqa: E402

SEED = 42
df = pd.read_csv(ROOT / "kazakh_profanity_share.csv", encoding="utf-8-sig")
df["strong"] = df.text.map(lambda t: ";".join(strong_hits(t)))
df["weak"] = df.text.map(lambda t: ";".join(weak_hits(t)))
df["spans"] = df.text.map(lambda t: ";".join(matched_spans(t)))

# out-of-fold probabilities (each row is scored by a model that never saw it)
oof = np.zeros(len(df))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X = df.text.map(normalize).values
for tr, te in skf.split(X, df.label):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(X[tr]), df.label.values[tr])
    oof[te] = clf.predict_proba(vec.transform(X[te]))[:, 1]
df["oof_prob"] = oof.round(3)

has_s, has_w = df.strong != "", df.weak != ""
reasons = pd.Series("", index=df.index)
reasons[(df.label == 0) & has_s] += "L0+strong;"
reasons[(df.label == 0) & has_w & ~has_s] += "L0+weak;"
reasons[(df.label == 1) & ~has_s & ~has_w] += "L1+no_lexicon;"
reasons[(df.label == 0) & (df.oof_prob > 0.8)] += "L0+model>0.8;"
reasons[(df.label == 1) & (df.oof_prob < 0.2)] += "L1+model<0.2;"
df["flag"] = reasons

out = df[df.flag != ""].sort_values(["flag", "oof_prob"])
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
out[["kazsandra_id", "split", "label", "oof_prob", "flag", "strong", "weak", "spans", "text"]].to_csv(
    ROOT / "data/interim/audit_candidates.csv", index=False, encoding="utf-8")

from sklearn.metrics import f1_score
print("OOF F1 (noisy labels):", round(f1_score(df.label, oof > 0.5), 4))
print("flagged rows:", len(out))
print(out.flag.str.split(";").explode().loc[lambda s: s != ""].value_counts().to_string())
