"""Step 4 - sanity check of the built dataset with a quick baseline.

NOT the final model. A char n-gram TF-IDF + logistic regression is trained
on train (optionally + synthetic) and evaluated on val, test, the test
subsets, and on clean_eval (ordinary reviews -> false-positive rate).
It answers: are the splits learnable, does the clean set behave, did the
added data help compared with the original course data alone?

Run:  python scripts/04_check_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lexicon import normalize  # noqa: E402

P = ROOT / "data/processed"
data = pd.read_csv(P / "kazakh_profanity_v2.csv")
synth = pd.read_csv(P / "train_synthetic.csv")
clean_eval = pd.read_csv(P / "clean_eval.csv")


def fit(train_df):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=4, max_iter=3000, class_weight="balanced")
    clf.fit(vec.fit_transform(train_df.text.map(normalize)), train_df.label)
    return lambda texts: clf.predict_proba(vec.transform(pd.Series(texts).map(normalize)))[:, 1]


def report(name, y, p, thr=0.5):
    y = y.astype(int).to_numpy()
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(set=name, n=len(y), pos=int(y.sum()),
                P=round(precision_score(y, yhat, zero_division=0), 3),
                R=round(recall_score(y, yhat, zero_division=0), 3),
                F1=round(f1_score(y, yhat, zero_division=0), 3),
                FP=int(fp), FN=int(fn), FP_rate_on_clean=round(fp / max(1, fp + tn), 4))


rows = []
train = data[data.split == "train"]
test, val = data[data.split == "test"], data[data.split == "val"]

# robustness probe: the SAME test rows rewritten with masking / Latin / keyboard folding
from augment import obfuscate_for_eval  # noqa: E402
test_obf = obfuscate_for_eval(test, seed=0)
setups = {
    "course_only(original labels)": train[train.source == "course"].assign(
        label=lambda d: d.original_label.astype(int)),
    "course_only(corrected)": train[train.source == "course"],
    "v2_train": train,
    "v2_train+synthetic": pd.concat([train, synth]),
}
for setup, tr in setups.items():
    predict = fit(tr)
    rows.append({"setup": setup, "n_train": len(tr), **report("val", val.label, predict(val.text))})
    rows.append({"setup": setup, "n_train": len(tr), **report("test", test.label, predict(test.text))})
    ct = test[test.source == "course"]
    rows.append({"setup": setup, "n_train": len(tr), **report("test[course rows]", ct.label, predict(ct.text))})
    rows.append({"setup": setup, "n_train": len(tr), **report("test_obfuscated", test_obf.label, predict(test_obf.text))})
    p_ce = predict(clean_eval.text)
    rows.append({"setup": setup, "n_train": len(tr), **report("clean_eval", clean_eval.label, p_ce)})
    if setup == "v2_train+synthetic":
        fp = clean_eval.assign(prob=p_ce.round(3))[(p_ce >= 0.5) & (clean_eval.label == 0)]
        fp.sort_values("prob", ascending=False).to_csv(P / "baseline_clean_eval_false_positives.csv", index=False)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
out.to_csv(P / "baseline_check.csv", index=False)
