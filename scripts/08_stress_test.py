"""Step 8 - stress test: how do models break when users disguise words?

Every test row is rewritten with one "attack" at a time:
  mask        inner letters -> *            қотақбас -> қ******с
  latin       Cyrillic -> Latin             qotaqbas
  kz_keyboard Kazakh letters -> Russian     котакбас
  stretch     repeated vowels               қотааақбас
  spaced      letters split by spaces       қ о т а қ б а с
  leet        digits / symbols for letters  қ0т@қб@с
  upper       CAPS                          ҚОТАҚБАС
For profane rows the attacked word is the profane one (found by the lexicon);
for clean rows a random word is attacked, so we also see whether the disguise
ALONE makes a model shout "profanity" (false positives).

Models compared (all trained on train only, thresholds from val):
  lexicon     rule: any lexicon hit -> 1
  tfidf_lr    char 2-5-gram TF-IDF + logistic regression
  xlmr_*      fine-tuned XLM-R runs found in models/

This is analysis after model selection (selection used val only).
Run:  python scripts/08_stress_test.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from augment import stretch, to_kz_keyboard, to_latin  # noqa: E402
from infer import load_classifier  # noqa: E402
from lexicon import normalize, strong_hits, weak_hits  # noqa: E402

P = ROOT / "data/processed"
train, val, test = (pd.read_csv(P / f"{s}.csv") for s in ("train", "val", "test"))
rng = np.random.default_rng(0)
WORD = re.compile(r"[\wәғқңөұүһі]+")
LEET = str.maketrans({"о": "0", "О": "0", "а": "@", "А": "@", "з": "3", "З": "3", "б": "6", "ч": "4", "і": "1"})


def target_words(text, label):
    words = WORD.findall(text)
    if label == 1:
        bad = [w for w in words if len(w) >= 2 and (strong_hits(w) or weak_hits(w))]
        if bad:
            return set(bad)
    long = [w for w in words if len(w) >= 3]
    return {long[int(rng.integers(len(long)))]} if long else set()


def on_words(fn):
    def attack(text, label):
        tw = target_words(text, label)
        return WORD.sub(lambda m: fn(m.group(0)) if m.group(0) in tw else m.group(0), text)
    return attack


def _mask(w):
    return w[0] + "*" * (len(w) - 2) + w[-1] if len(w) > 2 else w[0] + "*"


ATTACKS = {
    "original": lambda t, y: t,
    "mask": on_words(_mask),
    "latin": lambda t, y: to_latin(t, rng),
    "kz_keyboard": lambda t, y: to_kz_keyboard(t, rng),
    "stretch": lambda t, y: stretch(t, rng),
    "spaced": on_words(lambda w: " ".join(w)),
    "leet": on_words(lambda w: w.translate(LEET)),
    "upper": lambda t, y: t.upper(),
}

# ---------------------------------------------------------------- models
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
lr = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(vec.fit_transform(train.text.map(normalize)), train.label)
tf_proba = lambda texts: lr.predict_proba(vec.transform(pd.Series(texts).map(normalize)))[:, 1]  # noqa: E731
pv = tf_proba(val.text)
cand = np.linspace(0.05, 0.95, 91)
tf_thr = float(cand[np.argmax([f1_score(val.label, pv >= c) for c in cand])])

models = {
    "lexicon": lambda texts: np.array([1.0 if (strong_hits(t) or weak_hits(t)) else 0.0 for t in texts]),
    "tfidf_lr": tf_proba,
}
thresholds = {"lexicon": 0.5, "tfidf_lr": tf_thr}
for d in sorted((ROOT / "models").glob("*/inference_config.json")):
    clf = load_classifier(d.parent)
    models[d.parent.name] = clf.predict_proba
    thresholds[d.parent.name] = clf.threshold

# ---------------------------------------------------------------- run
rows, examples = [], []
y = test.label.to_numpy()
for aname, fn in ATTACKS.items():
    texts = [fn(t, lab) for t, lab in zip(test.text, test.label)]
    examples.append({"attack": aname, "example": next(t for t, lab in zip(texts, y) if lab == 1)[:80]})
    for mname, predict in models.items():
        yhat = (predict(texts) >= thresholds[mname]).astype(int)
        rows.append({"attack": aname, "model": mname,
                     "recall_profane": round(float(yhat[y == 1].mean()), 3),
                     "false_pos_rate_clean": round(float(yhat[y == 0].mean()), 3),
                     "f1": round(f1_score(y, yhat), 3)})

res = pd.DataFrame(rows)
out = ROOT / "results"
out.mkdir(exist_ok=True)
res.to_csv(out / "stress_test.csv", index=False)
(out / "stress_test_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
pd.set_option("display.width", 200)
for metric in ("f1", "recall_profane", "false_pos_rate_clean"):
    print(f"\n== {metric}")
    print(res.pivot(index="attack", columns="model", values=metric).loc[list(ATTACKS)].to_string())

# ---------------------------------------------------------------- fair comparison
# The 'mined_lexicon_hit' test rows were FOUND by the lexicon, so on the full test
# set the lexicon looks better than it is. Compare on rows not selected by it:
# the course test rows, plus false positives on clean_eval (lexicon-free by
# construction, so only the learned models are compared there).
ce = pd.read_csv(P / "clean_eval.csv")
course = test[test.source == "course"]
fair = []
for mname, predict in models.items():
    yc = (predict(course.text.tolist()) >= thresholds[mname]).astype(int)
    row = {"model": mname, "threshold": round(thresholds[mname], 3),
           "course_test_f1": round(f1_score(course.label, yc), 3),
           "course_test_recall": round(float(yc[course.label.to_numpy() == 1].mean()), 3),
           "course_test_fp_rate": round(float(yc[course.label.to_numpy() == 0].mean()), 3)}
    if mname != "lexicon":
        yce = (predict(ce.text.tolist()) >= thresholds[mname]).astype(int)
        row["clean_eval_fp_rate"] = round(float(yce[ce.label.to_numpy() == 0].mean()), 4)
    fair.append(row)
fair = pd.DataFrame(fair)
fair.to_csv(out / "model_comparison.csv", index=False)
print("\n== fair comparison (course test rows n=%d, clean_eval n=%d)" % (len(course), len(ce)))
print(fair.to_string(index=False))
