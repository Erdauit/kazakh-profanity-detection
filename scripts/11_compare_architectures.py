"""Step 11 - compare architectures (all trained on the same train + synthetic data).

Selection uses VALIDATION only: selection score = mean(val F1, val-obfuscated F1),
the same criterion that chose between the XLM-R runs. Test / clean_eval
numbers are reported for information. Also measured: parameters, training time
and latency per comment (batch of 1) on GPU and CPU.

Run (after training the models; GPU should be free):
  python scripts/11_compare_architectures.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from augment import obfuscate_for_eval  # noqa: E402
from infer import load_classifier  # noqa: E402
from lexicon import normalize, strong_hits, weak_hits  # noqa: E402

P = ROOT / "data/processed"
train = pd.concat([pd.read_csv(P / "train.csv"), pd.read_csv(P / "train_synthetic.csv")])
val, test, ce = (pd.read_csv(P / f) for f in ("val.csv", "test.csv", "clean_eval.csv"))
val_obf, test_obf = obfuscate_for_eval(val, seed=0), obfuscate_for_eval(test, seed=0)
lat_texts = val.text.sample(100, random_state=0).tolist()


def latency_ms(predict, texts):
    predict(texts[:5])                                   # warm-up
    t = time.perf_counter()
    for x in texts:
        predict([x])
    return round(1000 * (time.perf_counter() - t) / len(texts), 1)


def scores(predict, thr):
    f = lambda d: round(float(f1_score(d.label, predict(d.text.tolist()) >= thr)), 4)  # noqa: E731
    pce = predict(ce.text.tolist()) >= thr
    r = {"val_f1": f(val), "val_obf_f1": f(val_obf)}
    r["selection_score"] = round((r["val_f1"] + r["val_obf_f1"]) / 2, 4)
    r |= {"test_f1": f(test), "test_obf_f1": f(test_obf),
          "clean_eval_fp_rate": round(float(pce[ce.label.to_numpy() == 0].mean()), 4)}
    return r


rows = []
# --- baselines
lex = lambda texts: np.array([1.0 if (strong_hits(t) or weak_hits(t)) else 0.0 for t in texts])  # noqa: E731
rows.append({"model": "lexicon rules", "arch": "regex", "params": 0, "train_min": 0,
             **scores(lex, 0.5), "gpu_ms": None, "cpu_ms": latency_ms(lex, lat_texts[:30])})
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
t0 = time.time()
lr = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(vec.fit_transform(train.text.map(normalize)), train.label)
tf_min = round((time.time() - t0) / 60, 2)
tf = lambda texts: lr.predict_proba(vec.transform(pd.Series(texts).map(normalize)))[:, 1]  # noqa: E731
pv = tf(val.text.tolist())
cand = np.linspace(0.05, 0.95, 91)
tf_thr = float(cand[np.argmax([f1_score(val.label, pv >= c) for c in cand])])
rows.append({"model": "char TF-IDF + LR", "arch": "linear", "params": len(vec.vocabulary_) + 1, "train_min": tf_min,
             **scores(tf, tf_thr), "gpu_ms": None, "cpu_ms": latency_ms(tf, lat_texts[:30])})

# --- trained neural models
for cfg_path in sorted((ROOT / "models").glob("*/inference_config.json")):
    d = cfg_path.parent
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    clf = load_classifier(d)
    row = {"model": d.name, "arch": cfg.get("model_name", ""), "params": cfg.get("n_params"),
           "train_min": cfg.get("train_minutes"), "synthetic": cfg["train"]["synthetic"],
           **scores(clf.predict_proba, clf.threshold),
           "gpu_ms": latency_ms(clf.predict_proba, lat_texts) if torch.cuda.is_available() else None}
    del clf
    torch.cuda.empty_cache()
    cpu = load_classifier(d, device="cpu")
    row["cpu_ms"] = latency_ms(cpu.predict_proba, lat_texts[:30])
    del cpu
    rows.append(row)
    print(row, flush=True)

res = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
out = ROOT / "results"
out.mkdir(exist_ok=True)
res.to_csv(out / "architecture_comparison.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
