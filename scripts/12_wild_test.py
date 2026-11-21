"""Step 12 - "in the wild" test: ordinary reviews that contain NO known swear word.

Why: candidates for the dataset were found with the lexicon, so 413 of the 419
profane test rows contain a lexicon word - the test mostly measures "known
words and their disguises". Here the model meets what it would meet in
production: the rest of KazSAnDRA (never used anywhere), without any lexicon
hit, so every profane review in it is one the word list does not know.

  python scripts/12_wild_test.py sample     # pool -> scores of 3 models -> stratified sample to label
  (label data/interim/wild_label_part*.csv blind, by docs/labeling_guidelines.md
   -> data/wild_labels.csv with columns id,label,note; label -1 = out of scope)
  python scripts/12_wild_test.py evaluate   # weighted precision / recall -> results/wild_test.json

Pool: KazSAnDRA rows not in the course file, the dataset or clean_eval; exact
duplicates and near-duplicates of train rows (char TF-IDF cosine >= 0.8) removed;
no lexicon hit. Strata by the final model's score (moderation bands) plus the
rows only the other two models flag; each stratum is sampled at random and
results are re-weighted to the whole pool (stratified estimator).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lexicon import strong_hits, weak_hits  # noqa: E402
from textutils import anonymize, dedup_key  # noqa: E402

SEED = 42
P = ROOT / "data/processed"
INTERIM = ROOT / "data/interim"
POOL_SCORES = INTERIM / "wild_pool_scores.csv.gz"
SAMPLE = INTERIM / "wild_sample.csv"
LABELS = ROOT / "data/wild_labels.csv"
MODELS = ["kazroberta", "mdeberta", "charcnn"]
TARGET = {"H_auto_hide": 150, "F_flagged": 150, "R_review": 200, "O_other_models": 150, "U_unflagged": 350}

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["sample", "evaluate"])
args = ap.parse_args()


def thresholds():
    t = {m: json.loads((ROOT / f"models/{m}/inference_config.json").read_text(encoding="utf-8"))["threshold"] for m in MODELS}
    mod = json.loads((ROOT / "models/kazroberta/moderation.json").read_text(encoding="utf-8"))
    return t, mod["t_hide"], mod["t_review"]


def stratum(df):
    thr, t_hide, t_review = thresholds()
    pk = df.p_kazroberta
    other = (df.p_mdeberta >= thr["mdeberta"]) | (df.p_charcnn >= thr["charcnn"])
    return np.select([pk >= t_hide, pk >= thr["kazroberta"], pk >= t_review, other],
                     ["H_auto_hide", "F_flagged", "R_review", "O_other_models"], "U_unflagged")


def build_pool():
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from infer import load_classifier

    raw = ROOT / "data/raw/kazsandra"
    ks = pd.concat([pd.read_csv(raw / f) for f in ["06_sc_train_ib.csv", "09_sc_valid.csv", "10_sc_test.csv"]])
    ks = ks.rename(columns={"custom_id": "id", "label": "stars"})
    ks["stars"] += 1
    course = pd.read_csv(ROOT / "kazakh_profanity_share.csv", encoding="utf-8-sig").rename(columns={"kazsandra_id": "id"})
    known = pd.concat([pd.read_csv(P / "kazakh_profanity_v2.csv"), pd.read_csv(P / "clean_eval.csv"), course])
    n0 = len(ks)
    ks = ks[~ks.id.isin(known.id)].copy()
    ks["text"] = ks.text.astype(str).map(anonymize)
    ks = ks[ks.text.str.strip().str.len() >= 3]
    ks["key"] = ks.text.map(dedup_key)
    known_keys = set(known.text.astype(str).map(dedup_key))
    ks = ks[(ks.key.str.len() > 0) & ~ks.key.isin(known_keys)].drop_duplicates("key")
    ks = ks[~ks.text.map(lambda t: bool(strong_hits(t) or weak_hits(t)))]
    n_lexfree = len(ks)

    # near-duplicates of training rows: the model has (almost) seen them
    train = pd.read_csv(P / "train.csv")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True).fit(train.text)
    xt = vec.transform(train.text).T.tocsc()
    xp = vec.transform(ks.text)
    best = np.concatenate([(xp[i:i + 2000] @ xt).max(axis=1).toarray().ravel() for i in range(0, xp.shape[0], 2000)])
    ks = ks[best < 0.8].copy()
    print(f"KazSAnDRA {n0} -> unused, deduplicated, lexicon-free {n_lexfree} -> not near-dup of train {len(ks)}", flush=True)

    for m in MODELS:
        clf = load_classifier(ROOT / "models" / m)
        ks[f"p_{m}"] = np.round(clf.predict_proba(ks.text.tolist()), 5)
        print("scored", m, flush=True)
        del clf
        torch.cuda.empty_cache()
    ks = ks[["id", "domain", "stars", "text"] + [f"p_{m}" for m in MODELS]]
    ks.drop(columns="text").to_csv(POOL_SCORES, index=False, encoding="utf-8")   # texts: in the raw corpus
    return ks


if args.step == "sample":
    pool = build_pool()
    pool["stratum"] = stratum(pool)
    print(pool.stratum.value_counts().to_string())
    sample = pd.concat([g.sample(min(len(g), TARGET[s]), random_state=SEED) for s, g in pool.groupby("stratum")])
    sample.to_csv(SAMPLE, index=False, encoding="utf-8")
    # blind files for labeling: text only, shuffled, no scores, no stratum
    blind = sample[["id", "text"]].sample(frac=1, random_state=SEED).reset_index(drop=True)
    for k, part in enumerate(np.array_split(blind.index, 4), 1):
        blind.loc[part].to_csv(INTERIM / f"wild_label_part{k}.csv", index=False, encoding="utf-8")
    print("sample:", len(sample), "->", SAMPLE.relative_to(ROOT))

else:
    thr, t_hide, t_review = thresholds()
    pool = pd.read_csv(POOL_SCORES)
    pool["stratum"] = stratum(pool)
    N = pool.stratum.value_counts()
    s = pd.read_csv(SAMPLE).merge(pd.read_csv(LABELS), on="id", how="left", validate="1:1")
    assert s.label.notna().all(), "unlabeled rows in the sample"
    s["stratum"] = stratum(s)
    v = s[s.label.isin([0, 1])].copy()
    valid_frac = v.groupby("stratum").size() / s.groupby("stratum").size()
    # stratified (Horvitz-Thompson) weight: how many in-scope pool rows one labeled row stands for
    v["w"] = v.stratum.map(N * valid_frac / v.groupby("stratum").size())

    def est(d):
        out = {"profane_per_10k": 1e4 * (d.w * d.label).sum() / d.w.sum()}
        for m in MODELS:
            f = d[f"p_{m}"] >= thr[m]
            tp = (d.w * d.label * f).sum()
            out[f"{m}_precision"] = tp / (d.w * f).sum()
            out[f"{m}_recall"] = tp / (d.w * d.label).sum()
            out[f"{m}_false_positive_rate"] = (d.w * (1 - d.label) * f).sum() / (d.w * (1 - d.label)).sum()
        hide, review = d.p_kazroberta >= t_hide, (d.p_kazroberta >= t_review) & (d.p_kazroberta < t_hide)
        prof = (d.w * d.label).sum()
        out |= {"profane_auto_hidden": (d.w * d.label * hide).sum() / prof,
                "profane_to_human": (d.w * d.label * review).sum() / prof,
                "profane_published": (d.w * d.label * ~(hide | review)).sum() / prof,
                "auto_hide_precision": (d.w * d.label * hide).sum() / (d.w * hide).sum(),
                "clean_auto_hidden_per_10k": 1e4 * (d.w * (1 - d.label) * hide).sum() / (d.w * (1 - d.label)).sum(),
                "clean_to_human": (d.w * (1 - d.label) * review).sum() / (d.w * (1 - d.label)).sum()}
        return out

    point = est(v)
    # sensitivity: every label-1 the annotator marked "unsure" counted as 0
    unsure = v.note.fillna("").str.lower().str.startswith("unsure")
    strict = est(v.assign(label=np.where(unsure & (v.label == 1), 0, v.label)))
    # a stratum with 0 profane in n labeled rows can still hold up to 1 - 0.05**(1/n) (one-sided 95%);
    # the bootstrap cannot see that, so recall also gets a worst-case lower bound
    prof = (v.w * v.label).groupby(v.stratum).sum()
    flagged_prof = (v.w * v.label * (v.p_kazroberta >= thr["kazroberta"])).sum()
    worst = sum(N[h] * valid_frac[h] * (1 - 0.05 ** (1 / n)) for h, n in v.groupby("stratum").size().items()
                if v[v.stratum == h].label.sum() == 0)
    recall_lb = flagged_prof / (prof.sum() + worst)
    rng = np.random.default_rng(SEED)
    boots = [est(pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1_000_000_000))) for _, g in v.groupby("stratum")]))
             for _ in range(1000)]
    ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))] for k in point}
    by = v.groupby("stratum").agg(labeled=("label", "size"), profane=("label", "sum"))
    by["pool_size"] = N
    by["share_profane"] = (by.profane / by.labeled).round(3)
    market = pool[pool.domain == "market"]
    res = {
        "pool_size": int(len(pool)), "labeled": int(len(s)), "out_of_scope_(-1)": int((s.label == -1).sum()),
        "strata": by.reset_index().to_dict("records"),
        "estimates": {k: round(float(x), 4) for k, x in point.items()},
        "ci95_bootstrap_within_strata": {k: [round(a, 4), round(b, 4)] for k, (a, b) in ci.items()},
        "estimates_unsure_positives_as_0": {k: round(float(x), 4) for k, x in strict.items()},
        "kazroberta_recall_worst_case_lower_bound": round(float(recall_lb), 4),
        "unsure_labels": int(unsure.sum()),
        "kazroberta_flag_rate_by_domain": {d: round(float((g.p_kazroberta >= thr["kazroberta"]).mean()), 4) for d, g in pool.groupby("domain")},
        "market_reviews_in_pool": int(len(market)),
        "thresholds": {**thr, "t_hide": t_hide, "t_review": t_review},
    }
    (ROOT / "results/wild_test.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    s[["id", "stratum", "label", "note", "p_kazroberta", "p_mdeberta", "p_charcnn", "domain", "stars", "text"]] \
        .sort_values(["stratum", "p_kazroberta"], ascending=[True, False]) \
        .to_csv(ROOT / "results/wild_test_labeled.csv", index=False, encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
