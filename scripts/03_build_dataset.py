"""Step 3 - build the final dataset (v2) and the splits.

Inputs
  kazakh_profanity_share.csv               course dataset (3,061 rows)
  data/label_corrections.csv               manual fixes of course labels
  data/interim/mined_candidates.csv        step 2 output
  data/interim/review_decisions_part*.csv  manual review of disagreements
  data/interim/clean_eval_candidates.csv   step 2 output (ordinary reviews)
  data/manual_labels_clean_eval.csv        manual review of flagged clean_eval rows

Outputs (data/processed/)
  kazakh_profanity_v2.csv   all rows, column `split` in {train, val, test}
  train.csv / val.csv / test.csv
  train_synthetic.csv       obfuscated copies of TRAIN rows only (is_synthetic=1)
  clean_eval.csv            3,000 ordinary reviews, never used for training,
                            to measure false positives on normal text
  stats.json

Split policy
  * course rows keep the course split (train/val/test) so results stay
    comparable with the original split;
  * exact duplicates (same letters after normalisation) are removed;
  * near-duplicates (char 3-5-gram TF-IDF cosine >= 0.8) form a group and the
    whole group goes to ONE split (a group that touches the course test split
    goes to test, then val, then train);
  * new groups are split 80/10/10 with SEED = 42.

Run:  python scripts/03_build_dataset.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from augment import make_synthetic  # noqa: E402
from lexicon import normalize  # noqa: E402
from textutils import anonymize, dedup_key  # noqa: E402

SEED = 42
NEAR_DUP_THRESHOLD = 0.8
INTERIM, OUT = ROOT / "data/interim", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ course data
course = pd.read_csv(ROOT / "kazakh_profanity_share.csv", encoding="utf-8-sig")
fix = pd.read_csv(ROOT / "data/label_corrections.csv").set_index("kazsandra_id")
course["original_label"] = course.label
course["label"] = course.kazsandra_id.map(fix.new_label).fillna(course.label).astype(int)
course["label_source"] = np.where(course.kazsandra_id.isin(fix.index), "course_corrected", "course")
course["source"], course["subset"] = "course", "course"
course["domain"] = course.kazsandra_id.str[:3]
course["stars"] = np.nan

# ------------------------------------------------------------------ mined data
mined = pd.read_csv(INTERIM / "mined_candidates.csv")
decision_files = sorted(INTERIM.glob("review_decisions_part*.csv"))
decisions = (pd.concat([pd.read_csv(p) for p in decision_files]) if decision_files
             else pd.DataFrame(columns=["kazsandra_id", "label", "note"]))
decisions = decisions.drop_duplicates("kazsandra_id", keep="last").set_index("kazsandra_id")
missing = mined[mined.needs_review & ~mined.kazsandra_id.isin(decisions.index)]
if len(missing):
    print(f"WARNING: {len(missing)} rows need review but have no decision -> dropped")
mined["label"] = np.where(mined.needs_review, mined.kazsandra_id.map(decisions.label), mined.proposed)
mined["review_note"] = mined.kazsandra_id.map(decisions.note)
mined = mined[mined.label.isin([0, 1])].copy()
mined["label"] = mined.label.astype(int)
mined["label_source"] = np.where(mined.needs_review, "manual_review", "auto_lexicon+model")
mined["source"] = "kazsandra_mined"
mined["subset"] = mined.pool.replace({"profane_candidate": "mined_lexicon_hit"})
mined["original_label"] = np.nan

cols = ["kazsandra_id", "text", "label", "source", "subset", "label_source", "original_label", "domain", "stars"]
data = pd.concat([course.assign(split=course.split)[cols + ["split"]],
                  mined.assign(split=None)[cols + ["split"]]], ignore_index=True)

# ------------------------------------------------------------------ anonymise + exact dedup
data["text"] = data.text.astype(str).map(anonymize)
data["key"] = data.text.map(dedup_key)
before = len(data)
data = data[data.key.str.len() > 0]
# keep the course copy when a mined row duplicates it
data = data.sort_values("source").drop_duplicates("key", keep="first").reset_index(drop=True)
print(f"exact duplicates removed: {before - len(data)}")

# ------------------------------------------------------------------ near-duplicate groups
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
X = vec.fit_transform(data.text.map(normalize))
parent = list(range(len(data)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for start in range(0, X.shape[0], 2000):
    sims = (X[start:start + 2000] @ X.T).tocoo()
    for i, j, v in zip(sims.row + start, sims.col, sims.data):
        if i < j and v >= NEAR_DUP_THRESHOLD:
            parent[find(i)] = find(j)
data["group"] = [find(i) for i in range(len(data))]
print("near-duplicate groups with >1 row:", int((data.group.value_counts() > 1).sum()))

# ------------------------------------------------------------------ splits
prio = {"test": 0, "val": 1, "train": 2}
group_split = (data.dropna(subset=["split"]).assign(p=lambda d: d.split.map(prio))
               .groupby("group").p.min().map({v: k for k, v in prio.items()}))
moved = data[data.split.notna() & (data.group.map(group_split) != data.split)]
if len(moved):
    print(f"course rows moved to keep near-duplicates together: {len(moved)}")
data["split"] = data.group.map(group_split)

rng = np.random.default_rng(SEED)
new_groups = data[data.split.isna()].groupby("group").label.max()   # stratify by label
for lab in (0, 1):
    g = new_groups[new_groups == lab].index.to_numpy()
    g = g[rng.permutation(len(g))]
    n_te, n_va = round(0.1 * len(g)), round(0.1 * len(g))
    assign = {**{x: "test" for x in g[:n_te]}, **{x: "val" for x in g[n_te:n_te + n_va]},
              **{x: "train" for x in g[n_te + n_va:]}}
    m = data.split.isna() & data.group.isin(assign.keys())
    data.loc[m, "split"] = data.loc[m, "group"].map(assign)
assert data.split.notna().all()
assert data.groupby("group").split.nunique().max() == 1, "near-duplicate group crosses splits"

data["id"] = data.kazsandra_id
data["is_synthetic"] = 0
final_cols = ["id", "text", "label", "split", "source", "subset", "label_source", "original_label",
              "domain", "stars", "group", "is_synthetic"]
data = data[final_cols].sort_values(["split", "id"]).reset_index(drop=True)
data.to_csv(OUT / "kazakh_profanity_v2.csv", index=False, encoding="utf-8")
for s in ("train", "val", "test"):
    data[data.split == s].to_csv(OUT / f"{s}.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------------ synthetic (train only)
synth = make_synthetic(data[data.split == "train"], seed=SEED)
synth.to_csv(OUT / "train_synthetic.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------------ clean_eval (ordinary reviews)
ce = pd.read_csv(INTERIM / "clean_eval_candidates.csv")
ce_fix = pd.read_csv(ROOT / "data/manual_labels_clean_eval.csv").set_index("kazsandra_id")
ce["label"] = ce.kazsandra_id.map(ce_fix.label).fillna(0).astype(int)
ce["label_source"] = np.where(ce.kazsandra_id.isin(ce_fix.index), "manual_review", "silver_lexicon_free+model")
ce = ce[ce.label.isin([0, 1])].copy()
ce["text"] = ce.text.astype(str).map(anonymize)
ce = ce[~ce.text.map(dedup_key).isin(set(data.text.map(dedup_key)))]   # never overlaps the main data
ce = ce.rename(columns={"kazsandra_id": "id"})[["id", "text", "label", "label_source", "domain", "stars"]]
ce.to_csv(OUT / "clean_eval.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------------ stats
stats = {
    "seed": SEED,
    "near_dup_threshold": NEAR_DUP_THRESHOLD,
    "total": int(len(data)),
    "by_split": {s: {"n": int((data.split == s).sum()),
                     "profane": int(((data.split == s) & (data.label == 1)).sum())} for s in ("train", "val", "test")},
    "by_source_label": data.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
    "by_subset_label": data.groupby(["subset", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
    "by_label_source": data.label_source.value_counts().to_dict(),
    "course_labels_changed": int((data.label_source == "course_corrected").sum()),
    "synthetic_train_rows": int(len(synth)),
    "synthetic_by_type": synth.subset.value_counts().to_dict(),
    "clean_eval": {"n": int(len(ce)), "profane": int(ce.label.sum())},
}
(OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
