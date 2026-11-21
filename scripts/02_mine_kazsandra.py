"""Step 2 - mine extra examples from the full KazSAnDRA corpus.

KazSAnDRA (ISSAI, 175k Kazakh reviews from Google Play, Kaspi/market,
2GIS/maps and a bookstore) is the corpus the course dataset was sampled
from. Rows already present in kazakh_profanity_share.csv are excluded.

Candidates and how their label is proposed
------------------------------------------
* profane candidates   - any lexicon hit (strong or weak tier)
* hard negatives       - no lexicon hit, but the text contains an innocent
                         look-alike of a profane root (көтеру, сигнал, аман,
                         амал, қатынас, шешім, мал шаруашылығы, Among Us, ...)
* clean negatives      - no lexicon hit, 1-2 star reviews (negative != profane)
* clean random         - no lexicon hit, random reviews from every domain

A "reference" model (char TF-IDF + logistic regression, trained on the
course data AFTER label corrections) scores every candidate. The proposed
label is accepted automatically only when the lexicon and the model agree;
every disagreement is written out for manual review
(data/interim/review_queue.csv). Decisions from that review are stored in
data/manual_labels.csv and applied in step 3.

Run:  python scripts/02_mine_kazsandra.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lexicon import normalize, strong_hits, weak_hits  # noqa: E402
from textutils import anonymize, dedup_key  # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)
RAW = ROOT / "data/raw/kazsandra"
OUT = ROOT / "data/interim"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
course = pd.read_csv(ROOT / "kazakh_profanity_share.csv", encoding="utf-8-sig")
fix = pd.read_csv(ROOT / "data/label_corrections.csv")
course["label"] = course.kazsandra_id.map(fix.set_index("kazsandra_id").new_label).fillna(course.label).astype(int)

ks = pd.concat([pd.read_csv(RAW / f) for f in ["06_sc_train_ib.csv", "09_sc_valid.csv", "10_sc_test.csv"]])
ks = ks.rename(columns={"custom_id": "kazsandra_id", "label": "stars"})
ks["stars"] += 1                       # KazSAnDRA scores are 0..4 -> 1..5 stars
ks = ks[~ks.kazsandra_id.isin(course.kazsandra_id)].copy()
# names are masked later (step 3); here only contacts, so candidate sampling stays stable
ks["text"] = ks.text.astype(str).map(lambda t: anonymize(t, mask_names=False))
ks = ks[ks.text.str.strip().str.len() >= 3]

# drop exact duplicates of the course data and of each other
course_keys = set(course.text.map(dedup_key))
ks["key"] = ks.text.map(dedup_key)
ks = ks[(ks.key.str.len() > 0) & ~ks.key.isin(course_keys)].drop_duplicates("key")

ks["strong"] = ks.text.map(lambda t: ";".join(strong_hits(t)))
ks["weak"] = ks.text.map(lambda t: ";".join(weak_hits(t)))
print("KazSAnDRA rows after removing course rows / duplicates:", len(ks))

# ---------------------------------------------------------------- reference model
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
clf = LogisticRegression(C=4, max_iter=3000, class_weight="balanced")
clf.fit(vec.fit_transform(course.text.map(normalize)), course.label)

# ---------------------------------------------------------------- candidate pools
LOOKALIKE = (r"көтер|котер|сигнал|\bаман|\bамал|қатынас|катынас|шешім|шешімі|шешу|шешіп|мал шаруашылы|"
             r"\bамонг|among|малад[еи]ц|молод[еи]ц|сиқыр|сикыр|\bбокс|\bқотан|\bкотан|амин|"
             r"ешкім|итальян|\bсиге\b|сізге|қаншама|каншама|ташкент|қатыгез|ампер|блендер|блин")
lex_free = ks[(ks.strong == "") & (ks.weak == "")]
pools = {
    "profane_candidate": ks[(ks.strong != "") | (ks.weak != "")],
    "hard_negative": lex_free[lex_free.text.str.lower().str.contains(LOOKALIKE, regex=True)].sample(700, random_state=SEED),
}
rest = lex_free.drop(pools["hard_negative"].index)
pools["clean_negative"] = rest[rest.stars <= 2].sample(1000, random_state=SEED)
rest = rest.drop(pools["clean_negative"].index)
# product reviews (market) are the target domain of the task -> oversampled
pools["clean_random"] = pd.concat([
    rest[rest.domain == "market"].sample(1200, random_state=SEED),
    rest[rest.domain != "market"].sample(800, random_state=SEED),
])

rest = rest.drop(pools["clean_random"].index)
# extra "ordinary reviews" set, used ONLY for testing false positives (never trained on)
clean_eval = pd.concat([
    rest[rest.domain == "market"].sample(1500, random_state=SEED),
    rest[rest.domain != "market"].sample(1500, random_state=SEED),
])

cand = pd.concat([p.assign(pool=name) for name, p in pools.items()])
cand["model_prob"] = clf.predict_proba(vec.transform(cand.text.map(normalize)))[:, 1].round(3)

# ---------------------------------------------------------------- proposed labels
is_prof = cand.pool == "profane_candidate"
strong = cand.strong != ""
p = cand.model_prob
cand["proposed"] = np.where(is_prof, 1, 0)
agree = (
    (is_prof & strong & (p >= 0.5))          # obscene word + model agrees
    | (is_prof & ~strong & (p >= 0.8))       # insult word + model is confident
    | (~is_prof & (p < 0.5))                 # no lexicon hit + model says clean
)
cand["needs_review"] = ~agree
cand.loc[is_prof & ~strong & (p < 0.2), "proposed"] = 0   # likely literal мал/ит/есек -> review as 0

cols = ["kazsandra_id", "pool", "domain", "stars", "model_prob", "proposed", "needs_review", "strong", "weak", "text"]
cand[cols].to_csv(OUT / "mined_candidates.csv", index=False, encoding="utf-8")
cand[cand.needs_review][cols].sort_values(["pool", "model_prob"]).to_csv(OUT / "review_queue.csv", index=False, encoding="utf-8")

clean_eval["model_prob"] = clf.predict_proba(vec.transform(clean_eval.text.map(normalize)))[:, 1].round(3)
clean_eval["needs_review"] = clean_eval.model_prob >= 0.5
clean_eval[["kazsandra_id", "domain", "stars", "model_prob", "needs_review", "text"]].to_csv(
    OUT / "clean_eval_candidates.csv", index=False, encoding="utf-8")
print("clean_eval:", len(clean_eval), "flagged by model for review:", int(clean_eval.needs_review.sum()))

print(pd.crosstab(cand.pool, cand.needs_review, margins=True))
print("auto-accepted label distribution:\n", cand[~cand.needs_review].groupby("pool").proposed.value_counts())
