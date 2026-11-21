# Dataset card — Kazakh Profanity v2

Binary task: one Kazakh (or mixed Kazakh–Russian) product / app / place review →
`1` = contains Kazakh profanity (obscene words or vulgar insults), `0` = does not.

## 1. Sources

| part | where it comes from | rows |
|---|---|---|
| course data | `kazakh_profanity_share.csv` from the course (3,061 rows sampled from KazSAnDRA) | 3,033 (after removing 28 duplicates) |
| mined profane / insult candidates | full KazSAnDRA corpus, rows with a lexicon hit | 2,634 (2,309 profane / 325 clean after review) |
| hard negatives | KazSAnDRA rows with innocent look-alikes (көтеру, сигнал, аман, Among Us, …) | 699 |
| clean negatives | KazSAnDRA 1–2★ reviews without profanity (*negative ≠ profane*) | 995 (6 turned out profane) |
| clean random | KazSAnDRA random reviews, product reviews (Kaspi/market) oversampled | 1,999 (2 turned out profane) |
| synthetic (train only) | obfuscated copies of train rows — `train_synthetic.csv` | 1,677 |
| **clean_eval** (test only) | 3,000 more ordinary reviews, never trained on | 2,999 (3 profane) |

**KazSAnDRA** — Kazakh Sentiment Analysis Dataset of Reviews and Attitudes (ISSAI, 2024),
~175k reviews from Google Play (appstore), Kaspi and other marketplaces (market),
2GIS (mapping) and an online bookstore. Access: `python scripts/00_download_data.py`
(downloads the public files from https://github.com/IS2AI/KazSAnDRA). Please cite:
Yeshpanov R., Varol H. A. *KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and
Attitudes*, LREC-COLING 2024.

**Relation to product reviews.** Every row is a real user review of a product, an app,
a shop or a place. The course data is ~89% Google Play reviews; to get closer to
"reviews of goods" the clean sets oversample the *market* domain (Kaspi product
reviews), and `clean_eval` is 50% market reviews.

## 2. Labeling rules

Full rules: [`labeling_guidelines.md`](labeling_guidelines.md). Summary:

* **1** — Kazakh obscene words (қотақ, ам, көт, сік-/сіг-, шшс …), Russian mat inside a
  Kazakh / mixed comment (бля, нахуй, пиздец, заебал/зайбал, сука, далбаеб …), or vulgar /
  abusive insults aimed at someone (мал, есек/ешак, ақымақ, тупой, лох, чорт, derog. қатын,
  боқ/говно). This is the convention the course data already follows
  (94% of "мал" rows, 95% of "есек/ешак" rows and 54/55 "чорт" rows were labeled 1).
* **0** — negative reviews without such words, literal senses (мал шаруашылығы, көтеру,
  сигнал, қатынас, Toca Boca …), meta-mentions of swearing, negated use ("мал емес").
* **Mixed language** — Russian mat counts when it is used inside a Kazakh / mixed
  comment. Newly mined rows with no Kazakh at all were excluded (out of scope).
* **Masking & spelling** — с*ка, бл**, қо*ақ count as profanity; Russian-keyboard
  spelling (котак), Latin (qotaq), stretched letters (бляяя) are the same word.
* **Ambiguous** — follow the course majority for that word; otherwise exclude.

Labeled examples:

| text | label | why |
|---|---|---|
| Қотақбас екен | 1 | Kazakh obscene word (was 0 in course data → corrected) |
| Пашол наху жазылмады бла 0☆ | 1 | Russian mat inside a Kazakh comment |
| Б** з@еб@**л мына ойын | 1 | masked mat (бля, заебал) |
| Shshs Mgb sorip alat tartpan | 1 | Latin script, abbreviation шшс |
| Мал ойын тартпаңдар | 1 | "мал" (cattle) used as an insult |
| Жамаан ойын | 0 | negative review, no profanity |
| Ақтөбе облысы бойынша төрт түлік мал | 0 | literal livestock |
| Маған осы амок ас үнайды | 0 | "амок ас" = Among Us (was 1 → corrected) |
| Мал емес ойын екен зорё | 0 | negated: "not a bad game" (was 1 → corrected) |
| …мас күзетші Боқтық сөйлеп өзіме ұрсып тастады | 0 | reports swearing without swearing (was 1 → corrected) |
| Мына тока бока жаман 👎 | 0 | Toca Boca (a game) + "жаман" — negative, not profane |

## 3. How labels were produced

1. **Audit of the course labels** (`scripts/01_audit_labels.py`). Two signals:
   lexicon conflicts and a 5-fold out-of-fold char-TF-IDF model. 200 rows were flagged,
   every one was read; **42 labels were changed** — see
   [`../data/label_corrections.csv`](../data/label_corrections.csv) (each with a reason).
   Typical errors: obvious profanity labeled 0 ("Қотақбас екен", "заибал"), and 1 given
   to clean text ("Шешенстан" = Chechnya, "амок ас" = Among Us, a neutral Kaspi review,
   reports *about* swearing).
2. **Mining** (`scripts/02_mine_kazsandra.py`). A reference model (char TF-IDF + logistic
   regression trained on the corrected course data) scores every candidate. The proposed
   label is kept automatically only when lexicon and model agree
   (`label_source = auto_lexicon+model`); a manual spot-check of 70 such profane rows and
   60 clean rows found no clear errors. All **1,092 disagreements** were reviewed
   by hand against the guidelines (`label_source = manual_review`; LLM-assisted, see §7);
   44 undecidable / non-Kazakh rows were dropped.
3. **clean_eval** rows are lexicon-free and scored < 0.5 by the reference model
   ("silver" 0 labels); the 41 rows the model flagged were read by hand — 3 were in fact
   abusive (labeled 1), 1 was excluded.

## 4. Cleaning, duplicates, splits

* **Anonymisation** (`src/textutils.py`): phone numbers → `<phone>`, @handles → `@user`,
  e-mails → `<email>`, URLs → `<url>`, surnames / patronymics (-ов, -ева, -ұлы, -қызы)
  and "атым X" → `<name>` (also applied to the course rows).
* **Exact duplicates** (same letters after normalisation) removed: 28 (all inside the course data, 19 groups such as "Qotaq" ×5, "Котакбас ойын" ×2; 2 of these groups crossed splits).
* **Near-duplicates**: char 3–5-gram TF-IDF cosine ≥ 0.8 → one group; a group never
  crosses splits. The original course split had **18 near-duplicate groups
  spread over train/val/test** (e.g. "Пиздец ойын" in train, "Пиздец ойын ғо" in test);
  37 course rows were moved so that each group sits in one split
  (priority test → val → train).
* **Split**: course rows keep their course split; new groups are split 80/10/10,
  stratified by label, `SEED = 42`. Synthetic rows exist only in train.

| split | rows | profane | clean | profane % |
|---|---|---|---|---|
| train | 7,504 | 3,001 | 4,503 | 40.0% |
| val | 926 | 409 | 517 | 44.2% |
| test | 930 | 419 | 511 | 45.1% |
| *train_synthetic (extra)* | 1,677 | 978 | 699 | — |
| *clean_eval (extra test)* | 2,999 | 3 | 2,996 | 0.1% |
| **total real rows** | **9,360** | **3,829** | **5,531** | 40.9% |

Test composition by subset:

| test subset | rows | profane |
|---|---|---|
| course (original test rows + 37 moved) | 286 | 174 |
| mined_lexicon_hit | 286 | 244 |
| clean_random | 192 | 1 |
| clean_negative (1–2★, negative ≠ profane) | 93 | 0 |
| hard_negative (look-alike words) | 73 | 0 |
| **total** | **930** | **419** |

By label source: course 283, course_corrected 3, auto_lexicon+model 531, manual_review 113.

## 5. Synthetic examples (train only, `is_synthetic = 1`)

`src/augment.py` makes obfuscated copies of train rows of **both** classes, so that the
surface form is not a label shortcut: masking (қ******с), Latin script (qotaqbas),
Russian-keyboard folding (қ→к, ә→а, і→и), stretched letters, decorative stars
("5*****", "*текст*"). 

| type | profane | clean |
|---|---|---|
| mask (қ******с, с*ка) | 378 | — |
| latin (qotaqbas, zhaqsy) | 250 | 250 |
| kz_keyboard (қ→к, ә→а) | 250 | 250 |
| stretch (мааал, жақсыыы) | 100 | 99 |
| decorative_stars ("5*****") | — | 100 |

## 6. Files

```
data/processed/kazakh_profanity_v2.csv   all real rows + split column
data/processed/{train,val,test}.csv      the same rows, one file per split
data/processed/train_synthetic.csv       synthetic rows (train only, parent_id = source row)
data/processed/clean_eval.csv            ordinary reviews for the false-positive test
data/processed/stats.json                counts used in this card
data/label_corrections.csv               changed course labels + reason
data/interim/review_decisions_part*.csv  manual decisions for mined rows + note
```

Columns: `id` (KazSAnDRA id), `text`, `label`, `split`, `source`
(course / kazsandra_mined / synthetic), `subset`, `label_source`
(course / course_corrected / auto_lexicon+model / manual_review / inherited_from_parent),
`original_label` (course label before correction), `domain`, `stars`, `group`
(near-duplicate group id), `is_synthetic`.

## 7. Limitations

* One annotator convention; the boundary between "insult" and "negative opinion"
  (тупой, ақымақ, мал) is subjective. The rules follow the course data's majority.
* Review of mined disagreements and of the course audit was done with AI assistance
  (Claude) against the written guidelines and spot-checked; it is not a multi-annotator
  gold standard.
* Auto-accepted mined rows contain at least one lexicon word, so they are on average
  "easier" than real-world profanity; the course test rows and the manually reviewed
  rows are the harder part of the test set.
* `clean_eval` labels are silver: a profane review without any lexicon word and with a
  low reference-model score could still be hidden there.
* First names without a surname suffix are not removed automatically.
* Domain skew: most data are app reviews (Google Play); product reviews (market)
  are a minority even after oversampling.

## 8. Does the new data help? (baseline sanity check)

`python scripts/04_check_dataset.py` — the same char-TF-IDF + logistic regression,
threshold 0.5, trained on different training sets, evaluated on the v2 splits.
`test_obfuscated` = the same test rows rewritten with masking / Latin / Russian-keyboard
spelling (generated on the fly, never trained on). This is **not** the final model.

| training data | test F1 | test P / R | test[course rows] F1 | test_obfuscated F1 | false positives on clean_eval |
|---|---|---|---|---|---|
| course, original labels (2,464) | 0.891 | 0.949 / 0.840 | 0.914 | 0.879 | 37 / 2,996 (1.2%) |
| course, corrected labels (2,464) | 0.895 | 0.949 / 0.847 | 0.923 | 0.882 | 38 / 2,996 |
| v2 train (7,504) | **0.944** | 0.958 / 0.931 | **0.945** | 0.922 | **32 / 2,996 (1.1%)** |
| v2 train + synthetic (9,181) | 0.939 | 0.960 / 0.919 | 0.938 | **0.932** | 37 / 2,996 |

* More data mainly raises **recall** (0.85 → 0.93): the model sees many more spellings.
* Synthetic rows trade ~0.5 F1 on ordinary text for +1 F1 on obfuscated text.
* Remaining false positives on ordinary reviews are short angry slang without profanity
  ("зынго осы оиын", "Далбасаго", "Мына тока бока жаман") — exactly the
  "negative ≠ profane" boundary; listed in `data/processed/baseline_clean_eval_false_positives.csv`.
