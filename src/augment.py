"""Synthetic spelling variants (TRAIN split only, clearly marked is_synthetic=1).

Why: real users write the same profane word as қотақ / котак / qotaq / қо*ақ /
қотааақ. The course data has few masked or Latin-script examples, so the
model would learn "Latin script / stars" from a handful of rows. We create
obfuscated copies of training rows of BOTH classes, so that the surface form
(script, stars, stretched letters) is not by itself a label signal:

  profane rows: mask, latin, kz_keyboard, stretch
  clean rows:   latin, kz_keyboard, stretch, decorative_stars

Every synthetic row keeps the label of its parent and the parent's
near-duplicate group, and is never placed in val/test.
"""
import re

import numpy as np
import pandas as pd

from lexicon import strong_hits, weak_hits

_KZ_KEYBOARD = str.maketrans({"ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у",
                              "һ": "х", "і": "и", "Ә": "А", "Ғ": "Г", "Қ": "К", "Ң": "Н", "Ө": "О",
                              "Ұ": "У", "Ү": "У", "Һ": "Х", "І": "И"})
_LATIN = {"а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e", "ё": "yo",
          "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n",
          "ң": "n", "о": "o", "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u",
          "ү": "u", "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "",
          "ы": "y", "і": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya"}
_VOWELS = "аәеиоөұүыіяюэ"


def to_kz_keyboard(t: str, rng) -> str:
    return t.translate(_KZ_KEYBOARD)


def to_latin(t: str, rng) -> str:
    out = []
    for ch in t:
        low = ch.lower()
        rep = _LATIN.get(low, ch)
        out.append(rep.capitalize() if ch != low and rep else rep)
    return "".join(out)


def stretch(t: str, rng) -> str:
    words = t.split(" ")
    idx = [i for i, w in enumerate(words) if re.search(f"[{_VOWELS}]", w.lower())]
    if not idx:
        return t
    for i in rng.choice(idx, size=min(2, len(idx)), replace=False):
        w = words[i]
        pos = [k for k, c in enumerate(w) if c.lower() in _VOWELS][-1]
        words[i] = w[:pos + 1] + w[pos] * int(rng.integers(2, 5)) + w[pos + 1:]
    return " ".join(words)


def mask_profane(t: str, rng) -> str:
    """Replace inner letters of every profane token with '*': қотақбас -> қ******с."""
    def repl(m):
        w = m.group(0)
        if len(w) < 3 or not (strong_hits(w) or weak_hits(w)):
            return w
        keep_last = rng.random() < 0.5
        inner = len(w) - 1 - int(keep_last)
        return w[0] + "*" * inner + (w[-1] if keep_last else "")
    return re.sub(r"[\wәғқңөұүһі]+", repl, t)


def decorative_stars(t: str, rng) -> str:
    return rng.choice([t + " 5*****", "*" + t + "*", t + " ***", "**" + t + "**"])


PROFANE_PLAN = {"mask": (mask_profane, 400), "latin": (to_latin, 250),
                "kz_keyboard": (to_kz_keyboard, 250), "stretch": (stretch, 100)}
CLEAN_PLAN = {"latin": (to_latin, 250), "kz_keyboard": (to_kz_keyboard, 250),
              "stretch": (stretch, 100), "decorative_stars": (decorative_stars, 100)}


def make_synthetic(train: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = train[train.is_synthetic == 0]
    rows = []
    for label, plan in ((1, PROFANE_PLAN), (0, CLEAN_PLAN)):
        pool = real[(real.label == label) & (real.text.str.len() <= 300)]
        for name, (fn, n) in plan.items():
            cand = pool
            if name == "kz_keyboard":
                cand = pool[pool.text.str.contains("[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")]
            if name == "latin":
                cand = pool[pool.text.str.contains("[а-яәғқңөұүһі]", case=False)]
            cand = cand.sample(min(n, len(cand)), random_state=int(rng.integers(1 << 31)))
            for _, r in cand.iterrows():
                new = fn(r.text, rng)
                if new == r.text:
                    continue
                rows.append({"id": f"{r.id}_syn_{name}", "text": new, "label": label, "split": "train",
                             "source": "synthetic", "subset": f"synthetic_{name}",
                             "label_source": "inherited_from_parent", "original_label": np.nan,
                             "domain": r.domain, "stars": r.stars, "group": r.group,
                             "is_synthetic": 1, "parent_id": r.id})
    return pd.DataFrame(rows)


def obfuscate_for_eval(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Robustness probe: every row rewritten in one of three ways (cycled):
    profane rows -> mask / Latin / Russian keyboard, clean rows -> Latin / keyboard.
    Generated on the fly for evaluation only; never trained on."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["text"] = [
        (mask_profane if (lab == 1 and i % 3 == 0) else to_latin if i % 3 == 1 else to_kz_keyboard)(t, rng)
        for i, (t, lab) in enumerate(zip(df.text, df.label))
    ]
    return out
