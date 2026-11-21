"""Word-level explanation by occlusion (leave-one-word-out).

For a comment with words w1..wn we score the full text and n copies, each
with one word removed. importance(wi) = p(full) - p(without wi).
A large positive value means "this word pushed the comment towards profane".
It is model-agnostic, needs no gradients, and costs n+1 forward passes
(batched), which is fine for review-length texts.
"""
import re

_WORD = re.compile(r"\S+")


def explain(clf, text: str, top_k: int = 3, max_words: int = 80):
    words = _WORD.findall(str(text))[:max_words]
    if not words:
        return {"probability": float(clf.predict_proba([text])[0]), "words": []}
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = clf.predict_proba(variants)
    p_full = float(probs[0])
    imp = [(w, round(p_full - float(p), 3)) for w, p in zip(words, probs[1:])]
    imp.sort(key=lambda x: -x[1])
    return {"probability": round(p_full, 4), "words": imp[:top_k], "all": imp}


def format_explanation(exp, min_importance: float = 0.05) -> str:
    parts = [f"«{w}» ({d:+.2f})" for w, d in exp["words"] if d >= min_importance]
    return "triggered by: " + ", ".join(parts) if parts else "no single decisive word"
