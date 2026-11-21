"""Text -> model input preprocessing (identical for training and inference).

Rule-based, nothing is learned, so there is nothing to fit on train only:
  1. anonymise contacts / names (same as the dataset)        -> <phone>, @user ...
  2. lower-case                                               ҚОТАҚБАС -> қотақбас
  3. collapse 3+ repeats of the same character to 2           бляяяя -> бляя, 😡😡😡😡 -> 😡😡
  4. normalise whitespace
Kazakh letters, punctuation, '*' masks and emoji are KEPT: the sub-word
tokenizers (SentencePiece / byte-level BPE) split any string into known pieces, so there are
no unknown words; '*' inside 'с*ка' becomes its own token the model can use.
"""
import re

from textutils import anonymize

PREPROCESS_VERSION = "v1"
_REPEAT = re.compile(r"(.)\1{2,}", flags=re.S)
_SPACE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    t = anonymize(str(text))
    t = t.lower()
    t = _REPEAT.sub(r"\1\1", t)
    t = _SPACE.sub(" ", t).strip()
    return t
