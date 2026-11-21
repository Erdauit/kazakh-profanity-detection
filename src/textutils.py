"""Shared helpers: anonymisation and near-duplicate keys."""
import re

from lexicon import normalize

_PHONE = re.compile(r"(?<!\d)(?:\+?7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z0-9_.]{2,}")


# Capitalised surnames / patronymics (Нурпейсова, Табысбаев, Мартқызы, Ахметұлы ...)
# (-ин/-ина are NOT used: they collide with the Kazakh genitive, e.g. "Шешенин амы")
_SURNAME = re.compile(
    r"\b(?:[А-ЯӘҒҚҢӨҰҮҺІЁ][а-яәғқңөұүһіё]{2,}(?:ова|ева|ов|ев|ұлы|улы|қызы|кызы|овна|евна|ович|евич)"
    r"|[А-ЯӘҒҚҢӨҰҮҺІЁ]{3,}(?:ОВА|ЕВА|ОВ|ЕВ|ҰЛЫ|УЛЫ|ҚЫЗЫ|КЫЗЫ))\b"
)
# "менің атым Елжас" / "my name is ..." -> keep the phrase, mask the name
_MY_NAME = re.compile(r"(?i)\b(атым|атымыз|меня зовут)\s+[A-ZА-ЯӘҒҚҢӨҰҮҺІЁ][\wәғқңөұүһіё]+")


def anonymize(text: str, mask_names: bool = True) -> str:
    """Same conventions as the course dataset: <phone>, @user (+ <email>, <url>, <name>)."""
    t = str(text)
    t = _URL.sub("<url>", t)
    t = _EMAIL.sub("<email>", t)
    t = _PHONE.sub("<phone>", t)
    t = _HANDLE.sub("@user", t)
    if mask_names:
        t = _SURNAME.sub("<name>", t)
        t = _MY_NAME.sub(lambda m: m.group(1) + " <name>", t)
    return t


def dedup_key(text: str) -> str:
    """Exact-duplicate key: normalized letters only (ignores case, emoji,
    punctuation, Kazakh/Russian letter variants and repeated letters)."""
    return re.sub(r"[^а-я]", "", normalize(text))
