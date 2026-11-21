"""Kazakh / mixed Kazakh-Russian profanity lexicon.

Used ONLY for data preparation (label audit, candidate mining, hard-negative
search, error analysis). It is not the classifier: the model is trained
separately and never sees these rules.

Two tiers:
  STRONG - obscene vocabulary (Kazakh "боқтық сөз" + Russian mat used inside
           Kazakh comments). A match almost always means label 1.
  WEAK   - insults / vulgar but non-obscene words and ambiguous roots
           ("мал" = cattle, "есек" = donkey, "ақымақ" = fool, "говно", ...).
           A match is a candidate only; the final label needs a human decision.

All patterns run on normalize(text): lower case, Kazakh-specific letters
folded to their Russian-keyboard look-alikes (қ->к, ә->а, і->и, ...), common
Latin transliteration folded to Cyrillic, repeated letters collapsed
("бляяя" -> "бля").  This is how people actually type when they have no
Kazakh keyboard, so one pattern covers "қотақ", "котак", "qotaq", "kotak".
"""
from __future__ import annotations

import re

KZ_FOLD = str.maketrans({
    "қ": "к", "ғ": "г", "ң": "н", "ә": "а", "ө": "о", "ү": "у", "ұ": "у",
    "і": "и", "һ": "х", "ё": "е", "i": "и",
})

# Latin -> Cyrillic (Kazakh Latin 2018/2021 + informal translit). Digraphs first.
_LAT_DIGRAPHS = [("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ya", "я"),
                 ("yu", "ю"), ("ts", "ц")]
_LAT_SINGLE = str.maketrans({
    "a": "а", "á": "а", "ä": "а", "b": "б", "c": "с", "d": "д", "e": "е", "f": "ф",
    "g": "г", "ǵ": "г", "ğ": "г", "h": "х", "ı": "ы", "j": "ж", "k": "к", "l": "л",
    "m": "м", "n": "н", "ń": "н", "ñ": "н", "o": "о", "ó": "о", "ö": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "ş": "ш", "t": "т", "u": "у", "ú": "у", "ü": "у",
    "ū": "у", "v": "в", "w": "у", "x": "х", "y": "ы", "ý": "у", "z": "з",
})


def normalize(text: str) -> str:
    t = str(text).lower()
    if re.search(r"[a-zıáóúýńǵ]", t):
        for a, b in _LAT_DIGRAPHS:
            t = t.replace(a, b)
        t = t.translate(_LAT_SINGLE)
    t = t.translate(KZ_FOLD)
    t = re.sub(r"([а-яa-z])\1+", r"\1", t)   # бляяя -> бля, көөт -> кот (letters only)
    return t


W = r"[а-яa-z*@#$]"          # word character incl. masking symbols
B = rf"(?<!{W})"              # left word boundary
E = rf"(?!{W})"               # right word boundary

STRONG = {
    # --- Kazakh obscene roots ---
    "қотақ (penis)": rf"кота[кгх]|кот[*@#$]+[ак]|кота[*@#$]+бас|{B}кота\s?[пб]ас",
    "көт (ass)": rf"{B}кот(и|ы)?(н|м)?(ди|ды|е|а|ке|тен|нен|ен|не|инди|ынды)?{E}",
    "ам (vulva)": rf"{B}ам(ы|ын|ына|ынды|ыны|ынан|ым|ымды|ыз|дар|дары|дарын|дарынды|га|нын|ски|скы|ский|скии|"
                  rf"шык|шыг|щык|шелек|щелек|бас|бастар|жалак|жегир|ыбила)?{E}|амы?с[иы][гк]|шеше\w{{0,4}}ам",
    "сік/сіг (f*ck)": rf"{B}с[иы][кг](?!нал|ар|ма{E}|р|ыр|ырл|ыр[шл]|унд|ун[тд]|рет|л{E}|тывкар|ан|ер|ир|ак{E}|ыс{E}|ы{E}|ыл|ыш|ым|ынды|ыра|ыр|ен{E}|сен|ак|ымда)\w*|шеше\w{{0,6}}с[иы]?[гк]",
    "шшс (abbr.)": rf"{B}(ш+\.?ш*\.?с+|щ+\.?щ*\.?с+){E}",
    "жәлеп (whore)": rf"{B}жалеп",
    "қаншық (bitch)": rf"{B}каншык",
    "ташақ (balls)": rf"{B}та[шс]а[кг]",
    # --- Russian mat inside Kazakh comments ---
    "х*й": rf"ху[йяюе](?!ж)|{B}нах(у|ю)|ахуе|охуе",
    "п*зд": r"пи[зс]д|пезд",
    "еб-": rf"{B}(за|по|у|вы|до|про|от|на|раз|ни)?[еи]б(а[лнтшй]|ан|ат|у[тч]|ло|ош|ись)|{B}за[йи]б|заеб|долб[ао][её]?б|долбаю?б",
    "бл*ть": rf"{B}(бля|бляд|блят|бляч|блиат|блият|бляк|бляа){W}*|{B}блять",
    "с*ка": rf"{B}(сука|суки|сучк|сукалар|сукасы|сукин|сукилер|суканын|сукасн)",
    "other mat": r"мудак|мудил|пид[оа]р|пидр|гандон|гондон|шлюх|залуп|шалав|пздц|херн(?!анд)|{B}хули{E}|трах[нау]|грёбан|гребан".replace("{B}", B).replace("{E}", E),
    "д*лбаёб": r"д[ао]лб[ао]?[еиыйя]?[ео]?[бп]|д[ао]лб[ао]е?в(ск|тар)|д[ао]лб[ао]ю[бп]|д[ао]лбо[иы]?[оа][бп]|долболоп|д[ао]лбан(?!г)",
    "abbr: зб/дб/667": rf"{B}(зб|дб|ебн){E}|(?<![0-9]){B}667(?![0-9])",
}

WEAK = {
    "мал (cattle, insult)": rf"{B}мал(дар|ски|ский|скй|го|гой|га|сын|сындар|дын|ды|дан|дарды|дардын|дарски|гун)?{E}",
    "есек/ешак (donkey)": rf"{B}(есек|ешак|ешек)",
    "ақымақ (fool)": rf"{B}акы?мак",
    "ит (dog)": rf"{B}(ит|иттер|иттин|итпушык|итсин|итбас){E}",
    "шошқа (pig)": rf"{B}шошка",
    "шешең (your mother)": rf"{B}шеше(н|нин|нын|нди|нды|си|сн|сгин|скин){E}",
    "жалап": rf"{B}жалап",
    "сасық (stinky)": rf"{B}сасык",
    "бла/блат": rf"{B}(бла|блат|бле|блә){E}",
    "лох/дебил/даун": rf"{B}(лох|лохтар|дебил|даун|даундар|идиот|тупой|тупые|урод|уродтар|дура|дурак|чмо|чушпан){E}",
    "говно/жопа": r"говн|дерьм|жопа|жопу|сран|насра|срать",
    "нах/нахрен": rf"{B}(нах|нахрен|нафиг|похер){E}",
    "чорт (devil, slang insult)": r"ч[ое]рт",
    "боқ (shit)": rf"{B}бок(тын|ка|ты|кот|ко|пын|бас|{E})|{B}бок\s|{B}боккот|{B}богы{E}|{B}тыш(ат|ады|ып|ам|ты|а{E})",
    "қатын (derog. woman)": rf"{B}катын(дар|дарски|ски|ым|ды|дарды)?{E}",
    "көттер (asses)": rf"{B}кот+ер(и|ы)н",
}

# masked spellings: each run of masking symbols may hide 1-6 letters
_MASK_TARGETS = [
    "бля", "блять", "блядь", "бл", "сука", "суки", "сучка", "пиздец", "пизда", "пизду", "пздц", "хуй",
    "нахуй", "наху", "нахуя", "заебал", "ебать", "котак", "котакбас", "кот", "мал", "есек", "ешак",
    "сигейин", "сигейн", "сгин", "сиктим", "шешенди", "шешен", "далбаеб", "далбайоп", "говно",
    "ебтваюмать", "етваюмать",
]
_MASK_CHARS = r"[*#@$%]"


def masked_hits(text: str) -> list[str]:
    """Words where some letters were replaced by *, #, @ ... and the skeleton
    fits a profane word, e.g. 'с*ка', 'бл**', 'қо*ақ', 'п*здец'."""
    out = []
    n = normalize(text)
    for tok in re.findall(rf"[а-я]*{_MASK_CHARS}+[а-я{_MASK_CHARS[1:-1]}]*", n):
        core = tok.strip("*#@$%")
        if not re.search(_MASK_CHARS, tok) or not re.match(r"[а-я]", tok):
            # only stars, or a word wrapped in decorative stars ("*топ**")
            continue  # pure symbol runs ("*****", "5*****") are handled in weak_hits
        rx = re.sub(rf"{_MASK_CHARS}+", "[а-я]{0,6}", re.escape(tok).replace("\\", ""))
        if any(re.fullmatch(rx, t) for t in _MASK_TARGETS):
            out.append(tok)
        elif core and re.search(rf"[а-я]{_MASK_CHARS}+[а-я]", tok):
            out.append(tok)          # letter*letter inside a word: deliberate masking
    return out

_STRONG_RX = {k: re.compile(v) for k, v in STRONG.items()}
_WEAK_RX = {k: re.compile(v) for k, v in WEAK.items()}

# tokens that the roots above would catch but are innocent
SAFE_TOKENS = {
    "сигнал", "сигналы", "сигналды", "сигналын", "сигма", "амин", "аминь", "амина",
    "ибадат", "ибн", "блин", "блины", "котер", "котик", "котлет", "аман", "амал",
}


def _hits(rxs: dict, text: str) -> list[str]:
    n = normalize(text)
    out = []
    for name, rx in rxs.items():
        for m in rx.finditer(n):
            tok = m.group(0).strip()
            if tok in SAFE_TOKENS or len(tok) > 30:
                continue
            out.append(name)
            break
    return out


def strong_hits(text: str) -> list[str]:
    hits = _hits(_STRONG_RX, text)
    if masked_hits(text):
        hits.append("masked")
    return hits


def weak_hits(text: str) -> list[str]:
    hits = _hits(_WEAK_RX, text)
    # a whole word replaced by stars: "***** бұндай приложение" (not ratings like "5*****")
    if re.search(r"(?:^|\s)[*#]{3,}(?:\s|$|[!?.,])", str(text)):
        hits.append("fully masked word")
    return hits


def matched_spans(text: str) -> list[str]:
    """Normalized substrings that triggered any rule (for manual review)."""
    n = normalize(text)
    spans = []
    for rx in list(_STRONG_RX.values()) + list(_WEAK_RX.values()):
        spans += [m.group(0) for m in rx.finditer(n) if m.group(0) not in SAFE_TOKENS]
    return spans + masked_hits(text)


if __name__ == "__main__":
    for s in ["Қотақбас ойын", "Сигнал жоқ", "Аман болыңдар", "ШШС", "бляяя жақсы",
              "көтеріп тұр", "Мал ойын", "сыграть невозможно", "qotaqbas", "Шешеңнің амын"]:
        print(f"{s!r:30} strong={strong_hits(s)} weak={weak_hits(s)}")
