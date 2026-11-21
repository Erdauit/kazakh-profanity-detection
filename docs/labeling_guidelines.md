# Labeling guidelines — Kazakh profanity (v2)

Task: one comment / review → **1 = contains Kazakh profanity**, **0 = no profanity**.
"Profanity" here = *боқтық және балағат сөздер*: obscene words **and** vulgar/abusive
insults. This matches how the course dataset was labeled (e.g. 94% of rows with
the insult "мал" and 95% with "есек/ешак" are labeled 1); these rules make that
convention explicit and are applied to the original data and to every new row.

## Label 1 — the comment contains at least one of

**A. Kazakh obscene words** (any spelling)
қотақ / қотақбас, ам / амы / амын / амшық, көт / көтің / көттеріңді (vulgar "ass"),
сік- / сіг- (сігейін, сіктім, сігіл), ташақ, жәлеп, қаншық, шешеңді …, abbreviations
шшс / щщс / 667.

**B. Russian (or English) mat used inside a Kazakh or mixed comment**
х*й (нахуй, нихуя, похуй), п*зд- (пиздец, пиздес), еб- (ебать, заебал, зайбал, заибал),
бля / блять / блят, сука / сучка, долбоёб (далбаеб, далбайоп), пидор / пидараз,
гандон, хули, херня, fuck; abbreviations зб, дб.
Code-switching is how many Kazakh speakers swear, so Russian mat in a Kazakh
comment counts as profanity in this task.

**C. Vulgar or abusive insults aimed at a person, group, company or the app/product**
мал / малдар / малски ("cattle"), есек / ешак / ешакбас ("donkey"), ит ("dog"),
шошқа ("pig"), ақымақ, тупой, дебил, даун, лох, чорт, қатын (derogatory),
боқ / говно / гавно ("shit"), жопа, сасық (as an insult).

## Label 0

* Negative or angry reviews **without** such words: жаман, нашар, ұнамады, далбаса,
  фу, тфу, блин, ой, "ужас". *A negative review is not automatically profane.*
* Literal / neutral senses: мал шаруашылығы, мал базары, малдың еті (livestock),
  ит / есек as animals, көтеру / көтерілу (to raise), сигнал, аман / амал / амин,
  қатынас (relation), қатыным (my wife, neutral), шешім / шешу (decision),
  Among Us ("амонг ас", "амок ас"), малодец / малатца (молодец), Russian "мало",
  тышқан (mouse), бокс (boxing), Есекең (a character's name).
* **Meta mentions** — talking *about* swearing without swearing:
  "боқтауға болмайды", "күзетші боқтық сөйлеп ұрысты".
* **Negated use**: "мал емес ойын" ("not a bad game").
* Keyboard mash where a profane-looking letter run appears by accident.

## Spelling variation, masking, scripts

* Russian-keyboard spelling (котак for қотақ), Latin script (qotaq, kanshyk, malsyndar),
  repeated letters (бляяя), merged words (шешенамы) are treated exactly like the
  standard spelling.
* **Masked words count as profanity** when the skeleton clearly fits a profane word:
  с*ка, бл**, қо*ақ, п*здец, ше***. The writer still chose a profane word.
  A whole word replaced by stars with no recoverable skeleton ("*****") is labeled
  from context; rating stars ("5*****") are not masking.
* "секс", "18+" and sexual topics are **not** profanity by themselves.

## Clarifications added during review of mined rows

* **Idioms and similes → 0**: "адамды ақымақ қылу" (to fool people), "ит қорлық",
  "ит болдық", "малға әңгіме айтқандай", "мал сияқты адамдар".
* **Quoting**: a quoted obscene / mat word (A, B) is still 1 ("операторлар сука десе");
  a quoted or reported insult (C) that the writer does not aim at anyone is 0
  ("ойынды говно дейді", "тупой деген не сөз").
* **Excretion verbs** (тышу, обосраться): literal or about oneself → 0
  ("тышып қойдым"); used to run down a product → 1 ("pubg тышады").
* **Minced curses** that keep the curse frame count like masking:
  "шешеңді сыйлайын" (for шешеңді сігейін) → 1.
* **"бла / блә / blaaa" as an angry interjection** = бля → 1; "бла бла" (blah-blah) → 0.
* **топас** (= тупой) aimed at something → 1.
* **Only emoji** (🖕, 💩) with no word → 0: the label is about words.
* Rows in Uzbek, Kyrgyz or Russian only → excluded from mined data.

## Ambiguous cases

* If the decisive word has two plausible readings and context does not resolve it,
  follow the majority convention of the course data for that word
  (per-word counts: `python scripts/01_audit_labels.py`, see DATASET.md §2).
* If it still cannot be decided, the row is **excluded** (label −1 during review)
  rather than guessed. Excluded rows are listed in `data/manual_labels.csv`.
* Comments written entirely in another language (no Kazakh at all) are excluded
  from newly mined data (out of scope), but kept in the original course data.
