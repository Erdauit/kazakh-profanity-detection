"""Predict on your own Kazakh comments with the saved model (no retraining).

  python predict.py "Қотақбас ойын екен"            # one or more texts as arguments
  python predict.py --file comments.txt              # one comment per line
  python predict.py                                   # interactive: type, Enter; empty line exits
  python predict.py --no-explain "..."                # skip the word-level explanation
  python predict.py --model models/xlmr_base "..."    # another model folder (default: models/kazroberta)

Output: label (0/1), probability of profanity, the moderation decision
(auto-hide / human review / publish, thresholds from models/.../moderation.json
if present) and the words that drove the prediction (leave-one-word-out).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

from explain import explain, format_explanation  # noqa: E402
from infer import load_classifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("texts", nargs="*")
ap.add_argument("--model", default="models/kazroberta")
ap.add_argument("--file")
ap.add_argument("--no-explain", action="store_true")
args = ap.parse_args()

clf = load_classifier(ROOT / args.model)
mod_path = ROOT / args.model / "moderation.json"
mod = json.loads(mod_path.read_text(encoding="utf-8")) if mod_path.exists() else None
NAMES = {0: "0 = no profanity", 1: "1 = PROFANITY"}


def decision(p):
    if not mod:
        return ""
    if p >= mod["t_hide"]:
        return "auto-hide"
    if p >= mod["t_review"]:
        return "human review"
    return "publish"


def show(texts):
    for t, r in zip(texts, clf.predict(texts)):
        line = f"{NAMES[r['label']]:<17} p={r['probability']:.3f}"
        if mod:
            line += f"  [{decision(r['probability'])}]"
        print(f"{line}  | {t}")
        if not args.no_explain:
            print(f"{'':<19}{format_explanation(explain(clf, t))}")


if args.file:
    show([l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()])
elif args.texts:
    show(args.texts)
else:
    print(f"threshold = {clf.threshold:.3f}. Type a comment (empty line to quit):")
    for line in sys.stdin:
        if not line.strip():
            break
        show([line.strip()])
