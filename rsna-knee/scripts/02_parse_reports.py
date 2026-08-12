#!/usr/bin/env python3
"""Parse multilingual reports into four-state weak labels, and audit the parser.

    # produce weak labels
    python scripts/02_parse_reports.py --reports data/train_reports.csv \
        --out artifacts/weak_labels.parquet

    # THIS FIRST: what is the parser missing, per language?
    python scripts/02_parse_reports.py --reports data/train_reports.csv \
        --audit --top 40

    # how good are the weak labels really?  (needs gold labels)
    python scripts/02_parse_reports.py --reports data/train_reports.csv \
        --labels data/train_labels.csv --evaluate

The audit mode is the point, and it should be run before the weak labels are
used for anything. The shipped lexicon in ``kairos/text/ontology.py`` is a
starting point assembled from clinical vocabulary, **not** from this corpus.
``--audit`` dumps the highest-frequency sentences that matched nothing, grouped
by detected language, so the lexicon can be grown from what the reports
actually say. A rule extractor built from a real frequency list beats a neural
extractor trained on someone else's anatomy, and unlike the neural one you can
read it.

``--evaluate`` is the other guard rail. It scores the weak labels against gold
per label and per language, and reports sensitivity separately from
specificity, because the characteristic failure of a report parser is 0.99
specificity with 0.4 sensitivity -- which looks excellent on accuracy and is
useless as supervision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from kairos.constants import ASSERTION_SOFT_TARGET, TARGETS, Assertion
from kairos.text.ontology import KneeOntology, normalise_text, split_sentences


def detect_language(text: str) -> str:
    """Cheap script/stopword language identification.

    Deliberately not a model: at this stage we only need to *bucket the audit
    output*, and a 200 KB fastText model is a dependency the Kaggle notebook
    would have to carry. Replace with fastText or a transformer LID once the
    language distribution is known from the audit -- and check its accuracy on
    the reports themselves, since radiology prose is not the newswire text
    these models are trained on.
    """
    t = text.lower()
    # Kana BEFORE the CJK ideograph range: Japanese is written with kanji from
    # that same range, so checking CJK first labels every Japanese report as
    # Chinese -- which silently merges two languages in the audit output and
    # hides the gaps in whichever lexicon is weaker.
    if re.search(r"[぀-ゟ゠-ヿ]", t):
        return "ja"
    if re.search(r"[가-힯]", t):
        return "ko"
    if re.search(r"[一-鿿]", t):
        return "zh"
    # Greek was 7.5 % of the RSNA-2026 corpus and fell entirely into "unknown",
    # so its reports were parsed with no language-specific negation rules at all.
    if re.search(r"[Ͱ-Ͽἀ-῿]", t):
        return "el"
    if re.search(r"[Ѐ-ӿ]", t):
        # Bulgarian and Russian share the script, and the whole Bulgarian
        # subcorpus was being labelled "ru".  A single orthographic test is not
        # enough: ы/э are decisive for Russian and ъ-as-a-vowel for Bulgarian,
        # but a given sentence may contain none of them.  Score both.
        ru_markers = ("ы", "э", " не ", "выявл", "определя", "визуализ",
                      "суставн", "признак", "отмечает")
        bg_markers = ("ъ", " на ", "нормално", "изобразяване", "б.о.",
                      "особености", "ставен", "излив", "запазен", "данни за")
        ru = sum(t.count(m) for m in ru_markers)
        bg = sum(t.count(m) for m in bg_markers)
        return "bg" if bg > ru else "ru"
    markers = {
        "tr": (" ve ", " ile ", "izlen", "mevcut", "bulgu", "değişiklik"),
        "es": (" el ", " la ", " de la ", "con ", "sin ", "articul"),
        "pt": (" da ", " do ", " não ", "articula", "com "),
        "fr": (" le ", " la ", " des ", "avec ", "sans ", "articul"),
        "de": (" der ", " die ", " und ", "kein", "gelenk", "nachweis"),
        "it": (" il ", " della ", " con ", "artic", "non "),
        "nl": (" het ", " van ", " met ", "geen ", "gewricht"),
        # Croatian/Serbian/Bosnian.  It was landing in "pl" because Polish's
        # " bez " marker also matches Croatian "bez znakova", and in "pt" by
        # score ties -- so two distinct languages were merged into buckets
        # whose negation cues fit neither.  Listed before "pl" only for
        # readability; selection is by score, so the markers must discriminate:
        # đ/ć/č/š/ž and these stems do not occur in Polish.
        "hr": ("održan", "bez znakova", "primjeren", "prikaz", "menisk",
               "hrskavic", "ligament je", "koštan", "izljev", "uredne"),
        "pl": (" nie ", " oraz ", "staw", "więzadł", "łąkotk", "prawidłow"),
        "en": (" the ", " of the ", " with ", " no ", "joint", "signal"),
    }
    scores = {k: sum(t.count(m) for m in ms) for k, ms in markers.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def write_table(df, path: Path) -> Path:
    """Write parquet, falling back to CSV when no parquet engine is installed.

    Losing a completed corpus scan to a missing optional dependency is a bad
    trade; the caller gets a file either way, and a clear note about which.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        print("note: no parquet engine installed (pip install pyarrow); wrote CSV")
        return alt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", required=True, type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--text-column", default=None)
    ap.add_argument("--audit", action="store_true",
                    help="dump the most frequent unmatched sentences per language")
    ap.add_argument("--evaluate", action="store_true",
                    help="score weak labels against gold (requires --labels)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-confidence", type=float, default=0.6)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.reports)
    text_col = args.text_column or next(
        (c for c in df.columns
         if any(k in c.lower() for k in ("report", "text", "impression", "finding"))),
        None,
    )
    if text_col is None:
        print(f"no text column found in {list(df.columns)}; pass --text-column",
              file=sys.stderr)
        return 2
    uid_col = next((c for c in df.columns if "study" in c.lower()), df.columns[0])
    print(f"{len(df)} reports; text column {text_col!r}, uid column {uid_col!r}")

    onto = KneeOntology()
    languages = Counter()
    unmatched: dict[str, Counter] = defaultdict(Counter)
    rows, states, confs = [], [], []
    matched_sentences = 0
    total_sentences = 0

    for _, r in df.iterrows():
        text = str(r[text_col] or "")
        lang = detect_language(text)
        languages[lang] += 1

        matches = onto.extract(text, language=lang)
        weak = onto.to_weak_labels(matches)

        state = np.array([int(weak[t][0]) for t in TARGETS], dtype=np.int8)
        conf = np.array([weak[t][1] for t in TARGETS], dtype=np.float32)
        states.append(state)
        confs.append(conf)
        rows.append(str(r[uid_col]))

        if args.audit:
            hit_spans = {m.sentence for m in matches}
            for sent in split_sentences(normalise_text(text)):
                total_sentences += 1
                if sent in hit_spans:
                    matched_sentences += 1
                elif len(sent) > 12:
                    unmatched[lang][sent[:110]] += 1

    states = np.stack(states) if states else np.zeros((0, len(TARGETS)), np.int8)
    confs = np.stack(confs) if confs else np.zeros((0, len(TARGETS)), np.float32)

    # -- coverage report -------------------------------------------------- #
    print("\nlanguages detected:")
    for k, v in languages.most_common():
        print(f"  {k:<10} {v:>6}  ({v / max(len(df), 1):.1%})")
    if languages.get("unknown", 0) > 0.1 * len(df):
        print("  !! >10% unknown: the LID markers need the real corpus")

    print("\nper-label assertion coverage (fraction of reports):")
    w = max(len(t) for t in TARGETS) + 2
    print("label".ljust(w) + "  positive  negative uncertain  not-mentioned")
    for l, t in enumerate(TARGETS):
        c = Counter(states[:, l].tolist())
        n = max(len(states), 1)
        print(f"{t.ljust(w)}  "
              f"{c[int(Assertion.POSITIVE)] / n:>8.1%}  "
              f"{c[int(Assertion.NEGATIVE)] / n:>8.1%}  "
              f"{c[int(Assertion.UNCERTAIN)] / n:>8.1%}  "
              f"{c[int(Assertion.NOT_MENTIONED)] / n:>12.1%}")
        if c[int(Assertion.NOT_MENTIONED)] / n > 0.9:
            print(f"{'':<{w}}  !! never mentioned -- lexicon gap for this label")

    # -- audit ------------------------------------------------------------ #
    if args.audit:
        print(f"\nsentence match rate: {matched_sentences}/{total_sentences} "
              f"({matched_sentences / max(total_sentences, 1):.1%})")
        print("\n" + "=" * 70)
        print("MOST FREQUENT UNMATCHED SENTENCES -- grow the lexicon from these")
        print("=" * 70)
        for lang, ctr in sorted(unmatched.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"\n--- {lang} ({sum(ctr.values())} unmatched) ---")
            for sent, n in ctr.most_common(args.top):
                print(f"  {n:>5}  {sent}")
        if args.out:
            args.out.with_suffix(".audit.json").write_text(json.dumps(
                {k: dict(v.most_common(200)) for k, v in unmatched.items()},
                indent=2, ensure_ascii=False,
            ))
            print(f"\nfull audit written to {args.out.with_suffix('.audit.json')}")

    # -- evaluation against gold ------------------------------------------ #
    if args.evaluate:
        if not args.labels:
            print("--evaluate requires --labels", file=sys.stderr)
            return 2
        gold = pd.read_csv(args.labels).set_index(
            next(c for c in pd.read_csv(args.labels, nrows=1).columns
                 if "study" in c.lower())
        )
        common = [u for u in rows if u in gold.index]
        idx = [rows.index(u) for u in common]
        y = gold.loc[common, list(TARGETS)].to_numpy(dtype=float)
        s = states[idx]
        c = confs[idx]

        print(f"\nevaluating weak labels on {len(common)} studies with gold")
        print("label".ljust(w) + "   sens    spec    PPV   n_pos_pred  coverage")
        for l, t in enumerate(TARGETS):
            pred_pos = (s[:, l] == int(Assertion.POSITIVE)) & (c[:, l] >= args.min_confidence)
            covered = s[:, l] != int(Assertion.NOT_MENTIONED)
            gy = y[:, l] > 0.5
            tp = int((pred_pos & gy).sum())
            fn = int((~pred_pos & gy).sum())
            fp = int((pred_pos & ~gy).sum())
            tn = int((~pred_pos & ~gy).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)
            print(f"{t.ljust(w)}  {sens:>6.3f} {spec:>7.3f} {ppv:>6.3f} "
                  f"{int(pred_pos.sum()):>11} {covered.mean():>9.1%}")
        print("\nRead the sensitivity column, not the specificity one. A parser at "
              "0.99 spec / 0.40 sens\nlooks excellent on accuracy and is nearly "
              "useless as supervision.")

    # -- write ------------------------------------------------------------ #
    if args.out:
        soft = np.full_like(confs, np.nan, dtype=np.float32)
        for st, val in ASSERTION_SOFT_TARGET.items():
            soft[states == int(st)] = val
        out = pd.DataFrame({"StudyInstanceUID": rows})
        for l, t in enumerate(TARGETS):
            out[f"weak_{t}"] = soft[:, l]
            out[f"conf_{t}"] = confs[:, l]
            out[f"state_{t}"] = states[:, l]
        out["language"] = [detect_language(str(v or "")) for v in df[text_col]]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        written = write_table(out, args.out)
        print(f"\nwrote {written} ({len(out)} rows)")
        print("Feed this to 04_train.py so the 'weak_label' objective can be enabled.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
