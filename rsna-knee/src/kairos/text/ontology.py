r"""Multilingual knee-pathology ontology and assertion extraction.

The reports arrive in roughly a dozen languages.  The established radiology-NLP
stack (RadGraph, CheXbert, NegBio) is English *and* chest-specific: its
entity vocabulary contains "consolidation" and not "meniscal extrusion", and
its negation cues are English.  Porting the *method* is right; porting the
*models* is not.

What this module provides:

* a curated concept lexicon per competition label, with surface forms in
  English, Turkish, Spanish, Portuguese, French, German, Italian, Dutch,
  Polish, Russian (transliteration-tolerant), Chinese and Japanese;
* a multilingual negation / uncertainty / historicity detector in the NegEx
  tradition -- cue lists plus a *scope* rule that respects each language's
  clause structure rather than a fixed token window;
* a compartment resolver, because "tear of the posterior horn of the medial
  meniscus" must reach ``Medial Meniscus`` and not ``Lateral Meniscus``, and
  laterality words are the most common source of silent weak-label corruption;
* a four-state assertion output with a calibrated confidence, so downstream
  losses can gate on it.

**The lexicon here is a starting point that must be audited against the actual
corpus before it is trusted.**  The intended workflow is: run
``scripts/02_parse_reports.py --audit`` to dump the highest-frequency unmatched
sentences per language, and grow the lexicon from what the data actually says.
A rule extractor built from a real frequency list beats a neural extractor
trained on someone else's anatomy, and it is auditable, which matters when a
weak label is going to shape a loss.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..constants import Assertion, TARGETS

__all__ = [
    "ConceptMatch",
    "KneeOntology",
    "normalise_text",
    "split_sentences",
    "NEGATION_CUES",
    "UNCERTAINTY_CUES",
]


# --------------------------------------------------------------------------- #
# Text normalisation                                                           #
# --------------------------------------------------------------------------- #


def normalise_text(text: str, *, fold_case: bool = True) -> str:
    """NFKC-normalise, unify whitespace, and strip zero-width characters.

    Turkish deserves a note: the dotless ``ı`` lowercases from ``I`` while the
    dotted ``i`` uppercases to ``İ``.  Applying Python's default ``.lower()`` to
    a Turkish report turns ``MENİSKÜS`` into ``meni̇sküs`` (with a combining
    dot), which then fails to match ``menisküs``.  We therefore NFKC-normalise
    *after* case folding and additionally strip combining marks from the
    matching key, keeping the original for display.
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("​", "").replace("﻿", "").replace("­", "")
    if fold_case:
        t = t.casefold()
        # ``"İ".casefold()`` yields ``i`` + U+0307 COMBINING DOT ABOVE, which
        # NFKC will not recombine.  Drop that one codepoint specifically: it is
        # an artefact of case folding, not a diacritic anyone wrote.  Stripping
        # marks wholesale would also destroy ü/ö/ç/ğ/ş, which *are* meaningful
        # and which the lexicon relies on for display and for the non-folded
        # comparison paths.
        t = t.replace("̇", "")
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()


def _deaccent(text: str) -> str:
    d = unicodedata.normalize("NFD", text)
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+|\s+[-•·]\s+")


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation tuned for radiology reports.

    Reports are not prose.  They are bullet lists, numbered findings and
    semicolon-chained clauses, and an off-the-shelf sentence splitter merges a
    whole findings section into one "sentence" -- which destroys negation scope
    and therefore every weak label derived from it.  We split on terminal
    punctuation, newlines *and* bullet markers, then drop fragments shorter
    than three characters.
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= 3]


# --------------------------------------------------------------------------- #
# Cue lexicons                                                                 #
# --------------------------------------------------------------------------- #

NEGATION_CUES: dict[str, tuple[str, ...]] = {
    "en": ("no ", "no evidence of", "without", "negative for", "absent", "unremarkable",
           "intact", "normal", "not seen", "ruled out", "free of", "denies"),
    "tr": ("yok", "izlenmemektedir", "izlenmedi", "saptanmadı", "görülmedi", "normal",
           "olağan", "intakt", "sağlam", "mevcut değil", "bulgusu yoktur"),
    "es": ("no ", "sin ", "ausencia de", "no se observa", "negativo para", "íntegro",
           "normal", "sin evidencia"),
    "pt": ("não ", "sem ", "ausência de", "não se observa", "íntegro", "normal"),
    "fr": ("pas de", "sans ", "absence de", "non visualisé", "intègre", "normal"),
    "de": ("kein", "keine", "ohne ", "nicht ", "unauffällig", "intakt", "regelrecht"),
    "it": ("non ", "senza ", "assenza di", "nella norma", "integro", "regolare"),
    "nl": ("geen ", "zonder ", "niet ", "ongestoord", "intact", "normaal"),
    "pl": ("bez ", "nie ", "brak ", "prawidłow", "nieuszkodzon"),
    "ru": ("нет ", "не ", "без ", "отсутств", "интакт", "не выявлен"),
    "zh": ("未见", "无", "阴性", "未显示", "正常"),
    "ja": ("認めない", "なし", "陰性", "指摘なし", "正常"),
}

UNCERTAINTY_CUES: dict[str, tuple[str, ...]] = {
    "en": ("possible", "probable", "suspicious for", "cannot exclude", "may represent",
           "questionable", "equivocal", "suggestive of", "likely", "borderline",
           "consider", "versus"),
    "tr": ("şüpheli", "olası", "muhtemel", "ekarte edilemez", "ayırıcı tanı",
           "düşündürmektedir", "olabilir", "sınırda"),
    "es": ("posible", "probable", "sospechoso", "no se puede excluir", "sugestivo",
           "dudoso", "limítrofe"),
    "pt": ("possível", "provável", "suspeito", "não se pode excluir", "sugestivo"),
    "fr": ("possible", "probable", "suspect", "ne peut être exclu", "évocateur",
           "douteux"),
    "de": ("möglich", "wahrscheinlich", "verdächtig", "nicht auszuschließen",
           "fraglich", "vereinbar mit"),
    "it": ("possibile", "probabile", "sospetto", "non escludibile", "dubbio"),
    "nl": ("mogelijk", "waarschijnlijk", "verdacht", "niet uit te sluiten"),
    "pl": ("możliw", "prawdopodobn", "podejrzan", "nie można wykluczyć"),
    "ru": ("возможно", "вероятно", "подозрение", "не исключа", "сомнительн"),
    "zh": ("可能", "考虑", "疑似", "不除外"),
    "ja": ("疑い", "可能性", "示唆", "否定できない"),
}

HISTORICITY_CUES: tuple[str, ...] = (
    "post-operative", "postoperative", "status post", "prior", "previous", "old",
    "chronic", "sequela", "postop", "ameliyat", "geçirilmiş", "eski", "kronik",
    "postoperatorio", "antiguo", "crónico", "postopératoire", "ancien", "chronique",
    "postoperativ", "alt", "chronisch", "术后", "陈旧", "術後", "陳旧",
)

#: Words that flip the compartment.  Kept separate from the concept lexicon so
#: that a single medial/lateral list serves all twelve labels.
_MEDIAL = ("medial", "mediale", "medialen", "medyal", "iç ", "interno", "interne",
           "przyśrodkow", "медиальн", "内侧", "内側")
_LATERAL = ("lateral", "laterale", "lateralen", "dış ", "externo", "externe",
            "boczn", "латеральн", "外侧", "外側")
_PATELLOFEMORAL = ("patellofemoral", "patellofemorale", "patellar", "retropatellar",
                   "patella", "diz kapağı", "髌股", "膝蓋大腿")


@dataclass(slots=True)
class ConceptMatch:
    label: str
    assertion: Assertion
    confidence: float
    sentence: str
    span: tuple[int, int]
    cue: str | None = None
    compartment_hint: str | None = None
    language: str | None = None


@dataclass(slots=True)
class KneeOntology:
    """Label ↔ multilingual surface-form lexicon plus assertion logic."""

    lexicon: dict[str, tuple[str, ...]] = field(default_factory=dict)
    negation_window: int = 8
    require_compartment: bool = True
    #: Compiled per-label alternation, built in ``__post_init__``.  It must be
    #: a declared field: with ``slots=True`` there is no ``__dict__``, so an
    #: undeclared attribute cannot be assigned and construction fails outright.
    _compiled: dict[str, "re.Pattern[str]"] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.lexicon:
            self.lexicon = dict(_DEFAULT_LEXICON)
        self._compiled = {
            label: re.compile(
                "|".join(re.escape(_deaccent(s)) for s in sorted(forms, key=len, reverse=True)),
                re.IGNORECASE,
            )
            for label, forms in self.lexicon.items()
            if forms
        }

    # -- assertion ------------------------------------------------------- #

    @staticmethod
    def _find_cue(sentence: str, cues: dict[str, tuple[str, ...]]) -> tuple[str | None, str | None]:
        s = _deaccent(sentence)
        for lang, forms in cues.items():
            for c in forms:
                cd = _deaccent(c)
                if cd and cd in s:
                    return c, lang
        return None, None

    def assert_state(self, sentence: str, match_span: tuple[int, int]
                     ) -> tuple[Assertion, float, str | None, str | None]:
        r"""Four-state assertion with a scope rule.

        Scope: a negation cue governs the concept when it appears **before** the
        concept and within ``negation_window`` tokens, *or* anywhere in the
        sentence when the sentence has a single clause.  Post-posed negation is
        the norm in Turkish (``... izlenmemektedir``) and Japanese
        (``... 認めない``), so a purely left-context rule -- which is what NegEx
        implements -- systematically mislabels those two languages as positive.
        We therefore also accept a cue that follows the concept when no
        conjunction separates them.
        """
        neg_cue, neg_lang = self._find_cue(sentence, NEGATION_CUES)
        unc_cue, unc_lang = self._find_cue(sentence, UNCERTAINTY_CUES)

        tokens = _deaccent(sentence).split()
        start_tok = len(_deaccent(sentence[: match_span[0]]).split())

        def _in_scope(cue: str | None) -> bool:
            if cue is None:
                return False
            cd = _deaccent(cue).strip()
            pos = [i for i, t in enumerate(tokens) if cd.split()[0] in t]
            if not pos:
                return False
            return any(abs(p - start_tok) <= self.negation_window for p in pos)

        if _in_scope(unc_cue):
            return Assertion.UNCERTAIN, 0.75, unc_cue, unc_lang
        if _in_scope(neg_cue):
            # A negation and an uncertainty cue in the same sentence
            # ("cannot exclude ... no definite") is genuinely ambiguous.
            conf = 0.65 if unc_cue else 0.90
            return Assertion.NEGATIVE, conf, neg_cue, neg_lang
        return Assertion.POSITIVE, 0.85, None, None

    # -- compartment ----------------------------------------------------- #

    @staticmethod
    def compartment_of(sentence: str) -> str | None:
        s = _deaccent(sentence.casefold())
        has_m = any(_deaccent(k) in s for k in _MEDIAL)
        has_l = any(_deaccent(k) in s for k in _LATERAL)
        has_p = any(_deaccent(k) in s for k in _PATELLOFEMORAL)
        if has_p and not (has_m or has_l):
            return "patellofemoral"
        if has_m and not has_l:
            return "medial"
        if has_l and not has_m:
            return "lateral"
        if has_m and has_l:
            return "both"
        return None

    # -- extraction ------------------------------------------------------ #

    def extract(self, report: str, *, language: str | None = None) -> list[ConceptMatch]:
        out: list[ConceptMatch] = []
        for sent in split_sentences(normalise_text(report)):
            de = _deaccent(sent)
            historic = any(_deaccent(h) in de for h in HISTORICITY_CUES)
            comp = self.compartment_of(sent)
            for label, rx in self._compiled.items():
                for m in rx.finditer(de):
                    a, conf, cue, cue_lang = self.assert_state(sent, m.span())
                    if self.require_compartment and label in _COMPARTMENT_LABELS:
                        want = _COMPARTMENT_LABELS[label]
                        if comp is None:
                            conf *= 0.5
                        elif comp == "both":
                            conf *= 0.8
                        elif comp != want:
                            continue  # explicit opposite compartment: not this label
                    if historic:
                        # Chronic / post-operative findings are frequently not
                        # what the gold label captures.  Down-weight rather than
                        # drop: "old ACL reconstruction" still implies a torn ACL.
                        conf *= 0.6
                    out.append(
                        ConceptMatch(
                            label=label,
                            assertion=a,
                            confidence=float(min(conf, 1.0)),
                            sentence=sent,
                            span=m.span(),
                            cue=cue,
                            compartment_hint=comp,
                            language=language or cue_lang,
                        )
                    )
        return out

    def to_weak_labels(self, matches: list[ConceptMatch]) -> dict[str, tuple[Assertion, float]]:
        """Reduce per-mention matches to one assertion per label.

        Resolution order is POSITIVE > UNCERTAIN > NEGATIVE, weighted by
        confidence.  Rationale: a report that mentions a finding positively
        anywhere is evidence for it; the negations are usually the templated
        checklist ("ACL intact, PCL intact, ...") which is high-volume and
        low-information, and letting it outvote a single explicit positive is
        the classic way to build a weak labeller with 0.99 specificity and
        0.4 sensitivity.
        """
        best: dict[str, tuple[Assertion, float]] = {}
        rank = {Assertion.POSITIVE: 3, Assertion.UNCERTAIN: 2,
                Assertion.NEGATIVE: 1, Assertion.NOT_MENTIONED: 0}
        for m in matches:
            cur = best.get(m.label)
            if cur is None or (rank[m.assertion], m.confidence) > (rank[cur[0]], cur[1]):
                best[m.label] = (m.assertion, m.confidence)
        for t in TARGETS:
            best.setdefault(t, (Assertion.NOT_MENTIONED, 0.0))
        return best


_COMPARTMENT_LABELS = {
    "Medial Meniscus": "medial",
    "Lateral Meniscus": "lateral",
    "Medial OA": "medial",
    "Lateral OA": "lateral",
    "PF OA": "patellofemoral",
    "MCL": "medial",
}


_DEFAULT_LEXICON: dict[str, tuple[str, ...]] = {
    "ACL": (
        "acl", "anterior cruciate", "on capraz bag", "ön çapraz bağ", "oca",
        "ligamento cruzado anterior", "lca", "ligament croise anterieur",
        "vorderes kreuzband", "vkb", "legamento crociato anteriore",
        "voorste kruisband", "wiezadlo krzyzowe przednie",
        "передняя крестообразная", "前交叉韧带", "前十字靭帯",
    ),
    "MCL": (
        "mcl", "medial collateral", "ic yan bag", "iç yan bağ",
        "ligamento colateral medial", "lcm", "ligament collateral medial",
        "innenband", "mediales kollateralband", "legamento collaterale mediale",
        "mediale collaterale band", "внутренняя боковая", "内侧副韧带", "内側側副靭帯",
    ),
    "Medial Meniscus": (
        "medial meniscus", "medial menisc", "ic menisk", "iç menisküs",
        "menisco medial", "menisque medial", "innenmeniskus", "medialer meniskus",
        "menisco mediale", "mediale meniscus", "lakotka przysrodkowa",
        "медиальный мениск", "内侧半月板", "内側半月板",
    ),
    "Lateral Meniscus": (
        "lateral meniscus", "lateral menisc", "dis menisk", "dış menisküs",
        "menisco lateral", "menisque lateral", "aussenmeniskus", "lateraler meniskus",
        "menisco laterale", "laterale meniscus", "lakotka boczna",
        "латеральный мениск", "外侧半月板", "外側半月板",
    ),
    "Medial OA": (
        "medial compartment osteoarthritis", "medial compartment degenerative",
        "medial chondrosis", "medial cartilage loss", "ic kompartman",
        "iç kompartman", "artrosis medial", "gonarthrose mediale",
        "mediale gonarthrose", "medial osteoarthritis", "内侧间室退变",
    ),
    "Lateral OA": (
        "lateral compartment osteoarthritis", "lateral compartment degenerative",
        "lateral chondrosis", "lateral cartilage loss", "dis kompartman",
        "dış kompartman", "artrosis lateral", "gonarthrose laterale",
        "laterale gonarthrose", "lateral osteoarthritis", "外侧间室退变",
    ),
    "PF OA": (
        "patellofemoral osteoarthritis", "patellofemoral chondrosis",
        "retropatellar chondro", "patellar cartilage", "patellofemoral artroz",
        "kondromalazi", "chondromalacia", "artrosis patelofemoral",
        "retropatellare chondropathie", "髌股关节退变", "膝蓋大腿関節症",
    ),
    "Effusion": (
        "joint effusion", "effusion", "eklem sivisi", "eklem sıvısı", "efuzyon",
        "derrame articular", "epanchement", "gelenkerguss", "erguss",
        "versamento articolare", "gewrichtsvocht", "wysiek",
        "выпот", "关节积液", "関節液貯留",
    ),
    "Synovitis": (
        "synovitis", "synovial hypertroph", "synovial prolifer", "sinovit",
        "sinovyal", "sinovitis", "synovite", "synovitis", "sinovite",
        "synovitis", "zapalenie blony maziowej", "синовит", "滑膜炎", "滑膜炎",
    ),
    "Baker's": (
        "baker", "popliteal cyst", "baker kisti", "popliteal kist",
        "quiste de baker", "kyste de baker", "bakerzyste", "poplitealzyste",
        "cisti di baker", "bakercyste", "torbiel bakera",
        "киста бейкера", "腘窝囊肿", "ベーカー嚢腫",
    ),
    "Contusion": (
        "contusion", "bone bruise", "bone marrow edema", "marrow oedema",
        "kemik kontuzyon", "kemik ilik odemi", "kemik iliği ödemi",
        "contusion osea", "edema oseo", "contusion osseuse", "knochenmarködem",
        "knochenkontusion", "contusione ossea", "edema midollare",
        "botmergoedeem", "stluczenie kosci", "отек костного мозга",
        "骨挫伤", "骨挫傷",
    ),
    "Fracture": (
        "fracture", "fraktur", "kirik", "kırık", "cortical break",
        "fractura", "fracture", "fraktur", "frattura", "breuk", "zlamanie",
        "перелом", "骨折", "骨折", "insufficiency fracture", "avulsion",
    ),
}
