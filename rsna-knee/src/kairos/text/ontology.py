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

#: A trailing "sentence-final" period that is not one: a measurement
#: (``1.2x1.5cm.``), a roman ordinal (``hondromalacija ii.``) or a list number
#: (``impression: 1.``).  Splitting there severs a finding from the compartment
#: that qualifies it -- "cartilage defect, 1.2x1.5cm." / "at lateral trochlea."
#: -- and the orphaned half resolves to compartment ``None``, which multiplies
#: confidence by 0.5 and drops the match below every downstream gate.
#:
#: Case is *not* usable as a signal here: ``normalise_text`` casefolds before
#: this runs, so "the next fragment starts lowercase" is true of every fragment.
_CONTINUES = re.compile(
    r"(?:^|\s)(?:[0-9][^\s]*|i|ii|iii|iv|v|vi|vii|viii|ix|x|cm|mm|ml|nr|no|fig)\.$",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation tuned for radiology reports.

    Reports are not prose.  They are bullet lists, numbered findings and
    semicolon-chained clauses, and an off-the-shelf sentence splitter merges a
    whole findings section into one "sentence" -- which destroys negation scope
    and therefore every weak label derived from it.  We split on terminal
    punctuation, newlines *and* bullet markers, then rejoin the fragments whose
    "terminator" was a measurement or an ordinal, and drop what is left shorter
    than three characters.
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and _CONTINUES.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return [p for p in merged if len(p) >= 3]


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
    # Bulgarian, Croatian/Serbian and Greek were absent entirely.  Because
    # ``_find_cue`` scans every language's cue list regardless of the detected
    # language, an absent list does not merely weaken those reports -- it means
    # a negated finding in them is read as POSITIVE.  Greek was 7.5 % of the
    # corpus and every "χωρίς ενδαρθρική συλλογή υγρού" was a false Effusion.
    "bg": ("няма ", "не ", "без ", "б.о.", "без особености", "нормално",
           "запазен", "интакт", "не се "),
    # NB: no bare ``ne ``.  ``_find_cue`` tests every language's cues against
    # every sentence regardless of the detected language, so a two-letter cue
    # fires inside unrelated words in other languages -- ``ne `` matches the
    # English "bo|ne edema|", turning a positive OA finding into a negation.
    "hr": ("bez ", "nema ", "nije ", "nisu ", "ne nalazi", "uredn",
           "primjeren", "održan", "intaktn", "bez znakova", "bez osobitosti"),
    "el": ("χωρίς", "δεν ", "ουδεμία", "ακέραι", "φυσιολογικ",
           "εντός του φυσιολογικού", "χωρίς ευρήματα"),
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
    "bg": ("вероятно", "възможно", "съмнение", "не може да се изключи"),
    "hr": ("vjerojatno", "moguć", "sumnja", "ne može se isključiti",
           "suspektn", "vjerojatna"),
    "el": ("πιθαν", "ύποπτ", "δεν αποκλείεται", "συμβατ"),
    "zh": ("可能", "考虑", "疑似", "不除外"),
    "ja": ("疑い", "可能性", "示唆", "否定できない"),
}

#: Phrases that *contain* a negation word but negate nothing -- NegEx calls
#: these pseudo-negations.  They are removed from the sentence before cue
#: search, because otherwise "pequeño quiste poplíteo **no** complicado"
#: ("uncomplicated popliteal cyst") reports the cyst as absent, which is the
#: opposite of what it says.  Ordered longest-first at use so that a longer
#: phrase is stripped before a shorter one nested inside it.
PSEUDO_NEGATION_CUES: tuple[str, ...] = (
    "no complicado", "no complicada", "non complique", "not complicated",
    "not previously discussed", "no relevant prior", "no significant change",
    "no aggressive", "sin cambios significativos", "sin particularidad",
    "geen relevante", "keine relevante", "bez znacajne promjene",
)

HISTORICITY_CUES: tuple[str, ...] = (
    "post-operative", "postoperative", "status post", "prior", "previous", "old",
    "chronic", "sequela", "postop", "ameliyat", "geçirilmiş", "eski", "kronik",
    "postoperatorio", "antiguo", "crónico", "postopératoire", "ancien", "chronique",
    "postoperativ", "alt", "chronisch", "术后", "陈旧", "術後", "陳旧",
)

#: Words that flip the compartment.  Kept separate from the concept lexicon so
#: that a single medial/lateral list serves all twelve labels.
_MEDIAL = ("medial", "mediale", "medialen", "medyal", "iç ", "interno", "interne",
           "przyśrodkow", "медиальн", "内侧", "内側",
           # Added from the RSNA-2026 corpus audit.  Dutch ``mediaal`` and
           # Croatian ``medijaln`` do not contain the substring ``medial``, so
           # the original list silently returned ``None`` for every Dutch and
           # Croatian sentence -- which multiplied confidence by 0.5 and put
           # every compartment label under the 0.6 gate.
           "mediaal", "medijaln", "медиал", "έσω", "εσω")
_LATERAL = ("lateral", "laterale", "lateralen", "dış ", "externo", "externe",
            "boczn", "латеральн", "外侧", "外側",
            "lateraal", "латерал", "έξω", "εξω")
#: Patellofemoral markers.  ``patellofemora`` (no trailing ``l``) is deliberate:
#: it covers ``patellofemoral``/``patellofemorale``/``patellofemoraal`` in one
#: form, which the fully-spelled variant does not.
_PATELLOFEMORAL = ("patellofemora", "patelofemora", "femoropatella", "patellar",
                   "retropatellar", "patella", "rotulian", "rotulien",
                   "diz kapağı", "髌股", "膝蓋大腿")

#: Anatomy that is decisive for the *patellofemoral* compartment even when a
#: medial/lateral modifier is present: "lateral trochlea" and "medial patellar
#: facet" are patellofemoral OA, not tibiofemoral.  Used only for the three OA
#: labels -- the meniscus and MCL labels keep the plain medial/lateral rule,
#: where a patellar mention in the same sentence carries no such implication.
_PF_DECISIVE = ("trochlea", "trocle", "troklea", "трохле", "patellar facet",
                "faseta patele", "fasete patele", "faceta rotuliana",
                "patellofemora", "patelofemora", "femoropatella",
                "retropatellar", "rotulian", "patele", "patelarn")

#: One word that asserts all three OA compartments at once.  Without this the
#: sentence resolves to ``None`` and all three labels land at 0.425 confidence,
#: i.e. invisible to any downstream gate.
_TRICOMPARTMENTAL = ("tricompartmental", "tricompartimental", "tri-compartmental",
                     "tricompartimentale", "trikompartman", "trikompartmantal",
                     "trikompartmentell", "all three compartments",
                     "three compartments", "tres compartimentos",
                     "trois compartiments", "sva tri kompartmana")


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
    def _cue_present(cue: str, haystack: str) -> bool:
        """Substring test with a leading word boundary where that is meaningful.

        A bare substring test lets a short cue fire inside an unrelated word,
        across languages: Spanish ``pequeño quiste poplíteo`` contains ``no ``
        and so reported the cyst as absent.  The cues carry a trailing space to
        guard their right edge; nothing guarded the left.

        The boundary is applied only for scripts that separate words with
        spaces.  ``\\b`` is meaningless for Chinese and Japanese -- every
        ideograph is a word character, so a cue preceded by another ideograph
        would never match and both languages' negation would silently stop
        working.
        """
        cd = _deaccent(cue)
        if not cd:
            return False
        if re.match(r"[぀-ヿ一-鿿]", cd):
            return cd in haystack
        return re.search(r"\b" + re.escape(cd), haystack) is not None

    @staticmethod
    def _find_cue(sentence: str, cues: dict[str, tuple[str, ...]]) -> tuple[str | None, str | None]:
        s = _deaccent(sentence)
        # Blank out pseudo-negations first, longest match wins, so that the
        # negation word inside them cannot be found by the scan below.
        for p in sorted(PSEUDO_NEGATION_CUES, key=len, reverse=True):
            pd_ = _deaccent(p)
            if pd_ in s:
                s = s.replace(pd_, " " * len(pd_))
        for lang, forms in cues.items():
            for c in forms:
                if KneeOntology._cue_present(c, s):
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
        if any(_deaccent(k) in s for k in _TRICOMPARTMENTAL):
            return "all"
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

    @staticmethod
    def compartment_for_oa(sentence: str) -> str | None:
        """Compartment resolution for the three OA labels.

        Differs from :meth:`compartment_of` in one respect: decisive
        patellofemoral anatomy wins over a medial/lateral modifier, because in
        "full thickness cartilage defect at the lateral trochlea" or "high-grade
        cartilage loss along the medial patellar facet" the modifier names a
        *facet within the patellofemoral joint*, not a tibiofemoral compartment.
        Routing those to Lateral/Medial OA is not a near miss -- it is a wrong
        label on one target and a missing one on another.

        The meniscus and MCL labels deliberately keep the plain rule: a patellar
        mention in the same sentence as a meniscal tear implies nothing about
        which meniscus is torn.
        """
        s = _deaccent(sentence.casefold())
        if any(_deaccent(k) in s for k in _TRICOMPARTMENTAL):
            return "all"
        if any(_deaccent(k) in s for k in _PF_DECISIVE):
            return "patellofemoral"
        return KneeOntology.compartment_of(sentence)

    # -- extraction ------------------------------------------------------ #

    def extract(self, report: str, *, language: str | None = None) -> list[ConceptMatch]:
        out: list[ConceptMatch] = []
        for sent in split_sentences(normalise_text(report)):
            de = _deaccent(sent)
            historic = any(_deaccent(h) in de for h in HISTORICITY_CUES)
            comp = self.compartment_of(sent)
            comp_oa = self.compartment_for_oa(sent)
            for label, rx in self._compiled.items():
                for m in rx.finditer(de):
                    a, conf, cue, cue_lang = self.assert_state(sent, m.span())
                    if self.require_compartment and label in _COMPARTMENT_LABELS:
                        want = _COMPARTMENT_LABELS[label]
                        # A local name: rebinding ``comp`` here would leak the
                        # OA-specific resolution to every label examined later
                        # in the same sentence.
                        c = comp_oa if label in _OA_LABELS else comp
                        if c == "all":
                            # "Tricompartmental OA" asserts all three at once.
                            pass
                        elif c is None:
                            conf *= 0.5
                        elif c == "both":
                            conf *= 0.8
                        elif c != want:
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
                            compartment_hint=comp_oa if label in _OA_LABELS else comp,
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


_OA_LABELS = ("Medial OA", "Lateral OA", "PF OA")

#: The bare word "meniscus" in each language, shared by both meniscus labels
#: and routed by the compartment resolver -- exactly as :data:`_OA_GENERIC` is.
#:
#: Spanish names the compartments ``interno``/``externo``, not
#: ``medial``/``lateral``, so ``rotura de menisco interno`` (165 occurrences in
#: the corpus, the single most frequent finding sentence in that language)
#: matched nothing: the lexicon held ``menisco medial``.  ``interno`` was
#: already a medial marker, so supplying the bare noun is all that was needed.
#:
#: Substring matching does the morphology: ``menisc`` covers meniscus/menisci/
#: menisco/meniscal, ``menisk`` covers menisk/meniskus/menisküs.
_MENISCUS_GENERIC: tuple[str, ...] = (
    "menisc", "menisk", "menisque", "μηνισκ", "мениск", "半月板", "半月",
)

#: Vocabulary that means "degenerative disease of *a* compartment" without
#: naming which.  It is shared by all three OA labels and routed by
#: :meth:`KneeOntology.compartment_for_oa`, which is what makes one list serve
#: three targets: "cartilage loss along the medial femoral condyle" reaches
#: Medial OA and is skipped for Lateral OA by the ``c != want`` branch.
#:
#: Assembled from the frequency-ranked unmatched sentences of the RSNA-2026
#: corpus, per the workflow this module's docstring prescribes.  The original
#: list held only formal phrasings ("medial compartment osteoarthritis") and
#: scored 0.000 sensitivity on all three OA targets, with 99.6 / 100 / 94.1 %
#: of reports registering "not mentioned".
_OA_GENERIC: tuple[str, ...] = (
    # -- osteoarthritis / arthrosis ------------------------------------- #
    "osteoarthritis", "osteoarthrosis", "osteoarthritic", "arthrosis",
    "arthrose", "gonarthrose", "gonartroz", "artrosis", "artroz", "artrose",
    "osteoartrit", "artritick", "artriticke", "osteoartriticke", "artrotic",
    "arthritis of the", "остеоартр", "οστεοαρθρ",
    # Bare "OA" is how this corpus most often writes it ("oa of all three
    # compartments", "oa patelofemoral").  Anchored to a following compartment
    # or preposition rather than listed bare: an unanchored "oa" matches inside
    # any word ending in those two letters.
    "oa of", "oa patel", "oa medial", "oa lateral", "oa femorotibial",
    "oa de ", "oa del ", "oa dell",
    # -- osteophytes ----------------------------------------------------- #
    "osteophyt", "osteofit", "osteofito", "osteofiet", "osteofyt",
    "ostephyt", "остеофит", "οστεοφ",
    # -- chondropathy / chondromalacia / chondrosis ---------------------- #
    "chondropath", "chondromalac", "chondrosis", "chondrose",
    "condropat", "condromalac", "kondropat", "kondromalaz",
    "hondropat", "hondromalacij", "chondropathie", "chondromalazi",
    "χονδρομαλ", "хондромалаци",
    # -- cartilage loss / defect ----------------------------------------- #
    "cartilage loss", "cartilage defect", "cartilage thinning",
    "cartilage heterogeneity", "chondral defect", "chondral loss",
    "chondral ulcer", "cartilage denudation", "osteochondral defect",
    "knorpeldefekt", "knorpelschaden", "knorpelverlust",
    "kraakbeendefect", "kraakbeenverlies",
    "perte de cartilage", "perte cartilagineuse", "ulcere chondral",
    "perdida de cartilago", "ulcera condral", "defecto condral",
    "kikirdak kaybi", "kikirdak defekt", "kikirdak incelme",
    "denudacija hrskavice", "erozije zglobnih hrskavica",
    # -- degenerative joint disease -------------------------------------- #
    "degenerative joint disease", "degenerative change",
    "dejeneratif eklem", "degenerative veranderung", "degenerative veraenderung",
    "cambios degenerativos", "changements degeneratifs",
    "degeneratieve verandering", "degenerativne promjene",
    "osteoartriticke promjene", "дегенеративн", "εκφυλιστικ",
    # -- joint-space narrowing / subchondral reaction -------------------- #
    "joint space narrowing", "subchondral sclerosis", "subchondral cyst",
    "subchondral cystic", "subkondral skleroz", "subchondrale sklerose",
    "gelenkspaltverschmalerung", "pinzamiento articular",
)


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
    "Medial Meniscus": _MENISCUS_GENERIC + (
        "medial meniscus", "medial menisc", "ic menisk", "iç menisküs",
        "menisco medial", "menisco interno", "menisque medial", "menisque interne",
        "innenmeniskus", "medialer meniskus",
        "menisco mediale", "mediale meniscus", "lakotka przysrodkowa",
        "медиальный мениск", "内侧半月板", "内側半月板",
    ),
    "Lateral Meniscus": _MENISCUS_GENERIC + (
        "lateral meniscus", "lateral menisc", "dis menisk", "dış menisküs",
        "menisco lateral", "menisco externo", "menisque lateral", "menisque externe",
        "aussenmeniskus", "lateraler meniskus",
        "menisco laterale", "laterale meniscus", "lakotka boczna",
        "латеральный мениск", "外侧半月板", "外側半月板",
    ),
    "Medial OA": _OA_GENERIC + (
        "medial compartment osteoarthritis", "medial compartment degenerative",
        "medial chondrosis", "medial cartilage loss", "ic kompartman",
        "iç kompartman", "artrosis medial", "gonarthrose mediale",
        "mediale gonarthrose", "medial osteoarthritis", "内侧间室退变",
    ),
    "Lateral OA": _OA_GENERIC + (
        "lateral compartment osteoarthritis", "lateral compartment degenerative",
        "lateral chondrosis", "lateral cartilage loss", "dis kompartman",
        "dış kompartman", "artrosis lateral", "gonarthrose laterale",
        "laterale gonarthrose", "lateral osteoarthritis", "外侧间室退变",
    ),
    "PF OA": _OA_GENERIC + (
        "patellofemoral osteoarthritis", "patellofemoral chondrosis",
        "retropatellar chondro", "patellar cartilage", "patellofemoral artroz",
        "kondromalazi", "chondromalacia", "artrosis patelofemoral",
        "condropatia rotuliana", "condropatia femoropatelar",
        "retropatellare chondropathie", "髌股关节退变", "膝蓋大腿関節症",
    ),
    "Effusion": (
        "joint effusion", "effusion", "eklem sivisi", "eklem sıvısı", "efuzyon",
        "derrame articular", "epanchement", "gelenkerguss", "erguss",
        "versamento articolare", "gewrichtsvocht", "wysiek",
        "выпот", "关节积液", "関節液貯留",
        # Absent in five of the corpus languages, which is the likeliest
        # explanation for this label's 0.478 specificity: the parser saw the
        # English positives and none of the other languages' negations.
        "ставен излив", "излив",                       # bg
        "izljev", "hidrops",                           # hr
        "συλλογη υγρου", "ενδαρθρικη συλλογη",         # el
        "hydrops", "suprapatellaire recessus",         # nl
        "gelenkflussigkeit", "kniegelenkerguss",       # de
        "ici sivi", "sivi miktari", "sivi artis", "sıvı artışı",  # tr
    ),
    "Synovitis": (
        "synovitis", "synovial hypertroph", "synovial prolifer", "sinovit",
        "sinovyal", "sinovitis", "synovite", "sinovite",
        "zapalenie blony maziowej", "синовит", "滑膜炎",
        "synovial thicken", "thickened synovial", "thickend synovial",
        "synovial tissue", "synovialitis",
        "proliferacij", "sinovij",                                     # hr
        "синовиал", "хипертрофия на синовията",                       # bg
        "υμενιτ", "αρθρικου υμενα",                                   # el
    ),
    "Baker's": (
        "baker", "popliteal cyst", "baker kisti", "popliteal kist",
        "quiste de baker", "kyste de baker", "bakerzyste", "poplitealzyste",
        "cisti di baker", "bakercyste", "torbiel bakera",
        "киста бейкера", "腘窝囊肿", "ベーカー嚢腫",
        # Near misses against forms already present -- different word order or
        # inflection is enough to defeat substring matching.
        "poplitealn", "quiste poplite", "quistes poplite", "kyste poplite",
        "бекеров", "поплитеална киста", "κυστη poplitea",
    ),
    "Contusion": (
        "contusion", "bone bruise", "bone marrow edema", "marrow oedema",
        "kemik kontuzyon", "kemik ilik odemi", "kemik iliği ödemi",
        "contusion osea", "edema oseo", "contusion osseuse", "knochenmarködem",
        "knochenkontusion", "contusione ossea", "edema midollare",
        "botmergoedeem", "stluczenie kosci", "отек костного мозга",
        "骨挫伤", "骨挫傷",
        "knochenodem", "knochenmarksodem", "botoedeem",   # de / nl
        "kostani edem", "edem kostane srzi",              # hr
        "костномозъчен едем", "костен едем",              # bg
        "οιδημα μυελου", "οστικο οιδημα",                 # el
        "oedeme osseux", "oedeme de la moelle",           # fr
        "subchondral bone edema", "subchondral marrow edema",
    ),
    "Fracture": (
        "fracture", "fraktur", "kirik", "kırık", "cortical break",
        "fractura", "fracture", "fraktur", "frattura", "breuk", "zlamanie",
        "перелом", "骨折", "骨折", "insufficiency fracture", "avulsion",
    ),
}
