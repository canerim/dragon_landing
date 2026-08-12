"""Multilingual report parsing.

The ontology had a construction bug (a ``slots=True`` dataclass assigning an
undeclared attribute) that survived because nothing ever instantiated it. These
tests instantiate it and exercise the three things a naive NegEx port gets
wrong: Turkish casing, post-posed negation, and radiology report structure.
"""

from __future__ import annotations

import pytest

from kairos.constants import TARGETS, Assertion
from kairos.text.ontology import (
    KneeOntology,
    normalise_text,
    split_sentences,
)


@pytest.fixture(scope="module")
def onto():
    return KneeOntology()


def test_ontology_constructs_and_compiles_every_label(onto):
    """Regression: slots=True made this raise AttributeError on construction."""
    assert onto._compiled, "no compiled patterns"
    assert set(onto._compiled) <= set(TARGETS)
    for label in ("ACL", "MCL", "Effusion", "Fracture"):
        assert label in onto._compiled


def test_ontology_accepts_a_custom_lexicon():
    o = KneeOntology(lexicon={"ACL": ("kreuzband",)})
    m = o.extract("Ruptur des Kreuzband.")
    assert [x.label for x in m] == ["ACL"]


# --------------------------------------------------------------------------- #
# Normalisation                                                                #
# --------------------------------------------------------------------------- #


def test_turkish_dotted_i_survives_case_folding():
    """``.lower()`` turns MENİSKÜS into a combining-dot form that never matches."""
    folded = normalise_text("MENİSKÜS YIRTIĞI")
    assert "menisküs" in folded or "meniskus" in folded.replace("ü", "u")
    assert "̇" not in folded, "combining dot above survived normalisation"


def test_normalisation_strips_zero_width_and_collapses_whitespace():
    assert normalise_text("ACL​  \n tear") == "acl tear"


def test_sentence_splitting_handles_report_structure():
    """Reports are bullet lists and semicolon chains, not prose."""
    text = ("FINDINGS:\n- ACL intact; PCL intact\n- Medial meniscus tear\n"
            "• Joint effusion present.")
    sents = split_sentences(text)
    assert len(sents) >= 4
    assert any("medial meniscus tear" in s.lower() for s in sents)
    # The whole findings block must not collapse into one "sentence" -- that
    # would put every negation in scope of every concept.
    assert not any(len(s) > 80 for s in sents)


# --------------------------------------------------------------------------- #
# Assertion detection                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,label,expected",
    [
        ("Full-thickness tear of the anterior cruciate ligament.", "ACL",
         Assertion.POSITIVE),
        ("No evidence of anterior cruciate ligament tear.", "ACL",
         Assertion.NEGATIVE),
        ("Possible anterior cruciate ligament sprain.", "ACL",
         Assertion.UNCERTAIN),
        ("Joint effusion present.", "Effusion", Assertion.POSITIVE),
        ("No joint effusion.", "Effusion", Assertion.NEGATIVE),
    ],
)
def test_english_assertions(onto, text, label, expected):
    weak = onto.to_weak_labels(onto.extract(text))
    assert weak[label][0] == expected, f"{text!r} -> {weak[label]}"


def test_turkish_post_posed_negation(onto):
    """NegEx's left-context rule labels this POSITIVE. It is a negation."""
    weak = onto.to_weak_labels(
        onto.extract("Ön çapraz bağda yırtık izlenmemektedir.")
    )
    assert weak["ACL"][0] == Assertion.NEGATIVE


def test_japanese_post_posed_negation(onto):
    weak = onto.to_weak_labels(onto.extract("前十字靭帯損傷は認めない。"))
    assert weak["ACL"][0] == Assertion.NEGATIVE


def test_turkish_positive_is_not_swallowed(onto):
    weak = onto.to_weak_labels(
        onto.extract("Ön çapraz bağda tam kat yırtık mevcuttur.")
    )
    assert weak["ACL"][0] == Assertion.POSITIVE


@pytest.mark.parametrize(
    "text",
    [
        "Sin rotura del ligamento cruzado anterior.",
        "Pas de rupture du ligament croisé antérieur.",
        "Kein Nachweis einer Ruptur des vorderes Kreuzband.",
    ],
)
def test_negation_across_languages(onto, text):
    weak = onto.to_weak_labels(onto.extract(text))
    assert weak["ACL"][0] == Assertion.NEGATIVE, f"{text!r} -> {weak['ACL']}"


def test_uncertainty_lowers_confidence_below_a_plain_positive(onto):
    certain = onto.to_weak_labels(onto.extract("Tear of the ACL."))
    unsure = onto.to_weak_labels(onto.extract("Possible tear of the ACL."))
    assert unsure["ACL"][0] == Assertion.UNCERTAIN
    assert unsure["ACL"][1] <= certain["ACL"][1]


def test_historicity_downweights_but_does_not_drop(onto):
    """"Old ACL reconstruction" still implies a torn ACL -- weigh it, don't bin it."""
    fresh = onto.to_weak_labels(onto.extract("Tear of the ACL."))
    old = onto.to_weak_labels(onto.extract("Status post old ACL tear."))
    assert old["ACL"][0] == Assertion.POSITIVE
    assert old["ACL"][1] < fresh["ACL"][1]


# --------------------------------------------------------------------------- #
# Compartment resolution                                                       #
# --------------------------------------------------------------------------- #


def test_compartment_prevents_cross_assignment(onto):
    """The bug this exists for: a medial sentence producing a lateral label."""
    weak = onto.to_weak_labels(
        onto.extract("Tear of the posterior horn of the medial meniscus.")
    )
    assert weak["Medial Meniscus"][0] == Assertion.POSITIVE
    assert weak["Lateral Meniscus"][0] == Assertion.NOT_MENTIONED


def test_lateral_sentence_does_not_produce_a_medial_label(onto):
    weak = onto.to_weak_labels(onto.extract("Lateral meniscus root tear."))
    assert weak["Lateral Meniscus"][0] == Assertion.POSITIVE
    assert weak["Medial Meniscus"][0] == Assertion.NOT_MENTIONED


def test_ambiguous_compartment_is_kept_with_reduced_confidence(onto):
    both = onto.extract("Tears of the medial and lateral menisci.")
    labels = {m.label for m in both}
    assert {"Medial Meniscus", "Lateral Meniscus"} & labels


@pytest.mark.parametrize(
    "text,expected",
    [
        ("medial meniscus tear", "medial"),
        ("lateral compartment chondrosis", "lateral"),
        ("retropatellar chondromalacia", "patellofemoral"),
        ("iç menisküs yırtığı", "medial"),
        ("joint effusion", None),
    ],
)
def test_compartment_resolver(onto, text, expected):
    assert onto.compartment_of(text) == expected


# --------------------------------------------------------------------------- #
# Aggregation policy                                                           #
# --------------------------------------------------------------------------- #


def test_positive_outvotes_the_templated_negative_checklist(onto):
    """The checklist is high-volume and low-information; one explicit positive
    must win, or the parser lands at 0.99 specificity / 0.4 sensitivity."""
    text = ("ACL intact. PCL intact. MCL intact. LCL intact. "
            "Re-evaluation: full-thickness ACL tear is present.")
    weak = onto.to_weak_labels(onto.extract(text))
    assert weak["ACL"][0] == Assertion.POSITIVE


def test_unmentioned_labels_are_explicitly_not_mentioned(onto):
    weak = onto.to_weak_labels(onto.extract("Joint effusion."))
    assert set(weak) == set(TARGETS)
    assert weak["Fracture"][0] == Assertion.NOT_MENTIONED
    assert weak["Fracture"][1] == 0.0


def test_empty_and_garbage_input_is_safe(onto):
    for text in ("", "   ", "...", "\n\n", "12345"):
        weak = onto.to_weak_labels(onto.extract(text))
        assert set(weak) == set(TARGETS)
        assert all(v[0] == Assertion.NOT_MENTIONED for v in weak.values())


def test_confidence_is_bounded(onto):
    text = "Full-thickness tear of the medial meniscus with joint effusion."
    for m in onto.extract(text):
        assert 0.0 <= m.confidence <= 1.0


def test_language_identification_buckets_the_major_scripts():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "parse_reports",
        Path(__file__).resolve().parent.parent / "scripts" / "02_parse_reports.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.detect_language("前十字靭帯の完全断裂を認める") == "ja"
    assert mod.detect_language("前交叉韧带完全断裂") == "zh"
    assert mod.detect_language("разрыв передней крестообразной связки") == "ru"
    assert mod.detect_language("Ön çapraz bağda yırtık izlenmektedir ve bulgu") == "tr"
    assert mod.detect_language("") == "unknown"


# --------------------------------------------------------------------------- #
# OA vocabulary, grown from the RSNA-2026 corpus audit.                        #
#                                                                              #
# Every sentence below is a real (deaccented / truncated) line taken from the  #
# frequency-ranked unmatched-sentence dump of the 4407 training reports.  The  #
# shipped lexicon scored 0.000 sensitivity on all three OA targets against the #
# 58 gold-labelled studies, with 99.6 / 100 / 94.1 % of reports registering    #
# "not mentioned" -- these pin the fix so it cannot silently regress.          #
# --------------------------------------------------------------------------- #

OA_CASES = [
    ("oa patelofemoral.", "PF OA", Assertion.POSITIVE),
    ("condropatia rotuliana.", "PF OA", Assertion.POSITIVE),
    ("geen majeure kraakbeendefecten patellofemoraal.", "PF OA", Assertion.NEGATIVE),
    ("geen majeure kraakbeendefecten mediaal en lateraal femorotibiaal.",
     "Medial OA", Assertion.NEGATIVE),
    ("keine hohergradige chondropathie femoropatellar und femorotibial.",
     "PF OA", Assertion.NEGATIVE),
    ("hondromalacija ii. stupnja hrskavice medijalne fasete patele.",
     "PF OA", Assertion.POSITIVE),
    ("Full thickness cartilage loss along the medial femoral condyle "
     "and medial tibial plateau.", "Medial OA", Assertion.POSITIVE),
    ("diz eklemindeki trikompartmantal dejeneratif eklem hastaligi",
     "Medial OA", Assertion.POSITIVE),
    ("petits osteophytes marginaux.", "Medial OA", Assertion.POSITIVE),
]


@pytest.mark.parametrize("text,label,want", OA_CASES)
def test_oa_vocabulary_from_corpus_audit(onto, text, label, want):
    assert onto.to_weak_labels(onto.extract(text))[label][0] is want


def test_tricompartmental_asserts_all_three_oa_labels(onto):
    """One word, three targets.

    Without this the sentence resolves to compartment ``None``, every OA label
    lands at 0.425 confidence, and the weak-label loss -- which gates on
    confidence -- discards all three.
    """
    for text in ("tricompartmental marginal osteophytes",
                 "oa of all three compartments."):
        weak = onto.to_weak_labels(onto.extract(text))
        for label in ("Medial OA", "Lateral OA", "PF OA"):
            assert weak[label][0] is Assertion.POSITIVE, (text, label)
            assert weak[label][1] > 0.6, (text, label)


def test_patellofemoral_anatomy_beats_a_medial_lateral_modifier(onto):
    """"Lateral trochlea" is patellofemoral OA, not lateral-compartment OA.

    The modifier names a facet *within* the patellofemoral joint.  Routing it
    to Lateral OA is not a near miss: it is a wrong label on one target and a
    missing one on another.
    """
    text = ("Full thickness cartilage defect, 1.2x1.5cm. at lateral trochlea "
            "with subchondral bone edema.")
    weak = onto.to_weak_labels(onto.extract(text))
    assert weak["PF OA"][0] is Assertion.POSITIVE
    assert weak["PF OA"][1] > 0.6
    assert weak["Lateral OA"][0] is Assertion.NOT_MENTIONED

    spanish = ("Condropatia focal grado 4 del aspecto inferior de la vertiente "
               "medial de la troclea femoral.")
    weak = onto.to_weak_labels(onto.extract(spanish))
    assert weak["PF OA"][0] is Assertion.POSITIVE
    assert weak["Medial OA"][0] is Assertion.NOT_MENTIONED


def test_oa_vocabulary_does_not_leak_into_the_meniscus_labels(onto):
    """"Degenerative" qualifies menisci too; it must not become an OA positive
    that outranks the meniscal assertion in the same sentence."""
    text = "degenerative signal throughout the lateral meniscus without surfacing tear."
    weak = onto.to_weak_labels(onto.extract(text))
    assert weak["Lateral Meniscus"][0] is Assertion.NEGATIVE


def test_measurement_and_ordinal_periods_do_not_split_a_sentence():
    """A finding must stay attached to the compartment that qualifies it.

    ``cartilage defect, 1.2x1.5cm. | at lateral trochlea`` orphans the finding
    from "trochlea", and the orphan resolves to compartment ``None`` -- halving
    its confidence and dropping it below every downstream gate.
    """
    from kairos.text.ontology import normalise_text, split_sentences

    assert len(split_sentences(normalise_text(
        "Cartilage defect, 1.2x1.5cm. at lateral trochlea."))) == 1
    assert len(split_sentences(normalise_text(
        "Hondromalacija ii. stupnja hrskavice."))) == 1
    # ... while genuine sentence boundaries still split.
    assert len(split_sentences(normalise_text(
        "No fracture is seen. ACL is intact."))) == 2


def test_short_negation_cues_do_not_fire_inside_other_languages(onto):
    """``_find_cue`` scans every language's cue list against every sentence.

    A two-letter cue therefore matches inside unrelated words: Croatian ``ne ``
    fires on the English "bo|ne edema|" and flips a positive OA finding to a
    negation.  Cues must be long enough to survive that.
    """
    positives = [
        ("Osteophytes at the medial femoral condyle with subchondral bone edema.",
         "Medial OA"),
        ("Full thickness cartilage defect at lateral trochlea with bone edema.",
         "PF OA"),
        ("Bone marrow edema of the lateral tibial plateau.", "Contusion"),
    ]
    for text, label in positives:
        got = onto.to_weak_labels(onto.extract(text))[label][0]
        assert got is Assertion.POSITIVE, (text, label, got)


def test_language_identification_covers_the_corpus_languages():
    """Greek was 7.5 % of the corpus and fell entirely into "unknown"; Bulgarian
    was labelled Russian; Croatian was labelled Polish via Polish's ``bez``."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "parse_reports",
        Path(__file__).resolve().parent.parent / "scripts" / "02_parse_reports.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.detect_language("χωρίς ενδαρθρική συλλογή υγρού.") == "el"
    assert mod.detect_language("нормално изобразяване на латералния менискус.") == "bg"
    assert mod.detect_language("разрыв передней крестообразной связки, не выявлено") == "ru"
    assert mod.detect_language(
        "medijalni menisk bez znakova rupture i degeneracije, uredan prikaz") == "hr"


ROUND2_CASES = [
    # Spanish names the compartments interno/externo, not medial/lateral.
    ("resultados: rotura de menisco interno.", "Medial Meniscus", Assertion.POSITIVE),
    ("rotura de menisco interno.", "Lateral Meniscus", Assertion.NOT_MENTIONED),
    ("el menisque externe est sans particularite.", "Lateral Meniscus", Assertion.NEGATIVE),
    ("lateralni menisk bez znakova degeneracije ili rupture.",
     "Lateral Meniscus", Assertion.NEGATIVE),
    ("нормално изобразяване на латералния менискус.",
     "Lateral Meniscus", Assertion.NEGATIVE),
    # Effusion was missing in five of the corpus languages.
    ("малък ставен излив.", "Effusion", Assertion.POSITIVE),
    ("няма данни за ставен излив.", "Effusion", Assertion.NEGATIVE),
    ("bez signifikantnog izljeva u zglob.", "Effusion", Assertion.NEGATIVE),
    ("matige hydrops.", "Effusion", Assertion.POSITIVE),
    ("akzentuierte gelenkflussigkeit.", "Effusion", Assertion.POSITIVE),
    ("diz eklemi ici sivi miktari hafif derecede artmis.", "Effusion", Assertion.POSITIVE),
    ("diz eklemi ici sivi miktari normal.", "Effusion", Assertion.NEGATIVE),
    ("bez znakova poplitealne ciste.", "Baker's", Assertion.NEGATIVE),
    ("бекерова киста.", "Baker's", Assertion.POSITIVE),
    ("no hay quistes popliteos patologicos.", "Baker's", Assertion.NEGATIVE),
    ("geen botoedeem.", "Contusion", Assertion.NEGATIVE),
    ("kein subchondrales knochenodem.", "Contusion", Assertion.NEGATIVE),
    ("manji izljev u zglobnim prostorima, uz blazu proliferaciju sinovije.",
     "Synovitis", Assertion.POSITIVE),
]


@pytest.mark.parametrize("text,label,want", ROUND2_CASES)
def test_second_round_corpus_vocabulary(onto, text, label, want):
    assert onto.to_weak_labels(onto.extract(text))[label][0] is want


def test_pseudo_negation_does_not_negate(onto):
    """"quiste poplíteo **no** complicado" is an *uncomplicated* cyst -- the
    negation word qualifies "complicado", not the cyst's existence."""
    weak = onto.to_weak_labels(onto.extract("pequeno quiste popliteo no complicado."))
    assert weak["Baker's"][0] is Assertion.POSITIVE


def test_cue_needs_a_leading_word_boundary(onto):
    """A bare cue must not fire inside an unrelated word.

    Spanish ``pequeño `` ends in ``no ``, so a plain substring test reported
    every "pequeño quiste poplíteo" as an absent cyst.  The cues carry a
    trailing space to guard their right edge; nothing guarded the left.
    """
    assert onto.to_weak_labels(
        onto.extract("pequeno quiste popliteo."))["Baker's"][0] is Assertion.POSITIVE
    # ... and a real leading negation still fires.
    assert onto.to_weak_labels(
        onto.extract("no popliteal cyst."))["Baker's"][0] is Assertion.NEGATIVE


def test_word_boundary_rule_does_not_break_cjk_negation(onto):
    """``\\b`` is meaningless between ideographs -- every CJK character is a word
    character, so a boundary-anchored cue preceded by another ideograph would
    never match and Chinese/Japanese negation would silently stop working."""
    assert onto.to_weak_labels(
        onto.extract("前十字靭帯の完全断裂を認めない"))["ACL"] [0] is Assertion.NEGATIVE
    assert onto.to_weak_labels(
        onto.extract("未见前交叉韧带断裂"))["ACL"][0] is Assertion.NEGATIVE
