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
