import pytest

from app.safety import DIAGNOSIS_DOSE, EMERGENCY, NORMAL, OUT_OF_SCOPE, classify, redact


@pytest.mark.parametrize("raw,expected", [
    ("email me at jane.doe@example.com", "email me at [EMAIL]"),
    ("call 555-123-4567 or (555) 123 4567", "call [PHONE] or [PHONE]"),
    ("my SSN is 123-45-6789", "my SSN is [SSN]"),
    ("born 04/12/1980", "born [DATE]"),
    ("MRN: A1234567 has diabetes", "[MRN] has diabetes"),
    ("I live at 42 Elm Street", "I live at [ADDRESS]"),
    ("My name is John Smith and I have asthma", "my name is [NAME] and I have asthma"),
])
def test_redacts_phi(raw, expected):
    assert redact(raw) == expected


def test_redaction_keeps_medical_text():
    q = "What is the treatment for type 2 diabetes?"
    assert redact(q) == q


@pytest.mark.parametrize("q,route", [
    ("I want to kill myself", EMERGENCY),
    ("my dad has chest pain and is sweating", EMERGENCY),
    ("What are the symptoms of a heart attack?", NORMAL),  # informational, not acute
    ("What is lead poisoning?", NORMAL),
    ("how many mg of ibuprofen should I take", DIAGNOSIS_DOSE),
    ("do I have diabetes", DIAGNOSIS_DOSE),
    ("what's the weather tomorrow", OUT_OF_SCOPE),
    ("What causes glaucoma?", NORMAL),
])
def test_router(q, route):
    assert classify(q) == route


def test_emergency_beats_everything():
    assert classify("I took a whole bottle of pills, what dose is dangerous?") == EMERGENCY
