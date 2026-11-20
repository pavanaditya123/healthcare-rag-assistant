"""Output guardrails: checks on what the LLM wrote, before the user sees it.

The input side (safety.py) decides whether the LLM may answer at all. This
side checks the answer itself, because a harmless question can still produce
a harmful answer (NIH pages sometimes mention doses; the LLM can drift).

Order, all cheap and deterministic (~1 ms):
  1. citations_valid()   format: every [n] real, every paragraph cited   (answer.py)
  2. unsafe_output()     no personal dosing instructions or diagnoses
  3. ungrounded()        each cited sentence's content words appear in its cited passages
  4. add_disclaimer()    fixed footer, added in code, not left to the prompt

Any failure -> "degraded": the user gets the NIH passages verbatim instead.
"""
import re

CITATION = re.compile(r"\[(\d+)\]")

# ---------------------------------------------------------------- 2. unsafe content
# A generated answer must not tell the reader what dose to take or what they have.
UNSAFE_OUTPUT = [
    ("dose", re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|milligrams?|micrograms?|ml|mL|cc|units?|IU|tablets?|"
                        r"pills?|capsules?|puffs?|drops?)\b(?:[^.\n]{0,40}\b(?:every|per|a|each|times)\b)?",
                        re.IGNORECASE)),
    ("dosing_instruction", re.compile(r"\b(?:take|use|give|inject|double|increase|decrease) (?:\w+ ){0,3}"
                                      r"(?:every \d+|twice|three times|once) (?:a |per )?(?:day|daily|hours?)\b",
                                      re.IGNORECASE)),
    ("medication_advice", re.compile(r"\byou should (?:take|start taking|stop taking|switch to|increase|decrease|skip)\b",
                                     re.IGNORECASE)),
    ("diagnosis", re.compile(r"\byou (?:probably|likely|most likely|definitely|clearly|almost certainly) have\b|"
                             r"\b(?:your|these) symptoms (?:mean|indicate|suggest|show) (?:that )?you have\b",
                             re.IGNORECASE)),
]


def unsafe_output(text: str) -> str | None:
    """Name of the first unsafe pattern found in the answer, else None."""
    for name, pattern in UNSAFE_OUTPUT:
        if pattern.search(text):
            return name
    return None


# ---------------------------------------------------------------- 3. grounding
STOPWORDS = set("""a about above after again also an and any are as at be because been before being both but by can
could did do does doing during each few for from further had has have having he her here hers him his how i if in into
is it its itself just may might more most much must my no nor not now of off often on once only or other our out over
own same should so some such than that the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your people person called known include
including includes usually many may can also one two""".split())
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")  # a trailing [n] stays with its sentence


def content_stems(text: str) -> set[str]:
    """Lower-case words of 4+ letters, minus stopwords, cut to 5 letters so
    treat / treated / treatment all count as the same word."""
    words = re.findall(r"[a-z][a-z-]{3,}", text.lower())
    return {w[:5] for w in words if w not in STOPWORDS}


def sentence_supports(text: str, passages: list[str]) -> list[tuple[str, float]]:
    """(sentence, support) for every judgeable sentence of the answer.

    A paragraph's citations apply to every sentence in it (answers cite at the
    end of a paragraph). Support = share of a sentence's content stems found in
    the cited passages. This is a cheap lexical check, run on every request; the
    LLM-as-judge eval measures faithfulness properly, offline.
    """
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        cited = {int(n) for n in CITATION.findall(para) if 1 <= int(n) <= len(passages)}
        if not cited:
            continue  # uncited paragraphs are handled by citations_valid (disclaimer only)
        source_stems = set().union(*(content_stems(passages[n - 1]) for n in cited))
        for sentence in _SENTENCE.split(para):
            stems = content_stems(CITATION.sub("", sentence))
            if len(stems) >= 3:  # shorter ones ("See [1].") are too short to judge
                out.append((sentence.strip(), len(stems & source_stems) / len(stems)))
    return out


def min_support(text: str, passages: list[str]) -> float | None:
    """The weakest sentence's support (None if nothing was judgeable)."""
    supports = [v for _, v in sentence_supports(text, passages)]
    return min(supports) if supports else None


def ungrounded(text: str, passages: list[str], threshold: float) -> list[str]:
    """Sentences whose support is below the threshold."""
    return [sent for sent, v in sentence_supports(text, passages) if v < threshold]


# ---------------------------------------------------------------- 4. disclaimer
DISCLAIMER_TEXT = ("This is general health information from NIH sources, not medical advice. "
                   "Please talk to a doctor or other healthcare professional about your own situation.")


def add_disclaimer(text: str) -> str:
    return f"{text.rstrip()}\n\n{DISCLAIMER_TEXT}"
