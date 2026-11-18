"""Safety layer. Runs before anything is logged or sent to the LLM.

1. redact()   - mask PHI-like strings (emails, phones, SSNs, dates, MRNs,
                addresses, self-identified names) with regexes.
2. classify() - rule-based router:
     emergency      -> fixed helpline message, LLM never sees the question
     diagnosis_dose -> refuse (we never diagnose or recommend doses)
     out_of_scope   -> refuse (not a health question)
     normal         -> RAG pipeline

Rules are deliberately simple and auditable: for emergencies a regex you can
read beats a model you have to trust. The retrieval score threshold is a
second safety net for out-of-scope questions the rules miss.
"""
import re

EMERGENCY, DIAGNOSIS_DOSE, OUT_OF_SCOPE, NORMAL = "emergency", "diagnosis_dose", "out_of_scope", "normal"
ROUTES = [EMERGENCY, DIAGNOSIS_DOSE, OUT_OF_SCOPE, NORMAL]

# ---------------------------------------------------------------- PHI redaction
# Order matters: SSN before phone, so 123-45-6789 is not read as a phone number.
PHI_PATTERNS = [
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("PHONE", r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("MRN", r"\b(?:MRN|medical record(?: number)?|patient id|member id)\s*[:#]?\s*[A-Z0-9-]{4,}\b"),
    ("ADDRESS", r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b\.?"),
    ("NAME", r"(?<=\bmy name is )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
]
_PHI = [(label, re.compile(p, re.IGNORECASE if label in {"EMAIL", "MRN"} else 0)) for label, p in PHI_PATTERNS]
_MY_NAME = re.compile(r"\bmy name is\b", re.IGNORECASE)


def redact(text: str) -> str:
    text = _MY_NAME.sub("my name is", text)  # normalise case so the NAME look-behind matches
    for label, pattern in _PHI:
        text = pattern.sub(f"[{label}]", text)
    return text


# ---------------------------------------------------------------- routing
def _any(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


# Self-harm phrases are emergencies on their own (even "suicide warning
# signs?" gets the 988 line: a false alarm is cheap, a miss is not).
SELF_HARM = _any([
    r"kill(?:ing)? myself", r"end(?:ing)? my life", r"suicid\w*", r"wants? to die", r"hurt(?:ing)? myself",
    r"self[- ]harm", r"cut(?:ting)? myself",
])
# Acute symptoms count only when someone is experiencing them now
# ("my dad has chest pain"), not for informational questions
# ("what are the symptoms of a heart attack?").
PERSON = r"(?:i|i'm|im|i am|me|my|he|she|he's|she's|they|someone|somebody|baby|child|kid|husband|wife|mom|dad|mother|father|friend)"
ACUTE = (
    r"chest pain|chest (?:is )?(?:tight|pressure)|can'?t breathe|cannot breathe|not breathing|stopped breathing|"
    r"struggling to breathe|unconscious|unresponsive|passed out|won'?t wake up|"
    r"bleeding (?:heavily|a lot|badly|won'?t stop|that won'?t stop)|"
    r"face (?:is )?drooping|slurr\w* (?:speech|words|his speech|her speech)|"
    r"throat (?:is )?(?:closing|swelling)|choking|having a (?:heart attack|stroke)|anaphyla\w*|"
    r"(?:having|had|is having) a seizure|seizing|overdos\w*|took too many|took (?:a|the) whole|"
    r"swallowed (?:a |an |some )?(?:bleach|poison|detergent|batter\w*|magnets?|pills|\w+ pills)|(?:was|got|been|is) poisoned"
)
ACUTE_NOW = re.compile(rf"\b{PERSON}\b.{{0,60}}\b(?:{ACUTE})", re.IGNORECASE)

DIAGNOSIS_DOSE_RULE = _any([
    r"do i have", r"have i got", r"could i have", r"might i have", r"am i (?:sick|pregnant|diabetic|dying)",
    r"diagnose(?: me)?", r"what(?:'s| is) wrong with me", r"is (?:it|this|that|my)(?: \w+)? (?:cancer|cancerous|serious)",
    r"should i (?:take|stop|start|increase|decrease|double|skip|switch)",
    r"how (?:much|many) (?:mg|milligrams|ml|pills|tablets|capsules|units|puffs|doses)",
    r"how (?:much|many) \w+(?: \w+)? should (?:i|he|she|we|my \w+) (?:take|inject|give|use|have)",
    r"what (?:dose|dosage)", r"dosage", r"dose of", r"\d+\s?mg", r"how often should i take",
    r"can i (?:take|mix|combine) \w+(?: \w+)? (?:with|and)", r"prescribe",
])
OUT_OF_SCOPE_RULE = _any([
    r"weather", r"stock(?:s| price| market)?", r"bitcoin", r"crypto\w*", r"recipe", r"football", r"cricket",
    r"movie", r"song", r"lyrics", r"python", r"javascript", r"sql", r"write (?:a|me a) (?:poem|essay|story|code)",
    r"joke", r"capital of", r"president", r"election", r"flight", r"hotel", r"homework", r"translate",
])


def classify(text: str) -> str:
    """Return one of ROUTES. Emergency is checked first: safety beats everything."""
    if SELF_HARM.search(text) or ACUTE_NOW.search(text):
        return EMERGENCY
    if DIAGNOSIS_DOSE_RULE.search(text):
        return DIAGNOSIS_DOSE
    if OUT_OF_SCOPE_RULE.search(text):
        return OUT_OF_SCOPE
    return NORMAL


# ---------------------------------------------------------------- fixed responses
EMERGENCY_MESSAGE = (
    "This sounds like it could be an emergency. Please get help now:\n"
    "- Call 911 (or your local emergency number) for immediate danger, chest pain, "
    "trouble breathing, stroke signs, severe bleeding or loss of consciousness.\n"
    "- Call or text 988 (Suicide & Crisis Lifeline, 24/7) if you are thinking about "
    "suicide or self-harm.\n"
    "- Call Poison Control at 1-800-222-1222 for poisoning or overdose.\n"
    "This assistant cannot help with emergencies."
)
DIAGNOSIS_DOSE_MESSAGE = (
    "I can't diagnose conditions or recommend medicines or doses. Please ask your "
    "doctor or pharmacist, who can take your history into account. I can share general "
    "information, for example: \"What are the symptoms of type 2 diabetes?\""
)
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer general health questions using NIH sources (MedQuAD)."
)
NO_INFO_MESSAGE = (
    "I couldn't find reliable information about this in my NIH sources. "
    "Please check with a healthcare professional."
)
