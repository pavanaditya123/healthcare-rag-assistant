"""The question -> answer pipeline.

    redact PHI -> route -> retrieve -> threshold -> LLM -> output guardrails -> + disclaimer
                    |                      |          |              |
                 emergency /           "no reliable  LLM down /   invalid citation / unsafe /
                 injection /            info"        error         ungrounded -> sources-only
                 refusal (no LLM)
"""
import re
from dataclasses import asdict, dataclass, field

from app import guardrails, safety
from app.config import Settings
from app.llm import LLMError
from app.metrics import DEGRADED, GROUNDING_FLAGGED
from app.retrieval import Hit

SYSTEM_PROMPT = (
    "You are a health information assistant. Answer ONLY from the numbered NIH sources "
    "provided. End every paragraph with the numbers of the sources it uses, written exactly like "
    "[1] or [2][3] (square brackets, no other citation format). "
    "If the sources do not answer the question, say so. Never diagnose, never recommend "
    "doses, and suggest seeing a healthcare professional for personal medical decisions. "
    "Use plain language and at most 150 words."
)
MAX_SOURCE_CHARS = 1500  # keep the prompt small; MedQuAD answers can be very long
CITATION = re.compile(r"\[(\d+)\]")
DISCLAIMER = re.compile(r"doctor|health[- ]?care (?:professional|provider)|pharmacist|clinician", re.IGNORECASE)


@dataclass
class Answer:
    route: str  # emergency / prompt_injection / diagnosis_dose / out_of_scope / no_info / answered / degraded
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str | None = None
    context_cached: bool = False  # passages came from the context cache (retrieval skipped)
    blocked_by: str | None = None  # which guardrail replaced the LLM text, if any
    grounding_support: float | None = None  # weakest sentence's lexical support (guardrails.py)

    def to_dict(self) -> dict:
        return asdict(self)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    blocks = [
        f"[{i}] {h.payload['focus']} ({h.payload['url']})\n{h.payload['text'][:MAX_SOURCE_CHARS]}"
        for i, h in enumerate(hits, start=1)
    ]
    user = f"Question: {question}\n\nSources:\n" + "\n\n".join(blocks)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def citations_valid(text: str, n_sources: int) -> bool:
    """Rules:
      * every cited number must be a source we actually provided (no invented [7])
      * every paragraph must cite something, except a short "see your doctor" note
      * at least one citation overall
    """
    cited_all = [int(n) for n in CITATION.findall(text)]
    if not cited_all or any(n < 1 or n > n_sources for n in cited_all):
        return False
    for p in re.split(r"\n\s*\n", text.strip()):
        if not CITATION.search(p) and not DISCLAIMER.search(p):
            return False
    return True


def format_sources(hits: list[Hit]) -> list[dict]:
    return [
        {"n": i, "id": h.id, "title": h.payload["focus"], "url": h.payload["url"],
         "score": round(h.score, 3), "snippet": h.payload["text"][:300]}
        for i, h in enumerate(hits, start=1)
    ]


def sources_only(hits: list[Hit]) -> str:
    """Degraded mode: no generated text, just the top NIH passages, cited."""
    lines = ["The AI summary is unavailable right now. Most relevant NIH passages:"]
    lines += [f"- {h.payload['text'][:300].replace(chr(10), ': ', 1)}... [{i}]" for i, h in enumerate(hits[:3], 1)]
    return "\n".join(lines)


class Assistant:
    def __init__(self, retriever, llm, settings: Settings, context_cache=None):
        self.retriever, self.llm, self.s = retriever, llm, settings
        self.context_cache = context_cache  # optional AnswerCache(layer="context")

    def retrieve(self, question, sources, qtypes) -> tuple[list[Hit], bool]:
        """Top-k reranked hits, from the context cache when possible.
        Cached BEFORE the score threshold, so tuning MIN_RERANK_SCORE needs no flush."""
        key = f"{question}|{sorted(sources or [])}|{sorted(qtypes or [])}"
        if self.context_cache:
            cached = self.context_cache.get(key)
            if cached is not None:
                return [Hit(h["id"], h["score"], h["payload"]) for h in cached], True
        hits = self.retriever.search(question, sources=sources, qtypes=qtypes).reranked[: self.s.top_k]
        if self.context_cache:
            # Store only the text the prompt uses, to keep Redis small.
            self.context_cache.set(key, [
                {"id": h.id, "score": h.score,
                 "payload": {**h.payload, "text": h.payload["text"][:MAX_SOURCE_CHARS]}}
                for h in hits])
        return hits, False

    def _route_and_retrieve(self, question, sources, qtypes):
        """Shared by answer() and stream(): returns (final Answer, None) when we
        must not call the LLM, else (None, top hits). Also returns whether the
        passages came from the context cache."""
        route = safety.classify(question)
        if route == safety.EMERGENCY:
            return Answer(route, safety.EMERGENCY_MESSAGE), None, False
        if route == safety.PROMPT_INJECTION:
            return Answer(route, safety.PROMPT_INJECTION_MESSAGE), None, False
        if route == safety.DIAGNOSIS_DOSE:
            return Answer(route, safety.DIAGNOSIS_DOSE_MESSAGE), None, False
        if route == safety.OUT_OF_SCOPE:
            return Answer(route, safety.OUT_OF_SCOPE_MESSAGE), None, False
        hits, ctx = self.retrieve(question, sources, qtypes)
        if not hits or hits[0].score < self.s.min_rerank_score:
            return Answer("no_info", safety.NO_INFO_MESSAGE, context_cached=ctx), None, ctx
        return None, hits, ctx

    def finish(self, text: str, hits: list[Hit], srcs: list[dict], model, ctx: bool) -> Answer:
        """Output guardrails, shared by /ask and /ask/stream. Any failure means
        the user gets the NIH passages verbatim instead of generated text."""
        text = guardrails.normalize(text)
        passages = [h.payload["text"][:MAX_SOURCE_CHARS] for h in hits]
        support = guardrails.min_support(text, passages)
        weak = support is not None and support < self.s.grounding_min_support
        if not citations_valid(text, len(hits)):
            reason = "invalid_citations"
        elif guardrails.unsafe_output(text):
            reason = "unsafe_output:" + guardrails.unsafe_output(text)
        elif weak and self.s.grounding_enforce:
            reason = "ungrounded"
        else:
            if weak:  # shadow mode: count it, don't block
                GROUNDING_FLAGGED.labels("shadow").inc()
            return Answer("answered", guardrails.add_disclaimer(text.strip()), srcs, model, ctx,
                          grounding_support=support)
        if reason == "ungrounded":
            GROUNDING_FLAGGED.labels("blocked").inc()
        DEGRADED.labels(reason.split(":")[0]).inc()
        return Answer("degraded", guardrails.add_disclaimer(sources_only(hits)), srcs, model, ctx,
                      blocked_by=reason, grounding_support=support)

    def degraded_llm_down(self, hits, srcs, model, ctx) -> Answer:
        DEGRADED.labels("llm_unavailable").inc()
        return Answer("degraded", guardrails.add_disclaimer(sources_only(hits)), srcs, model, ctx,
                      blocked_by="llm_unavailable")

    def stream(self, question: str, sources=None, qtypes=None):
        """Yield events for /ask/stream:
          {"type": "sources", ...}  as soon as retrieval is done (~fast)
          {"type": "token", ...}    LLM text as it is generated
          {"type": "done", ...}     the final, validated Answer. If citations fail
                                    validation, "done" carries the sources-only answer
                                    and the client must replace what it displayed.
        """
        early, hits, ctx = self._route_and_retrieve(question, sources, qtypes)
        if early:
            yield {"type": "done", **early.to_dict()}
            return
        srcs = format_sources(hits)
        yield {"type": "sources", "sources": srcs, "context_cached": ctx}
        text, model = "", None
        try:
            for delta, model in self.llm.stream(build_messages(question, hits)):
                text += delta
                yield {"type": "token", "text": delta}
        except LLMError:
            yield {"type": "done", **self.degraded_llm_down(hits, srcs, model, ctx).to_dict()}
            return
        final = self.finish(text, hits, srcs, model, ctx)
        if final.route == "answered":  # the disclaimer is part of what the user sees
            yield {"type": "token", "text": "\n\n" + guardrails.DISCLAIMER_TEXT}
        yield {"type": "done", **final.to_dict()}

    def answer(self, question: str, sources: list[str] | None = None, qtypes: list[str] | None = None) -> Answer:
        """`question` must already be PHI-redacted (see main.py).
        `sources` / `qtypes` are optional metadata filters (see retrieval.search)."""
        early, hits, ctx = self._route_and_retrieve(question, sources, qtypes)
        if early:
            return early
        sources = format_sources(hits)

        try:
            text, model = self.llm.complete(build_messages(question, hits))
        except LLMError:
            return self.degraded_llm_down(hits, sources, None, ctx)
        return self.finish(text, hits, sources, model, ctx)
