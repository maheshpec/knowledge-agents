"""End-to-end answer-quality metrics (SPEC §9.2).

Two families:

* **ragas** — ``faithfulness`` and ``answer_relevance``. ragas is an optional
  dependency (``pip install '.[eval]'``) and needs an LLM + embeddings, so these
  wrappers take an injectable ``scorer`` and degrade to ``NaN`` when neither a
  scorer nor ragas is available. ``NaN`` is dropped by ``mean_aggregate``, so an
  offline run simply reports the metrics it could actually compute.
* **citation precision / recall** — an LLM *judge* decides, per claim, whether a
  cited chunk truly supports it (precision) and whether a supportable claim was
  left uncited (recall). The judge is injectable; the default offline judge uses
  lexical overlap so unit tests are deterministic and key-free.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from evaluation.metrics.base import MeanMetric, MetricResult, QueryOutcome

_NAN = float("nan")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


# --- citation judge ---------------------------------------------------------


@runtime_checkable
class CitationJudge(Protocol):
    """Decides whether evidence supports a claim, and whether a claim is supportable."""

    def supports(self, claim: str, evidence: str) -> bool: ...

    def is_supportable(self, claim: str, evidences: list[str]) -> bool: ...


class LexicalOverlapJudge:
    """Key-free deterministic judge: support iff token overlap clears a threshold.

    Used as the default offline judge and in unit tests. Not a substitute for the
    LLM judge in production — it only checks lexical overlap, not entailment.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def _overlap(self, claim: str, evidence: str) -> float:
        claim_tokens = _tokens(claim)
        if not claim_tokens:
            return 0.0
        return len(claim_tokens & _tokens(evidence)) / len(claim_tokens)

    def supports(self, claim: str, evidence: str) -> bool:
        return self._overlap(claim, evidence) >= self.threshold

    def is_supportable(self, claim: str, evidences: list[str]) -> bool:
        return any(self.supports(claim, ev) for ev in evidences)


def _evidence_by_chunk(outcome: QueryOutcome) -> dict[str, str]:
    return {c.chunk.chunk_id: c.chunk.text for c in outcome.candidates}


class CitationPrecision(MeanMetric):
    """Share of citations whose cited chunk actually supports the claim (LLM judge)."""

    name = "citation_precision"

    def __init__(self, judge: CitationJudge | None = None) -> None:
        self.judge = judge or LexicalOverlapJudge()

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gen = outcome.generation
        if gen is None or not gen.citations:
            # No citations to verify: vacuously precise (nothing wrong was cited).
            return MetricResult(name=self.name, value=1.0, detail={"citations": 0})
        evidence = _evidence_by_chunk(outcome)
        supported = 0
        for cite in gen.citations:
            claim = gen.text[cite.claim_span[0] : cite.claim_span[1]] or (cite.quote or "")
            ev = evidence.get(cite.source.chunk_id, "")
            if ev and self.judge.supports(claim, ev):
                supported += 1
        return MetricResult(
            name=self.name,
            value=supported / len(gen.citations),
            detail={"supported": supported, "citations": len(gen.citations)},
        )


class CitationRecall(MeanMetric):
    """Share of supportable claims that carry a citation (LLM judge)."""

    name = "citation_recall"

    def __init__(self, judge: CitationJudge | None = None) -> None:
        self.judge = judge or LexicalOverlapJudge()

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gen = outcome.generation
        if gen is None or not gen.text.strip():
            return MetricResult(name=self.name, value=0.0, detail={"claims": 0})
        evidences = [c.chunk.text for c in outcome.candidates]
        # Character ranges covered by a citation, to test "did this claim get cited".
        cited_spans = [c.claim_span for c in gen.citations]
        text = gen.text
        supportable = 0
        cited = 0
        cursor = 0
        for sentence in _sentences(text):
            start = text.find(sentence, cursor)
            cursor = start + len(sentence) if start >= 0 else cursor
            if not self.judge.is_supportable(sentence, evidences):
                continue
            supportable += 1
            end = start + len(sentence)
            if any(s < end and start < e for s, e in cited_spans):
                cited += 1
        if supportable == 0:
            return MetricResult(name=self.name, value=1.0, detail={"claims": 0})
        return MetricResult(
            name=self.name,
            value=cited / supportable,
            detail={"cited": cited, "supportable": supportable},
        )


# --- ragas wrappers ---------------------------------------------------------


@runtime_checkable
class AnswerScorer(Protocol):
    """Scores one (question, answer, contexts) triple in [0, 1]."""

    def __call__(self, question: str, answer: str, contexts: list[str]) -> float: ...


class _RagasMetric(MeanMetric):
    """Shared plumbing for ragas-backed metrics with an injectable scorer."""

    def __init__(self, scorer: AnswerScorer | None) -> None:
        self.scorer = scorer

    def _contexts(self, outcome: QueryOutcome) -> list[str]:
        return [c.chunk.text for c in outcome.candidates]

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gen = outcome.generation
        if gen is None or self.scorer is None:
            return MetricResult(name=self.name, value=_NAN, detail={"skipped": True})
        try:
            value = self.scorer(outcome.gold.query, gen.text, self._contexts(outcome))
        except Exception as exc:  # ragas/LLM failures must not abort a run
            return MetricResult(name=self.name, value=_NAN, detail={"error": str(exc)})
        return MetricResult(name=self.name, value=float(value))


class Faithfulness(_RagasMetric):
    """ragas faithfulness: are the answer's claims grounded in the contexts."""

    name = "faithfulness"

    def __init__(self, scorer: AnswerScorer | None = None) -> None:
        super().__init__(scorer or _try_ragas_scorer("faithfulness"))


class AnswerRelevance(_RagasMetric):
    """ragas answer relevance: does the answer address the question."""

    name = "answer_relevance"

    def __init__(self, scorer: AnswerScorer | None = None) -> None:
        super().__init__(scorer or _try_ragas_scorer("answer_relevancy"))


def _try_ragas_scorer(metric_name: str) -> AnswerScorer | None:
    """Build a ragas-backed scorer if ragas is importable, else ``None``.

    Imported lazily so the core install and unit tests never need ragas. The real
    ragas API requires a configured LLM + embeddings; we surface that lazily on
    first call rather than at construction so offline runs stay cheap.
    """
    try:  # pragma: no cover - exercised only when the optional dep is present
        import importlib

        importlib.import_module("ragas")
    except Exception:
        return None

    def _scorer(question: str, answer: str, contexts: list[str]) -> float:  # pragma: no cover
        from datasets import Dataset as HFDataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas import metrics as ragas_metrics  # type: ignore

        metric = getattr(ragas_metrics, metric_name)
        ds = HFDataset.from_dict(
            {"question": [question], "answer": [answer], "contexts": [contexts]}
        )
        result = evaluate(ds, metrics=[metric])
        return float(result[metric_name][0])

    return _scorer


def default_e2e_metrics(judge: CitationJudge | None = None) -> list[MeanMetric]:
    """The end-to-end metric suite from ``configs/eval.yaml``.

    ragas metrics self-disable (NaN) when ragas is unavailable; the citation
    metrics fall back to the lexical judge when none is injected.
    """
    return [
        Faithfulness(),
        AnswerRelevance(),
        CitationPrecision(judge),
        CitationRecall(judge),
    ]


__all__ = [
    "CitationJudge",
    "LexicalOverlapJudge",
    "CitationPrecision",
    "CitationRecall",
    "AnswerScorer",
    "Faithfulness",
    "AnswerRelevance",
    "default_e2e_metrics",
]
