"""
RAG evaluation metrics — faithfulness, relevance, and answer quality.

The most common RAG failure modes:
  - Hallucination: answer not grounded in retrieved context
  - Low relevance: retrieved chunks don't address the query
  - Context loss: answer ignores relevant parts of context

These metrics make failures measurable, not just observable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..retrieval.pipeline import RetrievalResult
from ..providers.llm import LLMProvider, LLMResponse


@dataclass
class FaithfulnessScore:
    """Does the answer come from the context, or did the model hallucinate?"""
    score: float              # 0.0 (hallucinated) to 1.0 (fully grounded)
    grounded_claims: int
    total_claims: int
    verdict: str              # "FAITHFUL", "PARTIAL", "HALLUCINATED"

    @classmethod
    def from_score(cls, score: float, grounded: int, total: int) -> "FaithfulnessScore":
        if score >= 0.8:
            verdict = "FAITHFUL"
        elif score >= 0.5:
            verdict = "PARTIAL"
        else:
            verdict = "HALLUCINATED"
        return cls(score=score, grounded_claims=grounded, total_claims=total, verdict=verdict)


@dataclass
class RelevanceScore:
    """Is the retrieved context relevant to the query?"""
    score: float              # 0.0 (irrelevant) to 1.0 (highly relevant)
    relevant_chunks: int
    total_chunks: int
    verdict: str              # "RELEVANT", "PARTIAL", "IRRELEVANT"

    @classmethod
    def from_score(cls, score: float, relevant: int, total: int) -> "RelevanceScore":
        if score >= 0.7:
            verdict = "RELEVANT"
        elif score >= 0.4:
            verdict = "PARTIAL"
        else:
            verdict = "IRRELEVANT"
        return cls(score=score, relevant_chunks=relevant, total_chunks=total, verdict=verdict)


@dataclass
class EvalResult:
    query: str
    answer: str
    faithfulness: FaithfulnessScore
    relevance: RelevanceScore
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_tokens: int
    cost_usd: float
    metadata: dict = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        return (self.faithfulness.score + self.relevance.score) / 2

    def __str__(self) -> str:
        return (
            f"EvalResult(\n"
            f"  query={self.query!r}\n"
            f"  faithfulness={self.faithfulness.score:.2f} ({self.faithfulness.verdict})\n"
            f"  relevance={self.relevance.score:.2f} ({self.relevance.verdict})\n"
            f"  overall={self.overall_score:.2f}\n"
            f"  latency={self.retrieval_latency_ms + self.generation_latency_ms:.0f}ms\n"
            f"  cost=${self.cost_usd:.4f}\n"
            f")"
        )


class RAGEvaluator:
    """
    End-to-end RAG quality evaluation.

    Measures faithfulness (grounding) and relevance (context quality)
    using LLM-as-judge — the same approach used in RAGAS and TruLens.

    Usage:
        evaluator = RAGEvaluator(llm=LLMProvider(config))
        result = evaluator.evaluate(query, answer, retrieval_result)
        print(result)
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieval: RetrievalResult,
    ) -> EvalResult:
        faithfulness = self._score_faithfulness(answer, retrieval.context())
        relevance = self._score_relevance(query, retrieval)

        total_tokens = 0
        cost = 0.0

        return EvalResult(
            query=query,
            answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            retrieval_latency_ms=retrieval.latency_ms,
            generation_latency_ms=0.0,
            total_tokens=total_tokens,
            cost_usd=cost,
        )

    def _score_faithfulness(self, answer: str, context: str) -> FaithfulnessScore:
        claims = self._extract_claims(answer)
        if not claims:
            return FaithfulnessScore.from_score(1.0, 0, 0)

        grounded = 0
        for claim in claims:
            prompt = (
                f"Context:\n{context}\n\n"
                f"Claim: {claim}\n\n"
                "Is this claim supported by the context above? "
                "Answer only YES or NO."
            )
            resp = self.llm.complete(prompt)
            if "yes" in resp.content.lower():
                grounded += 1

        score = grounded / len(claims)
        return FaithfulnessScore.from_score(score, grounded, len(claims))

    def _score_relevance(self, query: str, retrieval: RetrievalResult) -> RelevanceScore:
        if not retrieval.results:
            return RelevanceScore.from_score(0.0, 0, 0)

        relevant = 0
        for result in retrieval.results:
            prompt = (
                f"Query: {query}\n\n"
                f"Retrieved text:\n{result.text}\n\n"
                "Is this retrieved text relevant to answering the query? "
                "Answer only YES or NO."
            )
            resp = self.llm.complete(prompt)
            if "yes" in resp.content.lower():
                relevant += 1

        score = relevant / len(retrieval.results)
        return RelevanceScore.from_score(score, relevant, len(retrieval.results))

    def _extract_claims(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]


class EvalPipeline:
    """
    Batch evaluation across multiple queries.

    Produces aggregate metrics useful for comparing retrieval strategies
    or tracking quality over time.

    Usage:
        pipeline = EvalPipeline(evaluator)
        results = pipeline.run(queries, answers, retrievals)
        pipeline.summary(results)
    """

    def __init__(self, evaluator: RAGEvaluator) -> None:
        self.evaluator = evaluator

    def run(
        self,
        queries: list[str],
        answers: list[str],
        retrievals: list[RetrievalResult],
    ) -> list[EvalResult]:
        results = []
        for query, answer, retrieval in zip(queries, answers, retrievals):
            result = self.evaluator.evaluate(query, answer, retrieval)
            results.append(result)
        return results

    def summary(self, results: list[EvalResult]) -> dict:
        if not results:
            return {}

        avg_faithfulness = sum(r.faithfulness.score for r in results) / len(results)
        avg_relevance = sum(r.relevance.score for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)
        total_cost = sum(r.cost_usd for r in results)
        avg_latency = sum(r.retrieval_latency_ms + r.generation_latency_ms for r in results) / len(results)

        hallucinated = sum(1 for r in results if r.faithfulness.verdict == "HALLUCINATED")
        irrelevant = sum(1 for r in results if r.relevance.verdict == "IRRELEVANT")

        summary = {
            "total_queries": len(results),
            "avg_faithfulness": round(avg_faithfulness, 3),
            "avg_relevance": round(avg_relevance, 3),
            "avg_overall": round(avg_overall, 3),
            "hallucinated_count": hallucinated,
            "irrelevant_retrievals": irrelevant,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(avg_latency, 1),
        }

        print("\n=== RAG Eval Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        return summary
