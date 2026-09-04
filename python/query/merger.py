"""
Answer merger — combine sub-answers from decomposed queries into a final response.

Query decomposition retrieves and answers each sub-query independently.
The merger combines those partial answers into a single coherent response,
deduplicating overlapping content and resolving contradictions.

Three merge strategies:
  ConcatMerger    — simple concatenation with section headers (fast, no LLM)
  SummaryMerger   — LLM synthesizes sub-answers into a unified response
  StructuredMerger — LLM produces a structured answer with citations per sub-query

When to use:
  ConcatMerger    — when sub-queries cover distinct topics, no overlap
  SummaryMerger   — when sub-queries overlap and you want a flowing answer
  StructuredMerger — when readers need to trace each claim back to a sub-query
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..providers.llm import LLMProvider


@dataclass
class MergedAnswer:
    original_query: str
    sub_queries: list[str]
    sub_answers: list[str]
    final_answer: str
    strategy: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MergedAnswer(query={self.original_query!r}, "
            f"sub_answers={len(self.sub_answers)}, "
            f"strategy={self.strategy})"
        )


SYNTHESIS_PROMPT = (
    "You have answers to sub-questions that together address a complex question.\n\n"
    "Original question: {query}\n\n"
    "{sub_qa_pairs}\n\n"
    "Synthesize these into a single, coherent, complete answer to the original question. "
    "Remove redundancy. Resolve any contradictions by noting them explicitly."
)

STRUCTURED_PROMPT = (
    "Answer the following complex question using the sub-question answers provided.\n\n"
    "Original question: {query}\n\n"
    "{sub_qa_pairs}\n\n"
    "Provide a structured answer with these sections:\n"
    "1. Summary (2-3 sentences)\n"
    "2. Detailed Answer (organized by aspect)\n"
    "3. Key Points (bullet list)"
)


class ConcatMerger:
    """
    Simple merger — concatenate sub-answers with section headers.
    No LLM call. Best when sub-queries cover non-overlapping topics.
    """

    def merge(
        self,
        query: str,
        sub_queries: list[str],
        sub_answers: list[str],
    ) -> MergedAnswer:
        sections = []
        for sq, sa in zip(sub_queries, sub_answers):
            sections.append(f"**{sq}**\n{sa}")
        final = "\n\n".join(sections)

        return MergedAnswer(
            original_query=query,
            sub_queries=sub_queries,
            sub_answers=sub_answers,
            final_answer=final,
            strategy="concat",
        )


class SummaryMerger:
    """
    LLM-based merger — synthesizes sub-answers into a unified response.
    Best when sub-queries overlap or you want a flowing, coherent answer.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def merge(
        self,
        query: str,
        sub_queries: list[str],
        sub_answers: list[str],
    ) -> MergedAnswer:
        pairs = "\n\n".join(
            f"Q{i+1}: {sq}\nA{i+1}: {sa}"
            for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers))
        )
        prompt = SYNTHESIS_PROMPT.format(query=query, sub_qa_pairs=pairs)
        response = self.llm.complete(prompt)

        return MergedAnswer(
            original_query=query,
            sub_queries=sub_queries,
            sub_answers=sub_answers,
            final_answer=response.content.strip(),
            strategy="summary",
            metadata={
                "synthesis_tokens": response.total_tokens,
                "synthesis_cost_usd": response.cost_usd,
            },
        )


class StructuredMerger:
    """
    LLM-based merger producing a structured answer with citations.
    Best when readers need to trace claims back to specific sub-queries.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def merge(
        self,
        query: str,
        sub_queries: list[str],
        sub_answers: list[str],
    ) -> MergedAnswer:
        pairs = "\n\n".join(
            f"Sub-question {i+1}: {sq}\nAnswer {i+1}: {sa}"
            for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers))
        )
        prompt = STRUCTURED_PROMPT.format(query=query, sub_qa_pairs=pairs)
        response = self.llm.complete(prompt)

        return MergedAnswer(
            original_query=query,
            sub_queries=sub_queries,
            sub_answers=sub_answers,
            final_answer=response.content.strip(),
            strategy="structured",
            metadata={
                "synthesis_tokens": response.total_tokens,
                "synthesis_cost_usd": response.cost_usd,
            },
        )
