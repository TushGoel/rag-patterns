"""
Query Decomposition — multi-hop retrieval for complex questions.

Simple retrieval fails on complex questions that span multiple topics.
Query decomposition breaks a complex question into focused sub-queries,
retrieves for each independently, then merges deduplicated results.

Example:
  Complex: "How does HyDE improve retrieval compared to BM25 for technical docs?"
  Decomposed:
    1. "What is HyDE retrieval?"
    2. "What is BM25 retrieval?"
    3. "How do embedding-based and keyword-based retrieval compare?"

Each sub-query retrieves relevant chunks. The merged result contains
context for all three aspects — enabling a complete, accurate answer.

When to use: Multi-part questions, comparison queries, "how X relates to Y" questions.
Not ideal for: Simple factual lookups where decomposition adds unnecessary LLM calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..providers.llm import LLMProvider
from ..providers.embeddings import Embedder
from ..providers.vector_store import VectorStore, SearchResult
from ..chunking.strategies import Chunk
from .pipeline import RetrievalResult, VectorRetriever


@dataclass
class DecomposedResult:
    original_query: str
    sub_queries: list[str]
    sub_results: list[RetrievalResult]
    merged_results: list[SearchResult]
    latency_ms: float
    metadata: dict = field(default_factory=dict)

    @property
    def strategy(self) -> str:
        return "decomposed"

    def context(self, n: int = 5, separator: str = "\n\n---\n\n") -> str:
        return separator.join(r.text for r in self.merged_results[:n])

    def __repr__(self) -> str:
        return (
            f"DecomposedResult(query={self.original_query!r}, "
            f"sub_queries={len(self.sub_queries)}, "
            f"merged_results={len(self.merged_results)})"
        )


DECOMPOSE_PROMPT = (
    "Break the following question into {n} focused sub-questions that together "
    "cover all aspects needed to answer it. Each sub-question should be answerable "
    "independently from a document corpus.\n\n"
    "Question: {query}\n\n"
    "Output exactly {n} sub-questions, one per line, numbered 1. 2. 3. etc."
)


class QueryDecomposer:
    """
    Multi-hop retrieval via query decomposition.

    Decomposes a complex query into focused sub-queries using an LLM,
    retrieves for each sub-query independently, then merges and deduplicates.
    Results are ranked by frequency (chunks retrieved by multiple sub-queries rank higher).

    Usage:
        decomposer = QueryDecomposer(llm=llm, retriever=retriever, n_subqueries=3)
        result = decomposer.retrieve(
            "Compare hybrid retrieval and vector-only retrieval for code search"
        )
        # result.sub_queries = ["What is hybrid retrieval?", ...]
        # result.merged_results = deduplicated union of all sub-retrievals
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: VectorRetriever,
        n_subqueries: int = 3,
        top_k_per_subquery: int = 3,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.n_subqueries = n_subqueries
        self.top_k_per_subquery = top_k_per_subquery

    def index(self, chunks: list[Chunk]) -> None:
        self.retriever.index(chunks)

    def retrieve(self, query: str) -> DecomposedResult:
        import time
        start = time.time()

        sub_queries = self._decompose(query)
        sub_results = [
            self.retriever.retrieve(sq, top_k=self.top_k_per_subquery)
            for sq in sub_queries
        ]

        merged = self._merge(sub_results)

        return DecomposedResult(
            original_query=query,
            sub_queries=sub_queries,
            sub_results=sub_results,
            merged_results=merged,
            latency_ms=(time.time() - start) * 1000,
            metadata={"n_subqueries": self.n_subqueries},
        )

    def _decompose(self, query: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT.format(query=query, n=self.n_subqueries)
        response = self.llm.complete(prompt)
        lines = response.content.strip().splitlines()
        sub_queries = []
        for line in lines:
            line = line.strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 5:
                sub_queries.append(cleaned)
        return sub_queries[:self.n_subqueries] if sub_queries else [query]

    def _merge(self, sub_results: list[RetrievalResult]) -> list[SearchResult]:
        seen: dict[str, SearchResult] = {}
        frequency: dict[str, int] = {}

        for result in sub_results:
            for r in result.results:
                key = r.text[:100]
                frequency[key] = frequency.get(key, 0) + 1
                if key not in seen or r.score > seen[key].score:
                    seen[key] = r

        merged = sorted(
            seen.values(),
            key=lambda r: (frequency[r.text[:100]], r.score),
            reverse=True,
        )
        return merged
