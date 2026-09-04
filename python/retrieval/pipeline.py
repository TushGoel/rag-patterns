"""
RAG retrieval pipeline — vector, hybrid, and reranked retrieval.

Three composable patterns:
  VectorRetriever    — dense embedding similarity (fast, good recall)
  HybridRetriever    — dense + BM25 keyword (best of both worlds)
  RerankedRetriever  — adds cross-encoder reranking for precision

Each wraps the same VectorStore + Embedder so they're interchangeable.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..providers.embeddings import Embedder, EmbeddingProvider
from ..providers.vector_store import VectorStore, VectorStoreConfig, SearchResult
from ..chunking.strategies import Chunk


@dataclass
class RetrievalResult:
    query: str
    results: list[SearchResult]
    strategy: str
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def top(self, n: int = 3) -> list[SearchResult]:
        return self.results[:n]

    def context(self, n: int = 5, separator: str = "\n\n---\n\n") -> str:
        return separator.join(r.text for r in self.results[:n])

    def __repr__(self) -> str:
        return f"RetrievalResult(query={self.query!r}, results={len(self.results)}, strategy={self.strategy})"


class VectorRetriever:
    """
    Dense vector retrieval using embedding similarity.

    Embeds query, finds nearest neighbors in vector store.
    Fast, scales well, good recall on semantic similarity.

    Weakness: misses exact keyword matches (e.g., product IDs, names).
    Use HybridRetriever when exact match matters.
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.store = store or VectorStore(VectorStoreConfig())
        self.embedder = embedder or Embedder(EmbeddingProvider.LOCAL)

    def index(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        metadatas = [{"source": c.source, **c.metadata} for c in chunks]
        ids = [f"{c.source}_{c.chunk_index}" for c in chunks]
        self.store.add(texts, embeddings, metadatas, ids)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        import time
        start = time.time()
        query_embedding = self.embedder.embed(query)
        results = self.store.search(query_embedding, top_k=top_k)
        return RetrievalResult(
            query=query,
            results=results,
            strategy="vector",
            latency_ms=(time.time() - start) * 1000,
        )


class HybridRetriever:
    """
    Hybrid retrieval: dense vectors + BM25 keyword search.

    Combines semantic similarity (embedding) with exact keyword matching
    (BM25). Uses Reciprocal Rank Fusion (RRF) to merge ranked lists.

    When to use: documents with specific terminology, product names,
    IDs, or any query where exact words matter alongside meaning.

    RRF formula: score = Σ 1 / (k + rank_i) where k=60 (constant).
    """

    RRF_K = 60

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.store = store or VectorStore(VectorStoreConfig())
        self.embedder = embedder or Embedder(EmbeddingProvider.LOCAL)
        self._bm25 = None
        self._corpus: list[str] = []
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        self._chunks = chunks
        self._corpus = [c.text for c in chunks]
        tokenized = [text.lower().split() for text in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        metadatas = [{"source": c.source, **c.metadata} for c in chunks]
        ids = [f"{c.source}_{c.chunk_index}" for c in chunks]
        self.store.add(texts, embeddings, metadatas, ids)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        import time
        start = time.time()

        query_embedding = self.embedder.embed(query)
        vector_results = self.store.search(query_embedding, top_k=top_k * 2)

        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        rrf_scores: dict[str, float] = defaultdict(float)
        vector_texts: dict[str, SearchResult] = {}

        for rank, result in enumerate(vector_results):
            key = result.text[:100]
            rrf_scores[key] += 1.0 / (self.RRF_K + rank + 1)
            vector_texts[key] = result

        bm25_texts: dict[str, str] = {}
        for rank, idx in enumerate(bm25_ranked[:top_k * 2]):
            if idx < len(self._corpus):
                key = self._corpus[idx][:100]
                rrf_scores[key] += 1.0 / (self.RRF_K + rank + 1)
                bm25_texts[key] = self._corpus[idx]

        top_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        merged = []
        for key in top_keys:
            if key in vector_texts:
                r = vector_texts[key]
                merged.append(SearchResult(
                    text=r.text, score=rrf_scores[key],
                    source=r.source, metadata={**r.metadata, "rrf_score": rrf_scores[key]},
                ))
            elif key in bm25_texts:
                merged.append(SearchResult(
                    text=bm25_texts[key], score=rrf_scores[key],
                    source="bm25", metadata={"rrf_score": rrf_scores[key]},
                ))

        return RetrievalResult(
            query=query,
            results=merged,
            strategy="hybrid",
            latency_ms=(time.time() - start) * 1000,
            metadata={"rrf_k": self.RRF_K},
        )


class RerankedRetriever:
    """
    Two-stage retrieval: broad recall → cross-encoder precision.

    Stage 1: HybridRetriever fetches top_k * rerank_factor candidates.
    Stage 2: Cross-encoder reranks by query-document relevance score.

    Cross-encoders are more accurate than bi-encoders (embeddings) but
    too slow to run on the full corpus — that's why two stages.
    Use when precision matters more than speed (e.g., factual QA).
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_factor: int = 3,
    ) -> None:
        self.base = HybridRetriever(store=store, embedder=embedder)
        self.rerank_model = rerank_model
        self.rerank_factor = rerank_factor
        self._cross_encoder = None

    def index(self, chunks: list[Chunk]) -> None:
        self.base.index(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        import time
        start = time.time()

        candidates = self.base.retrieve(query, top_k=top_k * self.rerank_factor)

        try:
            from sentence_transformers import CrossEncoder
            if self._cross_encoder is None:
                self._cross_encoder = CrossEncoder(self.rerank_model)

            pairs = [(query, r.text) for r in candidates.results]
            scores = self._cross_encoder.predict(pairs)

            reranked = sorted(
                zip(candidates.results, scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            results = [
                SearchResult(
                    text=r.text,
                    score=float(score),
                    source=r.source,
                    metadata={**r.metadata, "rerank_score": float(score)},
                )
                for r, score in reranked
            ]
        except ImportError:
            results = candidates.results[:top_k]

        return RetrievalResult(
            query=query,
            results=results,
            strategy="reranked",
            latency_ms=(time.time() - start) * 1000,
            metadata={"rerank_model": self.rerank_model},
        )
