"""
Semantic caching — cache retrieval/RAG results by query *meaning*, not
exact string match.

An exact-match cache misses on every paraphrase: "How does chunking affect
retrieval quality?" and "What impact does chunk size have on retrieval
quality?" are the same question but different cache keys. At query volume,
that means paying full retrieval + generation latency and cost for
questions the system has effectively already answered.

This cache embeds the incoming query and compares it against embeddings of
previously cached queries using cosine similarity. A hit means "semantically
close enough" (above `similarity_threshold`), not "identical text."

TTL + a bounded size keep the cache from serving stale answers or growing
without bound: entries expire after `ttl_seconds`, and the oldest entry is
evicted once `max_size` is exceeded (FIFO — simple and predictable, unlike
LRU it doesn't need per-access bookkeeping).

Caveat: cache whatever granularity makes sense for your use case — raw
RetrievalResult, a full RAG answer, or just merged context. This module is
value-agnostic; it caches whatever you pass to `set()`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..providers.embeddings import Embedder, EmbeddingProvider


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero-vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class CacheEntry:
    query: str
    embedding: list[float]
    value: Any
    created_at: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CacheResult:
    """Outcome of a `SemanticCache.get()` lookup."""
    hit: bool
    value: Optional[Any] = None
    matched_query: Optional[str] = None
    similarity: float = 0.0

    def __repr__(self) -> str:
        if self.hit:
            return f"CacheResult(hit=True, matched_query={self.matched_query!r}, similarity={self.similarity:.3f})"
        return f"CacheResult(hit=False, similarity={self.similarity:.3f})"


class SemanticCache:
    """
    Cache keyed by embedding similarity instead of exact query text.

    Usage:
        cache = SemanticCache(embedder=embedder, similarity_threshold=0.92, ttl_seconds=300)

        cached = cache.get(query)
        if cached.hit:
            return cached.value   # served without hitting the retriever/LLM

        result = retriever.retrieve(query)
        cache.set(query, result)
        return result

    `similarity_threshold` trades precision for hit rate: higher (closer to
    1.0) only matches near-identical queries; lower risks returning a cached
    answer for a subtly different question. 0.92 is a reasonable starting
    point for sentence-embedding cosine similarity — tune against your own
    query distribution.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 300.0,
        max_size: int = 1000,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in (0, 1], got {similarity_threshold}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.embedder = embedder or Embedder(EmbeddingProvider.MOCK)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._entries: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0

    def get(self, query: str, now: Optional[float] = None) -> CacheResult:
        """
        Look up the closest cached query by embedding similarity.

        `now` is accepted for deterministic TTL testing — production callers
        can omit it and it defaults to wall-clock time.
        """
        now = now if now is not None else time.time()
        self._evict_expired(now)

        if not self._entries:
            self._misses += 1
            return CacheResult(hit=False)

        query_embedding = self.embedder.embed(query)
        best_entry: Optional[CacheEntry] = None
        best_score = -1.0
        for entry in self._entries:
            score = _cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            self._hits += 1
            return CacheResult(
                hit=True, value=best_entry.value,
                matched_query=best_entry.query, similarity=best_score,
            )

        self._misses += 1
        return CacheResult(hit=False, similarity=max(best_score, 0.0))

    def set(self, query: str, value: Any, now: Optional[float] = None) -> None:
        """Cache `value` under `query`'s embedding. Evicts expired/oldest entries first."""
        now = now if now is not None else time.time()
        self._evict_expired(now)

        embedding = self.embedder.embed(query)
        self._entries.append(CacheEntry(query=query, embedding=embedding, value=value, created_at=now))

        while len(self._entries) > self.max_size:
            self._entries.pop(0)   # FIFO eviction — oldest entry first

    def _evict_expired(self, now: float) -> None:
        self._entries = [e for e in self._entries if (now - e.created_at) <= self.ttl_seconds]

    def clear(self) -> None:
        self._entries = []

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }
