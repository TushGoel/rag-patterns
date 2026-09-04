"""
Chunking strategy benchmarking — compare quality across strategies.

Answers the question: which chunking strategy produces the best retrieval
for your specific content? Run this before committing to a strategy in production.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .strategies import Chunk


@dataclass
class ChunkStats:
    strategy: str
    total_chunks: int
    avg_chars: float
    min_chars: int
    max_chars: int
    std_dev: float
    avg_tokens_estimate: float

    @classmethod
    def from_chunks(cls, chunks: list["Chunk"], strategy: str) -> "ChunkStats":
        if not chunks:
            return cls(strategy=strategy, total_chunks=0, avg_chars=0,
                       min_chars=0, max_chars=0, std_dev=0, avg_tokens_estimate=0)
        sizes = [len(c.text) for c in chunks]
        return cls(
            strategy=strategy,
            total_chunks=len(chunks),
            avg_chars=statistics.mean(sizes),
            min_chars=min(sizes),
            max_chars=max(sizes),
            std_dev=statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
            avg_tokens_estimate=statistics.mean(sizes) * 4 // 3,
        )

    def __str__(self) -> str:
        return (
            f"[{self.strategy}] chunks={self.total_chunks} "
            f"avg={self.avg_chars:.0f}chars "
            f"min={self.min_chars} max={self.max_chars} "
            f"std={self.std_dev:.0f} "
            f"~{self.avg_tokens_estimate:.0f}tokens/chunk"
        )


def benchmark(text: str) -> dict[str, ChunkStats]:
    """
    Run all three chunking strategies on the same text and compare.

    Usage:
        results = benchmark(my_document_text)
        for strategy, stats in results.items():
            print(stats)
    """
    from .strategies import FixedChunker, SemanticChunker, RecursiveChunker

    strategies = {
        "fixed": FixedChunker(chunk_size=512, overlap=64),
        "semantic": SemanticChunker(max_chars=1000),
        "recursive": RecursiveChunker(chunk_size=512, overlap=64),
    }

    results = {}
    for name, chunker in strategies.items():
        chunks = chunker.chunk(text)
        results[name] = ChunkStats.from_chunks(chunks, strategy=name)

    return results
