"""
Chunking strategies — three composable approaches with quality benchmarking.

The chunking strategy is the most impactful RAG decision. Wrong chunk size
loses context. Wrong boundaries split sentences mid-thought. This module
makes the tradeoffs explicit and measurable.

Strategies:
  - FixedChunker: Simple, fast, predictable. Best for uniform text.
  - SemanticChunker: Sentence-boundary aware. Best for narrative docs.
  - RecursiveChunker: Hierarchical splitting. Best for structured content.

All chunkers produce Chunk objects with overlap tracking and source metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .benchmark import ChunkStats


@dataclass
class Chunk:
    """A document chunk ready for embedding and retrieval."""
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.text.split()) * 4 // 3

    def __repr__(self) -> str:
        return f"Chunk(index={self.chunk_index}/{self.total_chunks}, chars={len(self.text)}, source={self.source!r})"


class FixedChunker:
    """
    Split text into fixed-size chunks with overlap.

    Simple and predictable. Overlap ensures context isn't lost at boundaries.
    Best for: uniform text, API docs, structured data exports.

    Tradeoff: ignores sentence boundaries — may split mid-sentence.
    Use SemanticChunker when sentence integrity matters.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []

        chunks = []
        start = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(chunk_text)
            if end == len(text):
                break
            start += step

        total = len(chunks)
        result = []
        pos = 0
        for i, text_chunk in enumerate(chunks):
            idx = text.find(text_chunk, pos)
            start_char = idx if idx != -1 else pos
            result.append(Chunk(
                text=text_chunk,
                source=source,
                chunk_index=i,
                total_chunks=total,
                start_char=start_char,
                end_char=start_char + len(text_chunk),
                metadata={"strategy": "fixed", "chunk_size": self.chunk_size, "overlap": self.overlap},
            ))
            pos = start_char + len(text_chunk) - self.overlap

        return result

    def stats(self, chunks: list[Chunk]) -> ChunkStats:
        return ChunkStats.from_chunks(chunks, strategy="fixed")


class SemanticChunker:
    """
    Split text at sentence boundaries up to a target size.

    Preserves sentence integrity — no mid-sentence splits.
    Groups sentences until adding the next would exceed max_chars.
    Best for: articles, documentation, conversational text.

    Tradeoff: chunk sizes vary — some very short, some near max_chars.
    Use FixedChunker when uniform size matters (e.g., embedding cost control).
    """

    def __init__(self, max_chars: int = 1000, overlap_sentences: int = 1) -> None:
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        groups: list[list[str]] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            if current and current_len + len(sentence) + 1 > self.max_chars:
                groups.append(current[:])
                current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                current_len = sum(len(s) for s in current)
            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            groups.append(current)

        total = len(groups)
        chunks = []
        char_pos = 0
        for i, group in enumerate(groups):
            chunk_text = " ".join(group)
            idx = text.find(chunk_text[:50], char_pos)
            start_char = idx if idx != -1 else char_pos
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_index=i,
                total_chunks=total,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={
                    "strategy": "semantic",
                    "sentence_count": len(group),
                    "max_chars": self.max_chars,
                },
            ))
            char_pos = start_char + len(chunk_text)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def stats(self, chunks: list[Chunk]) -> ChunkStats:
        return ChunkStats.from_chunks(chunks, strategy="semantic")


class RecursiveChunker:
    """
    Hierarchical splitting using priority separator list.

    Tries to split on paragraph boundaries first, then sentences,
    then words, then characters — preserving as much structure as possible.
    Best for: code, structured markdown, documents with clear sections.

    Tradeoff: more complex, slightly slower. Worth it for code and docs
    where paragraph/section boundaries carry semantic meaning.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: Optional[list[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        raw_chunks = self._split(text, self.separators)
        merged = self._merge(raw_chunks)

        total = len(merged)
        chunks = []
        char_pos = 0
        for i, chunk_text in enumerate(merged):
            idx = text.find(chunk_text[:30], char_pos)
            start_char = idx if idx != -1 else char_pos
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_index=i,
                total_chunks=total,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={
                    "strategy": "recursive",
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                },
            ))
            char_pos = start_char + len(chunk_text)

        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)

        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split(part, remaining))

        return result

    def _merge(self, parts: list[str]) -> list[str]:
        merged = []
        current = ""

        for part in parts:
            if current and len(current) + len(part) + 1 > self.chunk_size:
                merged.append(current)
                overlap_text = current[-self.overlap:] if self.overlap else ""
                current = overlap_text + " " + part if overlap_text else part
            else:
                current = current + " " + part if current else part

        if current:
            merged.append(current)

        return merged

    def stats(self, chunks: list[Chunk]) -> ChunkStats:
        return ChunkStats.from_chunks(chunks, strategy="recursive")
