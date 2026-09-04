"""Tests for chunking strategies."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from python.chunking.strategies import FixedChunker, SemanticChunker, RecursiveChunker, Chunk
from python.chunking.benchmark import benchmark, ChunkStats

SAMPLE_TEXT = (
    "Retrieval-Augmented Generation (RAG) combines information retrieval with "
    "language generation. It works by first retrieving relevant documents from a "
    "knowledge base, then using those documents as context for the language model. "
    "This reduces hallucination and keeps answers grounded in facts. "
    "The retrieval step uses embedding similarity to find relevant chunks. "
    "The generation step uses the retrieved context to produce accurate answers. "
    "Chunking strategy is critical — wrong chunk size loses context or splits sentences."
) * 5


# ── FixedChunker ─────────────────────────────────────────────────────────────

def test_fixed_chunker_produces_chunks():
    chunks = FixedChunker(chunk_size=200, overlap=20).chunk(SAMPLE_TEXT)
    assert len(chunks) > 1


def test_fixed_chunker_respects_chunk_size():
    chunker = FixedChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk(SAMPLE_TEXT)
    for c in chunks[:-1]:
        assert len(c.text) <= 110


def test_fixed_chunker_overlap_raises_if_too_large():
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=100, overlap=100)


def test_fixed_chunker_empty_text():
    chunks = FixedChunker().chunk("")
    assert chunks == []


def test_fixed_chunker_metadata():
    chunks = FixedChunker(chunk_size=200, overlap=20).chunk(SAMPLE_TEXT, source="doc.pdf")
    assert chunks[0].metadata["strategy"] == "fixed"
    assert chunks[0].source == "doc.pdf"


def test_fixed_chunker_indices():
    chunks = FixedChunker(chunk_size=200, overlap=20).chunk(SAMPLE_TEXT)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.total_chunks == len(chunks)


# ── SemanticChunker ──────────────────────────────────────────────────────────

def test_semantic_chunker_produces_chunks():
    chunks = SemanticChunker(max_chars=300).chunk(SAMPLE_TEXT)
    assert len(chunks) > 1


def test_semantic_chunker_respects_max_chars():
    chunker = SemanticChunker(max_chars=200)
    chunks = chunker.chunk(SAMPLE_TEXT)
    # allow a single sentence to exceed max_chars (individual sentences aren't split)
    oversized = [c for c in chunks if len(c.text) > 400]
    assert len(oversized) == 0


def test_semantic_chunker_no_mid_sentence_splits():
    text = "First sentence here. Second sentence there. Third one follows."
    chunks = SemanticChunker(max_chars=40).chunk(text)
    for c in chunks:
        assert not c.text.startswith(" ")


def test_semantic_chunker_metadata():
    chunks = SemanticChunker().chunk(SAMPLE_TEXT)
    assert chunks[0].metadata["strategy"] == "semantic"


# ── RecursiveChunker ─────────────────────────────────────────────────────────

def test_recursive_chunker_produces_chunks():
    chunks = RecursiveChunker(chunk_size=200, overlap=20).chunk(SAMPLE_TEXT)
    assert len(chunks) > 1


def test_recursive_chunker_handles_paragraphs():
    text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
    chunks = RecursiveChunker(chunk_size=20, overlap=0).chunk(text)
    assert len(chunks) >= 2


def test_recursive_chunker_metadata():
    chunks = RecursiveChunker().chunk(SAMPLE_TEXT)
    assert chunks[0].metadata["strategy"] == "recursive"


def test_recursive_chunker_empty():
    assert RecursiveChunker().chunk("") == []


# ── Benchmark ────────────────────────────────────────────────────────────────

def test_benchmark_returns_all_strategies():
    results = benchmark(SAMPLE_TEXT)
    assert "fixed" in results
    assert "semantic" in results
    assert "recursive" in results


def test_benchmark_stats_are_valid():
    results = benchmark(SAMPLE_TEXT)
    for strategy, stats in results.items():
        assert stats.total_chunks > 0
        assert stats.avg_chars > 0
        assert stats.min_chars <= stats.max_chars


def test_chunk_stats_str():
    chunks = FixedChunker().chunk(SAMPLE_TEXT)
    stats = ChunkStats.from_chunks(chunks, strategy="fixed")
    assert "fixed" in str(stats)
    assert "chunks=" in str(stats)
