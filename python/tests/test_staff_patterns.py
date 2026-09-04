"""
Tests for Staff-level RAG patterns.
All tests use mock LLM and mock embeddings — no API keys needed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from unittest.mock import MagicMock, patch
from python.chunking.strategies import FixedChunker
from python.chunking.raptor import RaptorBuilder, RaptorNode
from python.providers.embeddings import Embedder, EmbeddingProvider
from python.providers.vector_store import VectorStore, VectorStoreConfig, VectorBackend
from python.providers.llm import LLMResponse, Provider
from python.retrieval.pipeline import VectorRetriever, RetrievalResult
from python.retrieval.hyde import HyDERetriever, HyDEResult
from python.retrieval.query_decomposer import QueryDecomposer, DecomposedResult
from python.retrieval.agentic import AgenticRetriever, AgenticRAGResult


DOCS = [
    "HyDE generates a hypothetical answer to improve embedding similarity for retrieval.",
    "BM25 is a sparse keyword-based retrieval algorithm using term frequency.",
    "Dense retrieval uses neural embeddings to find semantically similar documents.",
    "Hybrid retrieval combines dense vectors and sparse BM25 for better recall.",
    "Query decomposition breaks complex questions into focused sub-queries.",
    "Agentic RAG uses a self-correcting loop to improve retrieval quality.",
    "RAPTOR builds a hierarchical summary tree for multi-level retrieval.",
    "Reranking uses a cross-encoder to improve precision after initial retrieval.",
]


def _mock_llm(response_text: str = "Mock LLM response."):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        content=response_text,
        model="mock",
        provider="mock",
        input_tokens=100,
        output_tokens=50,
        latency_ms=10.0,
        cost_usd=0.001,
    )
    return llm


def _make_chunks():
    chunks = []
    for i, doc in enumerate(DOCS):
        c = FixedChunker().chunk(doc, source=f"doc_{i}.txt")
        chunks.extend(c)
    return chunks


def _make_retriever(collection_name="test-staff"):
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA, collection_name=collection_name))
    embedder = Embedder(EmbeddingProvider.MOCK)
    return VectorRetriever(store=store, embedder=embedder)


# ── HyDE ────────────────────────────────────────────────────────────────────

def test_hyde_returns_result():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA, collection_name="test-hyde-1"))
    embedder = Embedder(EmbeddingProvider.MOCK)
    llm = _mock_llm("HyDE retrieval improves embedding similarity by generating a hypothetical answer.")

    retriever = HyDERetriever(llm=llm, embedder=embedder, store=store)
    retriever.index(_make_chunks())
    result = retriever.retrieve("What is HyDE retrieval?", top_k=3)

    assert isinstance(result, HyDEResult)
    assert result.strategy == "hyde"
    assert len(result.hypothetical_document) > 0
    assert len(result.results) == 3


def test_hyde_generates_hypothesis():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA, collection_name="test-hyde-2"))
    embedder = Embedder(EmbeddingProvider.MOCK)
    llm = _mock_llm("Dense retrieval uses neural embeddings for semantic search.")

    retriever = HyDERetriever(llm=llm, embedder=embedder, store=store)
    retriever.index(_make_chunks())
    result = retriever.retrieve("How does dense retrieval work?")

    assert result.hypothetical_document == "Dense retrieval uses neural embeddings for semantic search."
    assert llm.complete.called


def test_hyde_has_latency():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA, collection_name="test-hyde-3"))
    embedder = Embedder(EmbeddingProvider.MOCK)
    retriever = HyDERetriever(llm=_mock_llm(), embedder=embedder, store=store)
    retriever.index(_make_chunks())
    result = retriever.retrieve("test query")
    assert result.latency_ms >= 0


# ── Query Decomposition ──────────────────────────────────────────────────────

def test_decomposer_returns_result():
    retriever = _make_retriever("test-decomp-1")
    retriever.index(_make_chunks())
    llm = _mock_llm("1. What is BM25?\n2. What is dense retrieval?\n3. How do they compare?")

    decomposer = QueryDecomposer(llm=llm, retriever=retriever, n_subqueries=3)
    result = decomposer.retrieve("Compare BM25 and dense retrieval")

    assert isinstance(result, DecomposedResult)
    assert len(result.sub_queries) <= 3
    assert len(result.merged_results) > 0


def test_decomposer_deduplicates_results():
    retriever = _make_retriever("test-decomp-2")
    retriever.index(_make_chunks())
    llm = _mock_llm("1. What is retrieval?\n2. How does search work?")

    decomposer = QueryDecomposer(llm=llm, retriever=retriever, n_subqueries=2)
    result = decomposer.retrieve("How does information retrieval work?")

    texts = [r.text for r in result.merged_results]
    assert len(texts) == len(set(texts))


def test_decomposer_context_method():
    retriever = _make_retriever("test-decomp-3")
    retriever.index(_make_chunks())
    llm = _mock_llm("1. What is RAG?")

    decomposer = QueryDecomposer(llm=llm, retriever=retriever)
    result = decomposer.retrieve("Explain RAG patterns")
    context = result.context(n=3)
    assert isinstance(context, str)


def test_decomposer_fallback_on_bad_llm_response():
    retriever = _make_retriever("test-decomp-4")
    retriever.index(_make_chunks())
    llm = _mock_llm("")

    decomposer = QueryDecomposer(llm=llm, retriever=retriever, n_subqueries=3)
    result = decomposer.retrieve("fallback test query")
    assert isinstance(result, DecomposedResult)
    assert len(result.merged_results) >= 0


# ── Agentic RAG ──────────────────────────────────────────────────────────────

def test_agentic_converges_on_sufficient_context():
    retriever = _make_retriever("test-agentic-1")
    retriever.index(_make_chunks())
    llm = _mock_llm("YES\nContext is sufficient.\nFinal answer about agentic RAG.")

    agent = AgenticRetriever(llm=llm, retriever=retriever, max_iterations=3)
    result = agent.run("What is agentic RAG?")

    assert isinstance(result, AgenticRAGResult)
    assert result.total_iterations >= 1
    assert len(result.final_answer) > 0


def test_agentic_iterates_on_insufficient():
    retriever = _make_retriever("test-agentic-2")
    retriever.index(_make_chunks())

    call_count = [0]
    def mock_complete(prompt):
        call_count[0] += 1
        if call_count[0] <= 2:
            return LLMResponse(content="NO\nMissing info. better search query",
                               model="mock", provider="mock",
                               input_tokens=10, output_tokens=5, latency_ms=5.0)
        return LLMResponse(content="YES\nSufficient. The answer is here.",
                           model="mock", provider="mock",
                           input_tokens=10, output_tokens=5, latency_ms=5.0)

    llm = MagicMock()
    llm.complete.side_effect = mock_complete

    agent = AgenticRetriever(llm=llm, retriever=retriever, max_iterations=3)
    result = agent.run("Complex multi-part question?")

    assert result.total_iterations >= 2


def test_agentic_respects_max_iterations():
    retriever = _make_retriever("test-agentic-3")
    retriever.index(_make_chunks())
    llm = _mock_llm("NO\nNever enough. try again")

    agent = AgenticRetriever(llm=llm, retriever=retriever, max_iterations=2)
    result = agent.run("impossible query")

    assert result.total_iterations <= 2


def test_agentic_result_repr():
    retriever = _make_retriever("test-agentic-4")
    retriever.index(_make_chunks())
    llm = _mock_llm("YES\nDone.")

    agent = AgenticRetriever(llm=llm, retriever=retriever)
    result = agent.run("test")
    assert "agentic" in repr(result) or "AgenticRAGResult" in repr(result)


# ── RAPTOR ───────────────────────────────────────────────────────────────────

def test_raptor_builds_tree():
    llm = _mock_llm("This is a summary of the cluster.")
    builder = RaptorBuilder(llm=llm, max_levels=2, cluster_size=3)
    chunks = _make_chunks()
    all_chunks = builder.build(chunks)

    assert len(all_chunks) > len(chunks)


def test_raptor_includes_summary_level():
    llm = _mock_llm("Cluster summary: documents cover RAG retrieval techniques.")
    builder = RaptorBuilder(llm=llm, max_levels=2, cluster_size=3)
    chunks = _make_chunks()
    all_chunks = builder.build(chunks)

    summary_chunks = [c for c in all_chunks if c.metadata.get("raptor_level", 0) > 0]
    assert len(summary_chunks) > 0


def test_raptor_leaf_chunks_preserved():
    llm = _mock_llm("Summary text.")
    builder = RaptorBuilder(llm=llm, max_levels=1, cluster_size=4)
    base_chunks = _make_chunks()
    all_chunks = builder.build(base_chunks)

    leaf_texts = {c.text for c in base_chunks}
    output_texts = {c.text for c in all_chunks}
    assert leaf_texts.issubset(output_texts)


def test_raptor_node_to_chunk():
    node = RaptorNode(text="summary text", level=1, source="raptor_l1")
    chunk = node.to_chunk(chunk_index=0, total=1)
    assert chunk.text == "summary text"
    assert chunk.metadata["raptor_level"] == 1


def test_raptor_stops_at_min_cluster_size():
    llm = _mock_llm("One summary.")
    builder = RaptorBuilder(llm=llm, max_levels=5, cluster_size=10, min_cluster_size=100)
    chunks = _make_chunks()
    all_chunks = builder.build(chunks)
    assert len(all_chunks) == len(chunks)
