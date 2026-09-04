"""Tests for retrieval pipeline using mock embeddings."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from python.chunking.strategies import FixedChunker
from python.providers.embeddings import Embedder, EmbeddingProvider
from python.providers.vector_store import VectorStore, VectorStoreConfig, VectorBackend
from python.retrieval.pipeline import VectorRetriever, HybridRetriever, RetrievalResult

DOCS = [
    "RAG combines retrieval with language generation to reduce hallucination.",
    "Chunking strategy determines how documents are split for embedding.",
    "Vector search finds semantically similar documents using embeddings.",
    "Hybrid retrieval combines dense and sparse methods for better recall.",
    "Evaluation metrics measure faithfulness and relevance of RAG answers.",
    "Reranking improves precision by scoring candidates with a cross-encoder.",
]


def _make_retriever():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA))
    embedder = Embedder(EmbeddingProvider.MOCK)
    return VectorRetriever(store=store, embedder=embedder)


def _index_docs(retriever):
    chunks = []
    for i, doc in enumerate(DOCS):
        c = FixedChunker().chunk(doc, source=f"doc_{i}.txt")
        chunks.extend(c)
    retriever.index(chunks)
    return chunks


# ── VectorRetriever ───────────────────────────────────────────────────────────

def test_vector_retriever_returns_results():
    r = _make_retriever()
    _index_docs(r)
    result = r.retrieve("What is RAG?", top_k=3)
    assert isinstance(result, RetrievalResult)
    assert len(result.results) == 3


def test_vector_retriever_strategy_label():
    r = _make_retriever()
    _index_docs(r)
    result = r.retrieve("chunking", top_k=2)
    assert result.strategy == "vector"


def test_vector_retriever_has_latency():
    r = _make_retriever()
    _index_docs(r)
    result = r.retrieve("retrieval", top_k=2)
    assert result.latency_ms >= 0


def test_retrieval_result_context():
    r = _make_retriever()
    _index_docs(r)
    result = r.retrieve("embeddings", top_k=3)
    context = result.context(n=2)
    assert isinstance(context, str)
    assert len(context) > 0


def test_retrieval_result_top():
    r = _make_retriever()
    _index_docs(r)
    result = r.retrieve("evaluation", top_k=5)
    top = result.top(2)
    assert len(top) == 2


def test_retrieval_scores_are_floats():
    r = _make_retriever()
    _index_docs(r)
    result = r.retrieve("vector search", top_k=3)
    for res in result.results:
        assert isinstance(res.score, float)


# ── HybridRetriever ───────────────────────────────────────────────────────────

def test_hybrid_retriever_returns_results():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        import pytest; pytest.skip("rank-bm25 not installed")

    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA))
    embedder = Embedder(EmbeddingProvider.MOCK)
    r = HybridRetriever(store=store, embedder=embedder)

    chunks = []
    for i, doc in enumerate(DOCS):
        c = FixedChunker().chunk(doc, source=f"doc_{i}.txt")
        chunks.extend(c)
    r.index(chunks)

    result = r.retrieve("hybrid retrieval recall", top_k=3)
    assert isinstance(result, RetrievalResult)
    assert result.strategy == "hybrid"
    assert len(result.results) <= 3
