"""Tests for providers — embeddings and vector store."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from python.providers.embeddings import Embedder, EmbeddingProvider
from python.providers.vector_store import VectorStore, VectorStoreConfig, VectorBackend, SearchResult


# ── Embedder ─────────────────────────────────────────────────────────────────

def test_mock_embedder_returns_vector():
    e = Embedder(EmbeddingProvider.MOCK)
    vec = e.embed("test text")
    assert isinstance(vec, list)
    assert len(vec) == 384


def test_mock_embedder_is_deterministic():
    e = Embedder(EmbeddingProvider.MOCK)
    v1 = e.embed("same text")
    v2 = e.embed("same text")
    assert v1 == v2


def test_mock_embedder_different_texts_differ():
    e = Embedder(EmbeddingProvider.MOCK)
    v1 = e.embed("text one")
    v2 = e.embed("text two")
    assert v1 != v2


def test_mock_embedder_batch():
    e = Embedder(EmbeddingProvider.MOCK)
    texts = ["text a", "text b", "text c"]
    vecs = e.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


def test_mock_embedder_normalized():
    e = Embedder(EmbeddingProvider.MOCK)
    vec = e.embed("normalize me")
    norm = sum(x**2 for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_embedder_dimension():
    assert Embedder(EmbeddingProvider.MOCK).dimension == 384
    assert Embedder(EmbeddingProvider.LOCAL).dimension == 384
    assert Embedder(EmbeddingProvider.OPENAI).dimension == 1536


# ── VectorStore ───────────────────────────────────────────────────────────────

def test_chroma_add_and_search():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA))
    e = Embedder(EmbeddingProvider.MOCK)

    texts = ["document about RAG", "document about chunking", "document about embeddings"]
    embeddings = e.embed_batch(texts)
    store.add(texts, embeddings)

    results = store.search(e.embed("RAG retrieval"), top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(isinstance(r.score, float) for r in results)


def test_chroma_count():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA))
    e = Embedder(EmbeddingProvider.MOCK)
    texts = ["a", "b", "c"]
    store.add(texts, e.embed_batch(texts))
    assert store.count() == 3


def test_chroma_search_returns_sources():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA, collection_name="test-sources"))
    e = Embedder(EmbeddingProvider.MOCK)
    texts = ["retrieval content about sources"]
    metadatas = [{"source": "test.pdf"}]
    store.add(texts, e.embed_batch(texts), metadatas=metadatas)
    results = store.search(e.embed("retrieval content"), top_k=1)
    assert results[0].source == "test.pdf"


def test_chroma_clear():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA))
    e = Embedder(EmbeddingProvider.MOCK)
    store.add(["test"], e.embed_batch(["test"]))
    store.clear()
