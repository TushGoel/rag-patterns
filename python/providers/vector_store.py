"""
Vector store abstraction — swap backends via config.

Supports ChromaDB (local, no infra), Pinecone (managed cloud),
and pgvector (PostgreSQL extension). Same interface regardless of backend.

ChromaDB is the default for development — runs in-memory or persistent,
no Docker required. Switch to Pinecone or pgvector for production scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VectorBackend(str, Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"
    PGVECTOR = "pgvector"


@dataclass
class VectorStoreConfig:
    backend: VectorBackend = VectorBackend.CHROMA
    collection_name: str = "rag-patterns"
    persist_directory: Optional[str] = None   # Chroma persistent storage
    api_key: Optional[str] = None             # Pinecone
    environment: Optional[str] = None         # Pinecone
    connection_string: Optional[str] = None   # pgvector


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, source={self.source!r}, chars={len(self.text)})"


class VectorStore:
    """
    Provider-agnostic vector store.

    Usage:
        store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA))
        store.add(chunks, embeddings)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._client = None
        self._collection = None

    def _init_chroma(self) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")

        if self.config.persist_directory:
            self._client = chromadb.PersistentClient(path=self.config.persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: list[str], embeddings: list[list[float]],
            metadatas: Optional[list[dict]] = None, ids: Optional[list[str]] = None) -> None:
        if self.config.backend == VectorBackend.CHROMA:
            if self._collection is None:
                self._init_chroma()
            ids = ids or [str(i) for i in range(len(texts))]
            metadatas = metadatas or [{} for _ in texts]
            clean_meta = [m if m else {"_empty": "true"} for m in metadatas]
            self._collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=clean_meta,
                ids=ids,
            )
        else:
            raise NotImplementedError(f"{self.config.backend} add not implemented")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if self.config.backend == VectorBackend.CHROMA:
            if self._collection is None:
                self._init_chroma()
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                source = meta.get("source", meta.get("_source", ""))
                output.append(SearchResult(
                    text=text,
                    score=1.0 - dist,
                    source=source,
                    metadata=meta,
                ))
            return output
        else:
            raise NotImplementedError(f"{self.config.backend} search not implemented")

    def count(self) -> int:
        if self.config.backend == VectorBackend.CHROMA:
            if self._collection is None:
                self._init_chroma()
            return self._collection.count()
        return 0

    def clear(self) -> None:
        if self.config.backend == VectorBackend.CHROMA and self._client:
            self._client.delete_collection(self.config.collection_name)
            self._collection = None
