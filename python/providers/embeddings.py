"""
Embedding provider abstraction.

Supports sentence-transformers (local, free), OpenAI text-embedding-3,
and a mock embedder for testing without API keys.

Local embeddings via sentence-transformers are the default —
no API costs, works offline, good quality for most use cases.
Switch to OpenAI for highest quality on production workloads.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional


class EmbeddingProvider(str, Enum):
    LOCAL = "local"          # sentence-transformers, runs on CPU
    OPENAI = "openai"        # text-embedding-3-small / large
    MOCK = "mock"            # deterministic fake embeddings for tests


class Embedder:
    """
    Provider-agnostic text embedder.

    Usage:
        embedder = Embedder(EmbeddingProvider.LOCAL)
        vector = embedder.embed("What is RAG?")
        vectors = embedder.embed_batch(["chunk 1", "chunk 2"])
    """

    DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
    MOCK_DIM = 384

    def __init__(
        self,
        provider: EmbeddingProvider = EmbeddingProvider.LOCAL,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._local_model = None

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == EmbeddingProvider.LOCAL:
            return self._local_embed(texts)
        elif self.provider == EmbeddingProvider.OPENAI:
            return self._openai_embed(texts)
        elif self.provider == EmbeddingProvider.MOCK:
            return self._mock_embed(texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        if self._local_model is None:
            self._local_model = SentenceTransformer(self.model or self.DEFAULT_LOCAL_MODEL)

        embeddings = self._local_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = OpenAI(api_key=self.api_key)
        model = self.model or self.DEFAULT_OPENAI_MODEL
        resp = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in resp.data]

    def _mock_embed(self, texts: list[str]) -> list[list[float]]:
        """Deterministic fake embeddings for testing — no API key needed."""
        result = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            import random
            rng = random.Random(seed)
            vec = [rng.uniform(-1, 1) for _ in range(self.MOCK_DIM)]
            norm = sum(x**2 for x in vec) ** 0.5
            result.append([x / norm for x in vec])
        return result

    @property
    def dimension(self) -> int:
        dims = {
            EmbeddingProvider.LOCAL: 384,
            EmbeddingProvider.OPENAI: 1536,
            EmbeddingProvider.MOCK: self.MOCK_DIM,
        }
        return dims.get(self.provider, 384)
