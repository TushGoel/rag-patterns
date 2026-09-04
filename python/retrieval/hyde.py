"""
HyDE — Hypothetical Document Embedding.

Standard RAG embeds the query and finds similar documents.
HyDE generates a hypothetical answer first, embeds that, then searches.

Why this works: A hypothetical answer looks like a document, not a question.
Embedding similarity between a hypothetical answer and real documents is higher
than between a question and those same documents — especially for factual QA.

Example:
  Query: "What is the capital of France?"
  Standard: embed("What is the capital of France?") → search
  HyDE: generate("Paris is the capital...") → embed that → search

  The embedding of "Paris is the capital of France, located along..."
  is much closer to factual documents about Paris than the question embedding.

When to use: Factual QA, knowledge-base search, technical documentation.
Not ideal for: Conversational queries, queries where you can't predict document style.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..providers.llm import LLMProvider
from ..providers.embeddings import Embedder
from ..providers.vector_store import VectorStore
from ..chunking.strategies import Chunk
from .pipeline import RetrievalResult, SearchResult


@dataclass
class HyDEResult(RetrievalResult):
    hypothetical_document: str = ""

    def __repr__(self) -> str:
        return (
            f"HyDEResult(query={self.query!r}, "
            f"hypothesis_chars={len(self.hypothetical_document)}, "
            f"results={len(self.results)})"
        )


HYDE_PROMPT = (
    "Write a short factual passage (2-4 sentences) that directly answers "
    "the following question. Write as if you are a document containing the answer.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)


class HyDERetriever:
    """
    Retrieval via Hypothetical Document Embedding.

    Generates a hypothetical answer using an LLM, embeds the hypothesis,
    and searches using that embedding instead of the raw query embedding.
    Improves retrieval precision for factual and technical questions.

    Usage:
        retriever = HyDERetriever(llm=llm, embedder=embedder, store=store)
        retriever.index(chunks)
        result = retriever.retrieve("What causes a thundering herd in distributed systems?")
        # result.hypothetical_document = "A thundering herd occurs when..."
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.store = store

    def index(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        metadatas = [{"source": c.source, **c.metadata} for c in chunks]
        ids = [f"{c.source}_{c.chunk_index}" for c in chunks]
        self.store.add(texts, embeddings, metadatas, ids)

    def retrieve(self, query: str, top_k: int = 5) -> HyDEResult:
        import time
        start = time.time()

        hypothesis_response = self.llm.complete(
            HYDE_PROMPT.format(query=query)
        )
        hypothesis = hypothesis_response.content.strip()

        hypothesis_embedding = self.embedder.embed(hypothesis)
        results = self.store.search(hypothesis_embedding, top_k=top_k)

        return HyDEResult(
            query=query,
            results=results,
            strategy="hyde",
            latency_ms=(time.time() - start) * 1000,
            hypothetical_document=hypothesis,
            metadata={
                "hypothesis_tokens": hypothesis_response.total_tokens,
                "hypothesis_cost_usd": hypothesis_response.cost_usd,
            },
        )
