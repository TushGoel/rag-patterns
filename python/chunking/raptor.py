"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Standard chunking splits documents into flat chunks. Long documents with
complex structure lose hierarchical context — a summary chunk at the top
level helps retrieve the right section before drilling into specifics.

RAPTOR builds a tree:
  Level 0: Original chunks (leaves)
  Level 1: Summaries of clusters of related chunks
  Level 2: Summaries of level-1 summaries
  ...until one root summary remains

Retrieval searches all levels simultaneously. A query about "distributed
system architecture" matches a level-2 summary; a query about a specific
retry policy matches a level-0 leaf chunk.

When to use: Long technical documents, books, codebases with multiple modules,
any corpus where section-level context matters for retrieval.
Not ideal for: Short documents where flat chunking is sufficient.

Reference: Sarthi et al., "RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval" (2024).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..providers.llm import LLMProvider
from .strategies import Chunk, FixedChunker


@dataclass
class RaptorNode:
    text: str
    level: int           # 0 = leaf, 1+ = summary
    children: list["RaptorNode"] = field(default_factory=list)
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def to_chunk(self, chunk_index: int = 0, total: int = 1) -> Chunk:
        return Chunk(
            text=self.text,
            source=self.source or f"raptor_level_{self.level}",
            chunk_index=chunk_index,
            total_chunks=total,
            start_char=0,
            end_char=len(self.text),
            metadata={"raptor_level": self.level, **self.metadata},
        )


SUMMARIZE_PROMPT = (
    "Summarize the following text concisely, preserving the key facts, "
    "concepts, and relationships. The summary will be used for document retrieval.\n\n"
    "Text:\n{text}\n\nSummary:"
)

CLUSTER_PROMPT = (
    "Given these text passages, identify {n_clusters} thematic groups. "
    "Return only the group assignments as: passage_index:group_index, one per line.\n\n"
    "{passages}\n\nAssignments:"
)


class RaptorBuilder:
    """
    Build a RAPTOR tree from document chunks.

    Clusters related chunks, summarizes each cluster, then recursively
    builds summary trees until a single root summary remains.
    All levels are returned as chunks ready for indexing.

    Usage:
        builder = RaptorBuilder(llm=llm, max_levels=3, cluster_size=5)
        all_chunks = builder.build(base_chunks)
        # all_chunks includes original + summary chunks at every level
        retriever.index(all_chunks)
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_levels: int = 3,
        cluster_size: int = 5,
        min_cluster_size: int = 2,
    ) -> None:
        self.llm = llm
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.min_cluster_size = min_cluster_size

    def build(self, base_chunks: list[Chunk]) -> list[Chunk]:
        """
        Build RAPTOR tree and return all chunks (leaves + summaries).
        Index all returned chunks — retrieval searches every level.
        """
        nodes = [
            RaptorNode(text=c.text, level=0, source=c.source, metadata=c.metadata)
            for c in base_chunks
        ]

        all_nodes = list(nodes)
        current_level = nodes

        for level in range(1, self.max_levels + 1):
            if len(current_level) < self.min_cluster_size:
                break

            clusters = self._cluster(current_level)
            summaries = []
            for cluster_nodes in clusters:
                if not cluster_nodes:
                    continue
                combined = "\n\n".join(n.text for n in cluster_nodes)
                summary_text = self._summarize(combined)
                summary_node = RaptorNode(
                    text=summary_text,
                    level=level,
                    children=cluster_nodes,
                    source=f"raptor_summary_l{level}",
                    metadata={"cluster_size": len(cluster_nodes)},
                )
                summaries.append(summary_node)
                all_nodes.append(summary_node)

            if not summaries:
                break
            current_level = summaries

        total = len(all_nodes)
        return [node.to_chunk(i, total) for i, node in enumerate(all_nodes)]

    def _cluster(self, nodes: list[RaptorNode]) -> list[list[RaptorNode]]:
        n_clusters = max(1, len(nodes) // self.cluster_size)
        clusters: list[list[RaptorNode]] = [[] for _ in range(n_clusters)]

        for i, node in enumerate(nodes):
            clusters[i % n_clusters].append(node)

        return [c for c in clusters if c]

    def _summarize(self, text: str) -> str:
        prompt = SUMMARIZE_PROMPT.format(text=text[:4000])
        response = self.llm.complete(prompt)
        return response.content.strip()
