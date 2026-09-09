"""
Role-based access control (RBAC) for retrieval.

A shared vector store often mixes documents scoped to different teams,
customers, or sensitivity levels. Retrieval alone doesn't know who's asking —
without a filtering step, whatever's in the corpus reaches the LLM context
(and the end user) regardless of whether the caller is authorized to see it.

This module tags chunks with the role(s)/scope(s) required to view them at
index time, then filters SearchResult candidates by the caller's granted
roles *before* they're assembled into context. Filtering happens outside the
LLM — never rely on a prompt instruction to make a model "not use" content it
was never authorized to see.

Fail-closed by default: a chunk with missing or empty `allowed_roles`
metadata, or a caller with no granted roles, is denied — not allowed. This is
the opposite of the common (and unsafe) default of treating "no ACL set" as
"public." Getting this default wrong turns every un-tagged document into an
accidental leak.

Composes with any existing retriever the same way RerankedRetriever composes
with HybridRetriever — wrap, don't replace.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..providers.vector_store import SearchResult
from .pipeline import RetrievalResult

# Metadata key chunks/documents use to declare who may see them.
ROLES_METADATA_KEY = "allowed_roles"

# Sentinel role that grants access regardless of a document's allowed_roles.
# Intended for trusted system/admin callers only — grant sparingly.
WILDCARD_ROLE = "*"


@dataclass
class AccessDecision:
    """One permit/deny decision, kept so filtering is defensible after the fact."""
    text_preview: str
    source: str
    allowed_roles: tuple[str, ...]
    caller_roles: tuple[str, ...]
    permitted: bool
    reason: str


@dataclass
class AuditEvent:
    """One retrieve() call's filtering outcome — the durable audit record."""
    event_id: str
    query: str
    caller_roles: tuple[str, ...]
    total_candidates: int
    permitted_count: int
    denied_count: int
    denied_sources: list[str]
    timestamp_ms: int

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "query": self.query[:200],
            "caller_roles": list(self.caller_roles),
            "total_candidates": self.total_candidates,
            "permitted_count": self.permitted_count,
            "denied_count": self.denied_count,
            "denied_sources": self.denied_sources,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class FilteredResult:
    """A RetrievalResult narrowed to what the caller is authorized to see."""
    query: str
    results: list[SearchResult]
    strategy: str
    caller_roles: tuple[str, ...]
    latency_ms: float = 0.0
    decisions: list[AccessDecision] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def denied_count(self) -> int:
        return sum(1 for d in self.decisions if not d.permitted)

    def top(self, n: int = 3) -> list[SearchResult]:
        return self.results[:n]

    def context(self, n: int = 5, separator: str = "\n\n---\n\n") -> str:
        return separator.join(r.text for r in self.results[:n])

    def __repr__(self) -> str:
        return (
            f"FilteredResult(query={self.query!r}, results={len(self.results)}, "
            f"denied={self.denied_count}, caller_roles={self.caller_roles})"
        )


def _extract_allowed_roles(metadata: dict) -> Optional[tuple[str, ...]]:
    """
    Read the allowed-roles tag off chunk/result metadata.

    Returns None when the tag is missing entirely, so callers can distinguish
    "not tagged" from "tagged but empty" — both fail closed, but the reason
    differs. Accepts a list/tuple, or a comma-separated string (Chroma and
    similar stores are happier with primitives at index time).
    """
    if ROLES_METADATA_KEY not in metadata:
        return None
    raw = metadata[ROLES_METADATA_KEY]
    if isinstance(raw, str):
        return tuple(r.strip() for r in raw.split(",") if r.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return None


def is_permitted(metadata: dict, caller_roles: tuple[str, ...]) -> tuple[bool, str]:
    """
    Decide whether `caller_roles` may see a chunk tagged with `metadata`.

    Fail-closed: a caller with no roles, or a document with missing/empty
    `allowed_roles`, is always denied — never treated as publicly visible.
    """
    if not caller_roles:
        return False, "caller has no granted roles (fail closed)"

    allowed_roles = _extract_allowed_roles(metadata)
    if allowed_roles is None:
        return False, "document has no allowed_roles metadata (fail closed)"
    if not allowed_roles:
        return False, "document allowed_roles is empty (fail closed)"

    caller_set = set(caller_roles)
    if WILDCARD_ROLE in caller_set:
        return True, "caller has wildcard role"
    if caller_set & set(allowed_roles):
        return True, "caller role matches allowed_roles"

    return False, "caller role not in allowed_roles"


class AccessControlledRetriever:
    """
    Wraps any retriever (VectorRetriever, HybridRetriever, RerankedRetriever, ...)
    and filters results by the caller's roles before returning them.

    Usage:
        base = HybridRetriever(store=store, embedder=embedder)
        retriever = AccessControlledRetriever(base)
        retriever.index(chunks)   # chunks carry {"allowed_roles": [...]} metadata

        result = retriever.retrieve("refund policy", caller_roles=("finance",), top_k=5)
        print(result.denied_count)        # chunks filtered out for this caller
        for event in retriever.audit_log:  # defensibility / compliance trail
            print(event.to_dict())
    """

    def __init__(self, base_retriever, audit: bool = True) -> None:
        self.base_retriever = base_retriever
        self.audit = audit
        self.audit_log: list[AuditEvent] = []

    def index(self, chunks) -> None:
        self.base_retriever.index(chunks)

    def retrieve(
        self,
        query: str,
        caller_roles: tuple[str, ...] = (),
        top_k: int = 5,
        over_fetch_factor: int = 3,
    ) -> FilteredResult:
        """
        Retrieve, then filter by role.

        `over_fetch_factor` pulls more candidates than `top_k` from the base
        retriever, since some fraction will be denied — without over-fetching,
        a caller with narrow access would silently get fewer than `top_k`
        results even when the corpus has enough visible chunks to fill it.
        """
        base_result: RetrievalResult = self.base_retriever.retrieve(
            query, top_k=top_k * over_fetch_factor
        )

        decisions: list[AccessDecision] = []
        permitted: list[SearchResult] = []

        for r in base_result.results:
            ok, reason = is_permitted(r.metadata, caller_roles)
            decisions.append(AccessDecision(
                text_preview=r.text[:80],
                source=r.source,
                allowed_roles=_extract_allowed_roles(r.metadata) or (),
                caller_roles=tuple(caller_roles),
                permitted=ok,
                reason=reason,
            ))
            if ok:
                permitted.append(r)
                if len(permitted) >= top_k:
                    break

        result = FilteredResult(
            query=query,
            results=permitted[:top_k],
            strategy=f"{base_result.strategy}+rbac",
            caller_roles=tuple(caller_roles),
            latency_ms=base_result.latency_ms,
            decisions=decisions,
            metadata={**base_result.metadata, "over_fetch_factor": over_fetch_factor},
        )

        if self.audit:
            self._record_audit(result)

        return result

    def _record_audit(self, result: FilteredResult) -> None:
        denied = [d for d in result.decisions if not d.permitted]
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            query=result.query,
            caller_roles=result.caller_roles,
            total_candidates=len(result.decisions),
            permitted_count=len(result.decisions) - len(denied),
            denied_count=len(denied),
            denied_sources=[d.source for d in denied],
            timestamp_ms=int(time.time() * 1000),
        )
        self.audit_log.append(event)
