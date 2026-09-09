"""Tests for role-based retrieval access control."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from python.chunking.strategies import Chunk
from python.providers.embeddings import Embedder, EmbeddingProvider
from python.providers.vector_store import VectorStore, VectorStoreConfig, VectorBackend, SearchResult
from python.retrieval.pipeline import VectorRetriever
from python.retrieval.access_control import (
    AccessControlledRetriever,
    FilteredResult,
    is_permitted,
    ROLES_METADATA_KEY,
    WILDCARD_ROLE,
)


def _chunk(text, source, chunk_index, allowed_roles=None, metadata_extra=None):
    metadata = dict(metadata_extra or {})
    if allowed_roles is not None:
        metadata[ROLES_METADATA_KEY] = allowed_roles
    return Chunk(
        text=text,
        source=source,
        chunk_index=chunk_index,
        total_chunks=1,
        start_char=0,
        end_char=len(text),
        metadata=metadata,
    )


def _make_retriever(collection_name="test-access-control"):
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA, collection_name=collection_name))
    embedder = Embedder(EmbeddingProvider.MOCK)
    base = VectorRetriever(store=store, embedder=embedder)
    return AccessControlledRetriever(base)


CHUNKS = [
    _chunk("Engineering runbook for on-call rotations.", "eng.md", 0, allowed_roles=["engineering"]),
    _chunk("Finance quarterly budget projections.", "finance.md", 0, allowed_roles=["finance"]),
    _chunk("Company-wide holiday calendar.", "holidays.md", 0, allowed_roles=["engineering", "finance"]),
    _chunk("Untagged legacy document with no ACL set.", "legacy.md", 0),  # no allowed_roles at all
]


# ── is_permitted (unit-level, no retrieval needed) ─────────────────────────

def test_authorized_role_is_permitted():
    ok, reason = is_permitted({ROLES_METADATA_KEY: ["engineering"]}, ("engineering",))
    assert ok is True


def test_unauthorized_role_is_denied():
    ok, reason = is_permitted({ROLES_METADATA_KEY: ["finance"]}, ("engineering",))
    assert ok is False
    assert "allowed_roles" in reason


def test_missing_role_metadata_fails_closed():
    """No allowed_roles key at all → deny, never default-allow."""
    ok, reason = is_permitted({}, ("engineering", "finance", "admin"))
    assert ok is False
    assert "no allowed_roles metadata" in reason


def test_empty_allowed_roles_fails_closed():
    ok, reason = is_permitted({ROLES_METADATA_KEY: []}, ("engineering",))
    assert ok is False
    assert "empty" in reason


def test_caller_with_no_roles_is_denied_even_for_open_docs():
    ok, reason = is_permitted({ROLES_METADATA_KEY: ["engineering"]}, ())
    assert ok is False
    assert "no granted roles" in reason


def test_wildcard_role_grants_access_to_any_tagged_doc():
    ok, _ = is_permitted({ROLES_METADATA_KEY: ["finance"]}, (WILDCARD_ROLE,))
    assert ok is True


def test_wildcard_role_still_fails_closed_on_missing_metadata():
    """Wildcard grants access to *tagged* docs — it does not bypass fail-closed
    behavior for documents with no ACL metadata at all."""
    ok, reason = is_permitted({}, (WILDCARD_ROLE,))
    assert ok is False


def test_string_allowed_roles_are_parsed():
    ok, _ = is_permitted({ROLES_METADATA_KEY: "engineering, finance"}, ("finance",))
    assert ok is True


# ── AccessControlledRetriever (integration through the pipeline) ──────────

def test_authorized_caller_sees_matching_doc():
    r = _make_retriever("test-access-control-auth")
    r.index(CHUNKS)
    result = r.retrieve("on-call rotations", caller_roles=("engineering",), top_k=5)
    assert isinstance(result, FilteredResult)
    assert any(res.source == "eng.md" for res in result.results)


def test_unauthorized_caller_does_not_see_doc():
    r = _make_retriever("test-access-control-unauth")
    r.index(CHUNKS)
    result = r.retrieve("quarterly budget projections", caller_roles=("engineering",), top_k=5)
    assert all(res.source != "finance.md" for res in result.results)


def test_untagged_document_is_never_returned_to_any_caller():
    """Fail closed at the integration level too: a document with no ACL
    metadata must not leak to a caller just because they hold some role."""
    r = _make_retriever("test-access-control-untagged")
    r.index(CHUNKS)
    result = r.retrieve("legacy document no ACL", caller_roles=("engineering", "finance"), top_k=5)
    assert all(res.source != "legacy.md" for res in result.results)


def test_caller_with_no_roles_gets_zero_results():
    r = _make_retriever("test-access-control-noroles")
    r.index(CHUNKS)
    result = r.retrieve("holiday calendar", caller_roles=(), top_k=5)
    assert result.results == []
    assert result.denied_count == len(result.decisions)


def test_multi_role_document_visible_to_either_role():
    r = _make_retriever("test-access-control-multirole")
    r.index(CHUNKS)
    eng_result = r.retrieve("holiday calendar", caller_roles=("engineering",), top_k=5)
    fin_result = r.retrieve("holiday calendar", caller_roles=("finance",), top_k=5)
    assert any(res.source == "holidays.md" for res in eng_result.results)
    assert any(res.source == "holidays.md" for res in fin_result.results)


def test_denied_count_reflects_filtered_chunks():
    r = _make_retriever("test-access-control-denied-count")
    r.index(CHUNKS)
    result = r.retrieve("budget engineering holiday legacy", caller_roles=("engineering",), top_k=10)
    assert result.denied_count >= 1
    assert all(res.source != "finance.md" for res in result.results)


def test_audit_log_records_every_retrieve_call():
    r = _make_retriever("test-access-control-audit")
    r.index(CHUNKS)
    assert r.audit_log == []
    r.retrieve("budget", caller_roles=("finance",), top_k=5)
    assert len(r.audit_log) == 1
    event = r.audit_log[0]
    assert event.caller_roles == ("finance",)
    assert event.total_candidates == event.permitted_count + event.denied_count


def test_audit_event_lists_denied_sources_for_defensibility():
    r = _make_retriever("test-access-control-audit-sources")
    r.index(CHUNKS)
    r.retrieve("budget engineering holiday legacy", caller_roles=("engineering",), top_k=10)
    event = r.audit_log[-1]
    assert "finance.md" in event.denied_sources
    assert "legacy.md" in event.denied_sources


def test_audit_can_be_disabled():
    store = VectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA, collection_name="test-access-control-noaudit"))
    embedder = Embedder(EmbeddingProvider.MOCK)
    base = VectorRetriever(store=store, embedder=embedder)
    r = AccessControlledRetriever(base, audit=False)
    r.index(CHUNKS)
    r.retrieve("budget", caller_roles=("finance",), top_k=5)
    assert r.audit_log == []


def test_filtered_result_context_and_repr():
    r = _make_retriever("test-access-control-context")
    r.index(CHUNKS)
    result = r.retrieve("on-call rotations", caller_roles=("engineering",), top_k=3)
    assert isinstance(result.context(n=2), str)
    assert "caller_roles" in repr(result)
