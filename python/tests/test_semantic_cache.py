"""
Tests for semantic caching.

MockEmbedder (used elsewhere in this repo) hashes exact text into
uncorrelated random vectors — perfect for offline determinism, but useless
for testing *semantic* similarity, since two different strings never
meaningfully overlap. These tests use a small ScriptedEmbedder that returns
hand-picked vectors instead, so similarity between "near duplicate" and
"different" queries is known and controllable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from python.retrieval.semantic_cache import SemanticCache, CacheResult


class ScriptedEmbedder:
    """Returns a pre-defined vector per query string instead of computing one."""

    def __init__(self, vectors: dict):
        self.vectors = vectors

    def embed(self, text: str):
        return self.vectors[text]


ORIGINAL = "How does chunking affect retrieval quality?"
PARAPHRASE = "What impact does chunk size have on retrieval quality?"
UNRELATED = "What's the best pizza topping?"

VECTORS = {
    ORIGINAL: [1.0, 0.0],
    PARAPHRASE: [0.97, 0.2],     # cosine(ORIGINAL, PARAPHRASE) ≈ 0.979 — near duplicate
    UNRELATED: [0.0, 1.0],        # cosine(ORIGINAL, UNRELATED) = 0.0 — genuinely different
}


def _make_cache(threshold=0.92, ttl=300.0, max_size=1000):
    return SemanticCache(embedder=ScriptedEmbedder(VECTORS), similarity_threshold=threshold, ttl_seconds=ttl, max_size=max_size)


# ── basic hit / miss ────────────────────────────────────────────────────────

def test_miss_on_empty_cache():
    cache = _make_cache()
    result = cache.get(ORIGINAL)
    assert isinstance(result, CacheResult)
    assert result.hit is False


def test_hit_on_exact_same_query():
    cache = _make_cache()
    cache.set(ORIGINAL, "cached answer")
    result = cache.get(ORIGINAL)
    assert result.hit is True
    assert result.value == "cached answer"
    assert result.similarity == pytest.approx(1.0)


def test_hit_on_near_duplicate_paraphrase():
    """A paraphrased query with the same intent should hit the cache even
    though the text differs — this is the whole point of semantic caching."""
    cache = _make_cache(threshold=0.92)
    cache.set(ORIGINAL, "cached answer")
    result = cache.get(PARAPHRASE)
    assert result.hit is True
    assert result.matched_query == ORIGINAL
    assert result.similarity >= 0.92


def test_miss_on_genuinely_different_query():
    cache = _make_cache(threshold=0.92)
    cache.set(ORIGINAL, "cached answer")
    result = cache.get(UNRELATED)
    assert result.hit is False
    assert result.similarity < 0.92


def test_threshold_controls_hit_sensitivity():
    """Raising the threshold above the paraphrase's similarity turns a hit into a miss."""
    cache = _make_cache(threshold=0.99)  # paraphrase similarity ≈ 0.979, below this
    cache.set(ORIGINAL, "cached answer")
    result = cache.get(PARAPHRASE)
    assert result.hit is False


# ── TTL / expiry ────────────────────────────────────────────────────────────

def test_entry_expires_after_ttl():
    cache = _make_cache(ttl=10.0)
    cache.set(ORIGINAL, "cached answer", now=1000.0)

    # Still within TTL
    result = cache.get(ORIGINAL, now=1005.0)
    assert result.hit is True

    # Past TTL
    result = cache.get(ORIGINAL, now=1011.0)
    assert result.hit is False


def test_expired_entries_are_purged_from_cache():
    cache = _make_cache(ttl=10.0)
    cache.set(ORIGINAL, "cached answer", now=1000.0)
    assert len(cache) == 1

    cache.get(ORIGINAL, now=1011.0)  # triggers eviction sweep, misses
    assert len(cache) == 0


# ── eviction / max_size ─────────────────────────────────────────────────────

def test_oldest_entry_evicted_when_over_max_size():
    vectors = {
        "q1": [1.0, 0.0, 0.0],
        "q2": [0.0, 1.0, 0.0],
        "q3": [0.0, 0.0, 1.0],
    }
    cache = SemanticCache(embedder=ScriptedEmbedder(vectors), similarity_threshold=0.99, ttl_seconds=300, max_size=2)
    cache.set("q1", "answer1", now=1.0)
    cache.set("q2", "answer2", now=2.0)
    cache.set("q3", "answer3", now=3.0)  # should evict q1 (oldest)

    assert len(cache) == 2
    assert cache.get("q1", now=4.0).hit is False
    assert cache.get("q3", now=4.0).hit is True


# ── config validation ───────────────────────────────────────────────────────

def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        SemanticCache(similarity_threshold=1.5)


def test_invalid_ttl_raises():
    with pytest.raises(ValueError):
        SemanticCache(ttl_seconds=0)


def test_invalid_max_size_raises():
    with pytest.raises(ValueError):
        SemanticCache(max_size=0)


# ── stats / clear ────────────────────────────────────────────────────────────

def test_stats_tracks_hits_and_misses():
    cache = _make_cache()
    cache.set(ORIGINAL, "cached answer")
    cache.get(ORIGINAL)     # hit
    cache.get(UNRELATED)    # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_clear_empties_cache():
    cache = _make_cache()
    cache.set(ORIGINAL, "cached answer")
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.get(ORIGINAL).hit is False
