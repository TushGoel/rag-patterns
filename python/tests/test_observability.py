"""Tests for retrieval observability and anomaly detection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from python.observability.retrieval_monitor import RetrievalMonitor, RetrievalAnomaly
from python.providers.vector_store import SearchResult
from python.retrieval.pipeline import RetrievalResult


def _make_result(query="test", strategy="vector", latency_ms=100.0, scores=None):
    scores = scores or [0.8, 0.7, 0.6]
    results = [
        SearchResult(text=f"doc {i}", score=s, source=f"doc_{i}.txt")
        for i, s in enumerate(scores)
    ]
    return RetrievalResult(
        query=query, results=results, strategy=strategy, latency_ms=latency_ms
    )


def test_monitor_records_event():
    monitor = RetrievalMonitor()
    result = _make_result()
    event = monitor.record(result, caller_id="test")
    assert event.query == "test"
    assert event.strategy == "vector"
    assert event.result_count == 3


def test_monitor_no_anomaly_within_slo():
    monitor = RetrievalMonitor(latency_slo_ms=500)
    result = _make_result(latency_ms=100)
    event = monitor.record(result)
    anomalies = monitor.check(event)
    assert not any(a.kind == "LATENCY_SLO" for a in anomalies)


def test_monitor_detects_latency_breach():
    monitor = RetrievalMonitor(latency_slo_ms=100)
    result = _make_result(latency_ms=300)
    event = monitor.record(result)
    anomalies = monitor.check(event)
    assert any(a.kind == "LATENCY_SLO" for a in anomalies)


def test_monitor_page_severity_extreme_latency():
    monitor = RetrievalMonitor(latency_slo_ms=100)
    result = _make_result(latency_ms=500)
    event = monitor.record(result)
    anomalies = monitor.check(event)
    pages = [a for a in anomalies if a.severity == "PAGE"]
    assert len(pages) > 0


def test_monitor_detects_empty_results():
    monitor = RetrievalMonitor()
    result = RetrievalResult(query="test", results=[], strategy="vector", latency_ms=50)
    event = monitor.record(result)
    anomalies = monitor.check(event)
    assert any(a.kind == "EMPTY_RESULTS" for a in anomalies)


def test_monitor_detects_low_relevance():
    monitor = RetrievalMonitor(min_relevance_score=0.5)
    result = _make_result(scores=[0.1, 0.2, 0.15])
    event = monitor.record(result)
    anomalies = monitor.check(event)
    assert any(a.kind == "LOW_RELEVANCE" for a in anomalies)


def test_monitor_stats():
    monitor = RetrievalMonitor()
    for _ in range(5):
        result = _make_result(latency_ms=100)
        event = monitor.record(result)
        monitor.check(event)
    stats = monitor.stats()
    assert stats["total_queries"] == 5
    assert "avg_latency_ms" in stats


def test_anomaly_str():
    a = RetrievalAnomaly(
        kind="LATENCY_SLO", severity="WARN", strategy="vector",
        metric=300, threshold=100, message="too slow"
    )
    assert "WARN" in str(a)
    assert "LATENCY_SLO" in str(a)
