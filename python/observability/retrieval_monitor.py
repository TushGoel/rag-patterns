"""
Retrieval observability — stream every RAG query as a telemetry event.

At production scale, batch analytics are too slow to catch retrieval
quality degradation. Streaming every query enables:
  - Real-time relevance score monitoring
  - Latency SLO enforcement per retrieval strategy
  - Cost tracking per caller/use case
  - Provider health comparison (which vector store is slow?)

Designed to integrate with kafka-patterns LLMEventStream.
Works standalone (in-memory) when Kafka is not available.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from ..retrieval.pipeline import RetrievalResult


@dataclass
class RetrievalEvent:
    """One RAG query captured as a telemetry event."""
    event_id: str
    query: str
    strategy: str
    top_k: int
    result_count: int
    avg_score: float
    latency_ms: float
    caller_id: str
    timestamp_ms: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "query": self.query[:200],
            "strategy": self.strategy,
            "top_k": self.top_k,
            "result_count": self.result_count,
            "avg_score": self.avg_score,
            "latency_ms": self.latency_ms,
            "caller_id": self.caller_id,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class RetrievalAnomaly:
    kind: str        # "LATENCY_SLO", "LOW_RELEVANCE", "EMPTY_RESULTS"
    severity: str    # "WARN", "PAGE"
    strategy: str
    metric: float
    threshold: float
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.kind} {self.strategy}: {self.message}"


class RetrievalMonitor:
    """
    Real-time anomaly detection on RAG retrieval events.

    Mirrors the InferenceMonitor pattern from kafka-patterns —
    same philosophy applied to the retrieval layer instead of LLM calls.

    Usage:
        monitor = RetrievalMonitor(latency_slo_ms=500, min_relevance=0.4)

        result = retriever.retrieve(query)
        event = monitor.record(result, caller_id="chat-api")
        anomalies = monitor.check(event)
        for a in anomalies:
            if a.severity == "PAGE":
                alert_oncall(str(a))
    """

    def __init__(
        self,
        latency_slo_ms: float = 500.0,
        min_relevance_score: float = 0.3,
        window_size: int = 100,
    ) -> None:
        self.latency_slo_ms = latency_slo_ms
        self.min_relevance_score = min_relevance_score
        self._events: deque = deque(maxlen=window_size)
        self._anomalies: list[RetrievalAnomaly] = []

    def record(self, result: RetrievalResult, caller_id: str = "default") -> RetrievalEvent:
        avg_score = (
            sum(r.score for r in result.results) / len(result.results)
            if result.results else 0.0
        )
        event = RetrievalEvent(
            event_id=str(uuid.uuid4()),
            query=result.query,
            strategy=result.strategy,
            top_k=len(result.results),
            result_count=len(result.results),
            avg_score=avg_score,
            latency_ms=result.latency_ms,
            caller_id=caller_id,
            timestamp_ms=int(time.time() * 1000),
        )
        self._events.append(event)
        return event

    def check(self, event: RetrievalEvent) -> list[RetrievalAnomaly]:
        anomalies = []

        if event.latency_ms > self.latency_slo_ms:
            severity = "PAGE" if event.latency_ms > self.latency_slo_ms * 2 else "WARN"
            a = RetrievalAnomaly(
                kind="LATENCY_SLO",
                severity=severity,
                strategy=event.strategy,
                metric=event.latency_ms,
                threshold=self.latency_slo_ms,
                message=f"latency={event.latency_ms:.0f}ms exceeds SLO {self.latency_slo_ms:.0f}ms",
            )
            anomalies.append(a)
            self._anomalies.append(a)

        if event.result_count == 0:
            a = RetrievalAnomaly(
                kind="EMPTY_RESULTS",
                severity="PAGE",
                strategy=event.strategy,
                metric=0,
                threshold=1,
                message=f"zero results returned for query={event.query[:50]!r}",
            )
            anomalies.append(a)
            self._anomalies.append(a)

        elif event.avg_score < self.min_relevance_score:
            a = RetrievalAnomaly(
                kind="LOW_RELEVANCE",
                severity="WARN",
                strategy=event.strategy,
                metric=event.avg_score,
                threshold=self.min_relevance_score,
                message=f"avg_score={event.avg_score:.3f} below threshold {self.min_relevance_score}",
            )
            anomalies.append(a)
            self._anomalies.append(a)

        return anomalies

    def stats(self) -> dict:
        if not self._events:
            return {"total_queries": 0}
        events = list(self._events)
        return {
            "total_queries": len(events),
            "avg_latency_ms": round(sum(e.latency_ms for e in events) / len(events), 1),
            "avg_relevance_score": round(sum(e.avg_score for e in events) / len(events), 3),
            "total_anomalies": len(self._anomalies),
            "empty_result_count": sum(1 for e in events if e.result_count == 0),
        }
