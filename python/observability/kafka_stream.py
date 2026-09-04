"""
Kafka-backed retrieval event streaming.

Publishes every RAG query as a Kafka event — enabling real-time
retrieval quality monitoring, SLO enforcement, and per-caller analytics.

Integrates with kafka-patterns:
  - Uses ReliableProducer (acks=all, idempotent delivery)
  - Schema matches kafka-patterns LLMInvocationEvent pattern
  - Downstream consumers can use kafka-patterns InferenceMonitor logic

Falls back to in-memory (RetrievalMonitor) when Kafka is unavailable.
This makes the pattern testable without a running Kafka broker.

Usage with Kafka:
    stream = RetrievalEventStream(bootstrap_servers="localhost:9092")
    result = retriever.retrieve(query)
    stream.publish(result, caller_id="chat-api")

Usage without Kafka (in-memory fallback):
    stream = RetrievalEventStream()  # no bootstrap_servers
    stream.publish(result)
    print(stream.stats())
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .retrieval_monitor import RetrievalEvent, RetrievalMonitor
from ..retrieval.pipeline import RetrievalResult

RETRIEVAL_TOPIC = "rag-retrieval-events"


@dataclass
class KafkaConfig:
    bootstrap_servers: str
    topic: str = RETRIEVAL_TOPIC
    acks: str = "all"
    idempotence: bool = True


class RetrievalEventStream:
    """
    Stream RAG retrieval events to Kafka with in-memory fallback.

    Every query → published as a structured event with:
      - query (truncated for privacy)
      - retrieval strategy used
      - result count and avg relevance score
      - latency
      - caller ID for per-service tracking

    Compatible with kafka-patterns consumer patterns for downstream
    SLO monitoring and alerting.

    Usage:
        # With Kafka
        stream = RetrievalEventStream(bootstrap_servers="localhost:9092")

        # Without Kafka (tests, local dev)
        stream = RetrievalEventStream()

        result = retriever.retrieve(query)
        stream.publish(result, caller_id="api")
        print(stream.stats())
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: str = RETRIEVAL_TOPIC,
        latency_slo_ms: float = 500.0,
        min_relevance_score: float = 0.3,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer = None
        self._monitor = RetrievalMonitor(
            latency_slo_ms=latency_slo_ms,
            min_relevance_score=min_relevance_score,
        )
        self._published: list[RetrievalEvent] = []
        self._use_kafka = bootstrap_servers is not None

        if self._use_kafka:
            self._init_producer()

    def _init_producer(self) -> None:
        try:
            from confluent_kafka import Producer
            self._producer = Producer({
                "bootstrap.servers": self.bootstrap_servers,
                "acks": "all",
                "enable.idempotence": True,
                "retries": 5,
            })
        except ImportError:
            self._use_kafka = False

    def publish(self, result: RetrievalResult, caller_id: str = "default") -> bool:
        event = self._monitor.record(result, caller_id=caller_id)
        self._published.append(event)

        anomalies = self._monitor.check(event)
        for a in anomalies:
            pass

        if self._use_kafka and self._producer:
            return self._publish_kafka(event)

        return True

    def _publish_kafka(self, event: RetrievalEvent) -> bool:
        try:
            payload = json.dumps(event.to_dict()).encode("utf-8")
            self._producer.produce(
                topic=self.topic,
                key=event.caller_id.encode("utf-8"),
                value=payload,
            )
            self._producer.poll(0)
            return True
        except Exception:
            return False

    def flush(self, timeout: float = 5.0) -> None:
        if self._use_kafka and self._producer:
            self._producer.flush(timeout=timeout)

    def stats(self) -> dict:
        base = self._monitor.stats()
        base["kafka_enabled"] = self._use_kafka
        base["topic"] = self.topic
        return base

    def anomalies(self) -> list:
        return self._monitor._anomalies
