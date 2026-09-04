"""
Tests for new features: image/audio ingestion, Gemini, merger, dashboard, Kafka stream.
All tests use mocks — no API keys, no Kafka broker, no OCR/audio deps needed.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from unittest.mock import MagicMock, patch
from python.ingestion.document import SourceType
from python.ingestion.image_loader import ImageLoader, IMAGE_EXTENSIONS
from python.ingestion.audio_loader import AudioLoader, AUDIO_EXTENSIONS
from python.providers.llm import LLMResponse, Provider, LLMConfig, LLMProvider
from python.query.merger import ConcatMerger, SummaryMerger, StructuredMerger, MergedAnswer
from python.eval.dashboard import EvalDashboard, DashboardConfig
from python.eval.metrics import EvalResult, FaithfulnessScore, RelevanceScore
from python.observability.kafka_stream import RetrievalEventStream
from python.retrieval.pipeline import RetrievalResult
from python.providers.vector_store import SearchResult


def _mock_llm(text="Mock response."):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        content=text, model="mock", provider="mock",
        input_tokens=10, output_tokens=5, latency_ms=5.0, cost_usd=0.001,
    )
    return llm


def _eval_result(query="test query", faith=0.9, rel=0.8):
    return EvalResult(
        query=query,
        answer="The answer is X.",
        faithfulness=FaithfulnessScore.from_score(faith, int(faith * 5), 5),
        relevance=RelevanceScore.from_score(rel, int(rel * 5), 5),
        retrieval_latency_ms=120.0,
        generation_latency_ms=0.0,
        total_tokens=100,
        cost_usd=0.002,
    )


def _make_retrieval(query="test"):
    results = [SearchResult(text=f"doc {i}", score=0.8-i*0.1, source=f"doc_{i}.txt") for i in range(3)]
    return RetrievalResult(query=query, results=results, strategy="vector", latency_ms=50.0)


# ── Image Loader ──────────────────────────────────────────────────────────────

def test_image_loader_extensions():
    assert ".png" in IMAGE_EXTENSIONS
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".tiff" in IMAGE_EXTENSIONS


def test_image_loader_rejects_non_image():
    import pytest
    with pytest.raises(ValueError, match="Unsupported image format"):
        ImageLoader().load("document.pdf")


def test_image_loader_ocr_called():
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = "extracted text from image"
    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = MagicMock()

    with patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "PIL": mock_pil, "PIL.Image": mock_pil.Image}):
        loader = ImageLoader(engine="tesseract")
        doc = loader.load("test.png")
        assert doc.source_type == SourceType.IMAGE
        assert doc.metadata["ocr_engine"] == "tesseract"


def test_image_loader_missing_deps():
    import pytest
    with patch.dict("sys.modules", {"pytesseract": None, "PIL": None}):
        with pytest.raises(Exception):
            ImageLoader()._tesseract("test.png")


# ── Audio Loader ──────────────────────────────────────────────────────────────

def test_audio_loader_extensions():
    assert ".mp3" in AUDIO_EXTENSIONS
    assert ".wav" in AUDIO_EXTENSIONS
    assert ".m4a" in AUDIO_EXTENSIONS


def test_audio_loader_rejects_non_audio():
    import pytest
    with pytest.raises(ValueError, match="Unsupported audio format"):
        AudioLoader().load("document.pdf")


def test_audio_loader_transcribes():
    mock_result = {
        "text": "This is the transcribed audio content.",
        "language": "en",
        "segments": [{"start": 0.0, "end": 5.2, "text": "This is the transcribed audio content.", "id": 0}],
    }
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_result
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"whisper": mock_whisper}):
        loader = AudioLoader(model="base")
        doc = loader.load("meeting.mp3")
        assert doc.source_type == SourceType.AUDIO
        assert "transcribed" in doc.content
        assert doc.metadata["language"] == "en"
        assert doc.metadata["duration_s"] == 5.2


def test_audio_loader_with_timestamps():
    mock_result = {
        "text": "Full transcript.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "First segment.", "id": 0},
            {"start": 3.0, "end": 6.0, "text": "Second segment.", "id": 1},
        ],
    }
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_result
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"whisper": mock_whisper}):
        loader = AudioLoader()
        docs = loader.load_with_timestamps("audio.wav")
        assert len(docs) == 2
        assert docs[0].metadata["start_s"] == 0.0
        assert docs[1].metadata["start_s"] == 3.0


# ── Gemini Provider ───────────────────────────────────────────────────────────

def test_gemini_provider_in_enum():
    assert Provider.GEMINI == "gemini"


def test_gemini_llm_config():
    config = LLMConfig(provider=Provider.GEMINI, model="gemini-1.5-flash", api_key="test-key")
    assert config.provider == Provider.GEMINI
    assert config.model == "gemini-1.5-flash"


def test_gemini_cost_calculation():
    llm = LLMProvider(LLMConfig(provider=Provider.GEMINI, model="gemini-1.5-flash"))
    cost = llm._gemini_cost(1000, 500)
    assert cost > 0
    assert cost < 0.01


# ── Query Merger ──────────────────────────────────────────────────────────────

def test_concat_merger():
    merger = ConcatMerger()
    result = merger.merge(
        query="Compare A and B",
        sub_queries=["What is A?", "What is B?"],
        sub_answers=["A is X.", "B is Y."],
    )
    assert isinstance(result, MergedAnswer)
    assert result.strategy == "concat"
    assert "A is X." in result.final_answer
    assert "B is Y." in result.final_answer


def test_summary_merger():
    llm = _mock_llm("A and B are different because X.")
    merger = SummaryMerger(llm=llm)
    result = merger.merge(
        query="Compare A and B",
        sub_queries=["What is A?", "What is B?"],
        sub_answers=["A is X.", "B is Y."],
    )
    assert result.strategy == "summary"
    assert result.final_answer == "A and B are different because X."
    assert llm.complete.called


def test_structured_merger():
    llm = _mock_llm("1. Summary\n2. Details\n3. Key points")
    merger = StructuredMerger(llm=llm)
    result = merger.merge(
        query="Explain retrieval",
        sub_queries=["What is vector search?"],
        sub_answers=["Vector search uses embeddings."],
    )
    assert result.strategy == "structured"
    assert len(result.final_answer) > 0


def test_merger_preserves_sub_queries():
    merger = ConcatMerger()
    sqs = ["Q1", "Q2", "Q3"]
    result = merger.merge("complex query", sqs, ["A1", "A2", "A3"])
    assert result.sub_queries == sqs
    assert result.original_query == "complex query"


# ── Eval Dashboard ────────────────────────────────────────────────────────────

def test_dashboard_add_result():
    dashboard = EvalDashboard()
    dashboard.add(_eval_result())
    assert len(dashboard._results) == 1


def test_dashboard_add_results_batch():
    dashboard = EvalDashboard()
    dashboard.add_results([_eval_result(f"q{i}") for i in range(5)])
    assert len(dashboard._results) == 5


def test_dashboard_summary_stats():
    dashboard = EvalDashboard()
    dashboard.add(_eval_result(faith=0.9, rel=0.8))
    dashboard.add(_eval_result(faith=0.7, rel=0.6))
    stats = dashboard._summary_stats()
    assert stats["total"] == 2
    assert 0.7 <= stats["avg_faithfulness"] <= 0.9
    assert "hallucinated_count" in stats


def test_dashboard_saves_html():
    dashboard = EvalDashboard(config=DashboardConfig(title="Test Report"))
    dashboard.add(_eval_result())
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = dashboard.save(f.name)
    import os
    assert os.path.exists(path)
    content = open(path).read()
    assert "Test Report" in content
    assert "chart.js" in content.lower()
    assert "faithfulness" in content.lower() or "Faithfulness" in content


def test_dashboard_html_has_table():
    dashboard = EvalDashboard()
    dashboard.add(_eval_result(query="What is RAG?"))
    html = dashboard._render()
    assert "<table" in html
    assert "What is RAG?" in html


# ── Kafka Stream ──────────────────────────────────────────────────────────────

def test_kafka_stream_inmemory_fallback():
    stream = RetrievalEventStream()
    assert not stream._use_kafka
    result = _make_retrieval()
    success = stream.publish(result, caller_id="test")
    assert success


def test_kafka_stream_records_events():
    stream = RetrievalEventStream()
    for i in range(3):
        stream.publish(_make_retrieval(f"query {i}"), caller_id="api")
    stats = stream.stats()
    assert stats["total_queries"] == 3


def test_kafka_stream_stats():
    stream = RetrievalEventStream(latency_slo_ms=1000)
    stream.publish(_make_retrieval())
    stats = stream.stats()
    assert "avg_latency_ms" in stats
    assert "kafka_enabled" in stats
    assert stats["kafka_enabled"] is False


def test_kafka_stream_detects_anomaly():
    stream = RetrievalEventStream(latency_slo_ms=10)
    slow_result = RetrievalResult(
        query="slow query",
        results=[SearchResult(text="doc", score=0.8, source="x.txt")],
        strategy="vector",
        latency_ms=5000.0,
    )
    stream.publish(slow_result)
    assert len(stream.anomalies()) > 0


def test_kafka_stream_topic_default():
    stream = RetrievalEventStream()
    assert stream.topic == "rag-retrieval-events"
