"""
Audio ingestion — speech-to-text transcription.

Transcribes audio files (MP3, WAV, M4A, FLAC, OGG) to text for indexing.
Uses OpenAI Whisper as the primary transcription engine — state-of-the-art
accuracy, multilingual support, runs locally without API calls.

Whisper models: tiny (39M), base (74M), small (244M), medium (769M), large (1.5B)
Default: "base" — good balance of speed and accuracy for most use cases.

Prerequisites:
    pip install openai-whisper
    # Whisper downloads model weights on first use (~74MB for base)

Supported formats: MP3, WAV, M4A, FLAC, OGG, WEBM
"""
from __future__ import annotations

from pathlib import Path
from .document import Document, SourceType

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


class AudioLoader:
    """
    Transcribe audio files to text using Whisper.

    Usage:
        loader = AudioLoader(model="base")
        doc = loader.load("meeting_recording.mp3")
        print(doc.content)   # full transcript
        print(doc.metadata)  # {"duration_s": 240.5, "language": "en", ...}
    """

    def __init__(self, model: str = "base", language: str = None) -> None:
        self.model_name = model
        self.language = language
        self._model = None

    def load(self, path: str) -> Document:
        path_obj = Path(path)
        if path_obj.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {path_obj.suffix}")

        result = self._transcribe(path)
        text = result.get("text", "").strip()
        language = result.get("language", self.language or "unknown")
        duration = self._estimate_duration(result)

        return Document(
            content=text,
            source=path,
            source_type=SourceType.AUDIO,
            metadata={
                "file_name": path_obj.name,
                "extension": path_obj.suffix.lower(),
                "whisper_model": self.model_name,
                "language": language,
                "duration_s": duration,
                "segment_count": len(result.get("segments", [])),
            },
        )

    def load_with_timestamps(self, path: str) -> list[Document]:
        """
        Load audio and return one Document per segment with timestamps.
        Useful for long recordings where timestamp context matters.
        """
        result = self._transcribe(path)
        segments = result.get("segments", [])

        if not segments:
            return [self.load(path)]

        docs = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            docs.append(Document(
                content=text,
                source=path,
                source_type=SourceType.AUDIO,
                metadata={
                    "start_s": round(seg.get("start", 0), 2),
                    "end_s": round(seg.get("end", 0), 2),
                    "segment_id": seg.get("id", 0),
                    "whisper_model": self.model_name,
                },
            ))
        return docs

    def _transcribe(self, path: str) -> dict:
        try:
            import whisper
        except ImportError:
            raise ImportError("pip install openai-whisper")

        if self._model is None:
            self._model = whisper.load_model(self.model_name)

        options = {}
        if self.language:
            options["language"] = self.language

        return self._model.transcribe(path, **options)

    def _estimate_duration(self, result: dict) -> float:
        segments = result.get("segments", [])
        if segments:
            return round(segments[-1].get("end", 0), 2)
        return 0.0
