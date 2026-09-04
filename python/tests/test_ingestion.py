"""Tests for multi-modal document ingestion."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from python.ingestion.document import (
    Document, SourceType, TextLoader, CodeLoader, load
)


def _write_temp(content: str, suffix: str = ".txt") -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.flush()
    return f.name


def test_text_loader_produces_document():
    path = _write_temp("Hello world. This is a test document.")
    doc = TextLoader().load(path)
    assert isinstance(doc, Document)
    assert doc.source_type == SourceType.TEXT
    assert "Hello world" in doc.content


def test_document_word_count():
    doc = Document(content="one two three four five", source="test", source_type=SourceType.TEXT)
    assert doc.word_count == 5


def test_document_char_count():
    doc = Document(content="hello", source="test", source_type=SourceType.TEXT)
    assert doc.char_count == 5


def test_code_loader_single_file():
    path = _write_temp("def hello():\n    return 'world'\n", suffix=".py")
    docs = CodeLoader().load(path)
    assert len(docs) == 1
    assert docs[0].source_type == SourceType.CODE
    assert docs[0].metadata["language"] == "python"


def test_code_loader_detects_language():
    loader = CodeLoader()
    assert loader._detect_language(".py") == "python"
    assert loader._detect_language(".go") == "go"
    assert loader._detect_language(".ts") == "typescript"
    assert loader._detect_language(".rs") == "rust"
    assert loader._detect_language(".java") == "java"


def test_code_loader_empty_file_skipped():
    path = _write_temp("", suffix=".py")
    docs = CodeLoader().load(path)
    assert len(docs) == 0


def test_load_auto_detects_text():
    path = _write_temp("Sample text content.", suffix=".md")
    docs = load(path)
    assert len(docs) >= 1
    assert docs[0].source_type in (SourceType.TEXT, SourceType.CODE)


def test_load_auto_detects_python():
    path = _write_temp("x = 1\ny = 2\n", suffix=".py")
    docs = load(path)
    assert len(docs) == 1
    assert docs[0].source_type == SourceType.CODE


def test_document_repr():
    doc = Document(content="test content", source="file.txt", source_type=SourceType.TEXT)
    assert "file.txt" in repr(doc)
    assert "TEXT" in repr(doc)


def test_text_loader_preserves_content():
    content = "Line one.\nLine two.\nLine three."
    path = _write_temp(content)
    doc = TextLoader().load(path)
    assert doc.content == content
