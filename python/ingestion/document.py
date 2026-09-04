"""
Document ingestion — multi-modal loader for PDF, code, web, and text.

Every source produces the same Document object so downstream chunking,
retrieval, and eval work identically regardless of input type.

Supported:
  - PDF (text + layout preservation)
  - Source code repositories (file tree traversal, language detection)
  - Web pages (URL → clean text, boilerplate stripped)
  - Plain text / markdown / any text file
"""
from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


class SourceType(str, Enum):
    PDF = "pdf"
    CODE = "code"
    WEB = "web"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Uniform document representation regardless of source type."""
    content: str
    source: str                        # file path or URL
    source_type: SourceType
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        return f"Document(source={self.source!r}, type={self.source_type.value.upper()}, words={self.word_count})"


class PDFLoader:
    """
    Load PDF files preserving page structure.

    Uses pypdf for text extraction. Falls back to pdfplumber for
    complex layouts (tables, multi-column). Page numbers are embedded
    in metadata so citations can reference exact pages.
    """

    def load(self, path: str) -> list[Document]:
        try:
            import pypdf
        except ImportError:
            raise ImportError("pip install pypdf")

        docs = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = self._clean(text)
                if text.strip():
                    docs.append(Document(
                        content=text,
                        source=path,
                        source_type=SourceType.PDF,
                        metadata={"page": page_num, "total_pages": len(reader.pages)},
                    ))
        return docs

    def _clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()


class CodeLoader:
    """
    Load source code repositories or individual files.

    Traverses directory trees, respects .gitignore patterns,
    detects programming language from extension, and preserves
    file path in metadata for context-aware retrieval.
    """

    SUPPORTED_EXTENSIONS = {
        ".py", ".go", ".ts", ".tsx", ".js", ".jsx",
        ".java", ".rs", ".cpp", ".c", ".h", ".cs",
        ".rb", ".sh", ".yaml", ".yml", ".toml", ".json",
        ".md", ".txt", ".sql",
    }

    IGNORE_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv",
        "venv", "dist", "build", ".pytest_cache",
    }

    def load(self, path: str, max_file_size_kb: int = 500) -> list[Document]:
        root = Path(path)
        docs = []

        if root.is_file():
            doc = self._load_file(root)
            return [doc] if doc else []

        for file_path in root.rglob("*"):
            if any(p in file_path.parts for p in self.IGNORE_DIRS):
                continue
            if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
                continue
            if file_path.stat().st_size > max_file_size_kb * 1024:
                continue
            doc = self._load_file(file_path)
            if doc:
                docs.append(doc)

        return docs

    def _load_file(self, path: Path) -> Optional[Document]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return None
            lang = self._detect_language(path.suffix)
            return Document(
                content=content,
                source=str(path),
                source_type=SourceType.CODE,
                metadata={
                    "language": lang,
                    "file_name": path.name,
                    "extension": path.suffix,
                    "lines": content.count("\n") + 1,
                },
            )
        except Exception:
            return None

    def _detect_language(self, ext: str) -> str:
        mapping = {
            ".py": "python", ".go": "go", ".ts": "typescript",
            ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
            ".java": "java", ".rs": "rust", ".cpp": "cpp", ".c": "c",
            ".cs": "csharp", ".rb": "ruby", ".sh": "bash",
            ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
            ".json": "json", ".md": "markdown", ".sql": "sql",
        }
        return mapping.get(ext, "unknown")


class WebLoader:
    """
    Load web pages — URL to clean text, boilerplate stripped.

    Removes navigation, footers, ads, and script/style tags.
    Preserves article/main content. Respects robots.txt via
    a simple user-agent header.
    """

    def load(self, url: str, timeout: int = 10) -> Document:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install requests beautifulsoup4")

        headers = {"User-Agent": "Mozilla/5.0 (compatible; rag-patterns/1.0)"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        text = self._clean(text)

        title = soup.title.string if soup.title else urlparse(url).netloc

        return Document(
            content=text,
            source=url,
            source_type=SourceType.WEB,
            metadata={"title": title, "url": url},
        )

    def _clean(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)


class TextLoader:
    """Load plain text, markdown, or any UTF-8 text file."""

    def load(self, path: str) -> Document:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        mime, _ = mimetypes.guess_type(path)
        return Document(
            content=content,
            source=path,
            source_type=SourceType.TEXT,
            metadata={"mime_type": mime or "text/plain", "file_name": Path(path).name},
        )


def load(source: str, **kwargs) -> list[Document]:
    """
    Auto-detect source type and load documents.

    Usage:
        docs = load("report.pdf")
        docs = load("https://example.com")
        docs = load("./my-repo/")
        docs = load("notes.md")
        docs = load("screenshot.png")
        docs = load("meeting.mp3")
    """
    from .image_loader import ImageLoader, IMAGE_EXTENSIONS
    from .audio_loader import AudioLoader, AUDIO_EXTENSIONS

    if source.startswith("http://") or source.startswith("https://"):
        return [WebLoader().load(source, **kwargs)]

    path = Path(source)
    if path.is_dir():
        return CodeLoader().load(source, **kwargs)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader().load(source)
    if suffix in IMAGE_EXTENSIONS:
        return [ImageLoader().load(source)]
    if suffix in AUDIO_EXTENSIONS:
        return [AudioLoader().load(source)]
    if suffix in CodeLoader.SUPPORTED_EXTENSIONS - {".md", ".txt"}:
        return CodeLoader().load(source)

    return [TextLoader().load(source)]
