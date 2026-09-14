# Contributing

Thanks for looking at `rag-patterns`. This repo collects production-shaped RAG
patterns — ingestion, chunking, retrieval, providers, eval, and observability —
each one self-contained and testable without API keys. Contributions that add
a new pattern, harden an existing one, or improve test coverage are welcome.

## Dev environment setup

```bash
git clone https://github.com/TushGoel/rag-patterns.git
cd rag-patterns
python3 -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

`requirements.txt` installs everything needed to run every pattern with the
`MOCK` or `LOCAL` embedding provider and the `CHROMA` vector store backend —
no API keys required. LLM provider SDKs (`openai`, `anthropic`, `boto3`) are
commented out in `requirements.txt` since most patterns are tested against a
mocked `LLMProvider` — uncomment only the one(s) you need if you're wiring up
a real provider.

## Running tests

Tests must pass with no external services and no API keys — this is what
keeps CI fast and deterministic. Use `EmbeddingProvider.MOCK` and
`VectorBackend.CHROMA` (ephemeral, in-memory) in tests, and mock `LLMProvider`
calls with `unittest.mock.MagicMock` (see `python/tests/test_staff_patterns.py`
for the pattern).

Run the full suite exactly as CI does (`.github/workflows/ci.yml`):

```bash
pytest python/tests/ -v --tb=short
```

Run a single file or test while iterating:

```bash
pytest python/tests/test_retrieval.py -v
pytest python/tests/test_retrieval.py::test_vector_retriever_returns_results -v
```

All 147 existing tests must stay green. If your change touches a shared
module (`providers/`, `chunking/strategies.py`), run the full suite before
opening a PR — several other patterns import from those modules.

## Code style

Match the conventions already used throughout `python/`:

- **`from __future__ import annotations`** at the top of every module, and
  built-in generics for type hints (`list[str]`, `dict`, `Optional[int]`) —
  not `typing.List` / `typing.Dict`.
- **Dataclasses** for result/event/config objects (`@dataclass`, with
  `field(default_factory=dict)` for mutable defaults). Give result objects a
  `__repr__` or `__str__` when printing them is part of the usage flow.
- **`str, Enum` subclasses** for provider/backend/strategy switches (see
  `EmbeddingProvider`, `VectorBackend`, `Provider`).
- **Docstrings on every public class**: a one-line summary, then a short
  "when to use this vs. the alternative" note, then a `Usage:` block with a
  runnable example. Look at `HybridRetriever` or `RetrievalMonitor` for the
  shape to follow.
- **Fail loud on missing optional dependencies** — wrap the import in
  `try/except ImportError` and raise `ImportError("pip install <package>")`
  rather than letting an `ModuleNotFoundError` propagate unexplained.
- **No bare `print` in library code** except in explicitly report-style
  methods (e.g. `EvalPipeline.summary`) — keep the rest side-effect free and
  return data structures instead.
- Keep new dependencies optional wherever feasible: import them lazily inside
  the function/method that needs them (as `pipeline.py` does with
  `rank_bm25` and `sentence_transformers`), not at module load time, so the
  rest of the repo keeps working if that one extra isn't installed.

## Adding a new pattern

New patterns follow the existing module layout — one focused file per
pattern, plus a matching test file. To add one:

1. **Pick the right home.** `ingestion/` for loading a new source type,
   `chunking/` for a new splitting strategy, `retrieval/` for a new
   retrieval/composition strategy, `providers/` for a new embedding/LLM/
   vector-store backend, `eval/` for a new quality metric, `observability/`
   for a new telemetry/monitoring concern. If none fit, propose a new
   top-level `python/<area>/` directory in your PR description before
   writing code.
2. **One new file, named for the pattern** — e.g.
   `python/retrieval/my_pattern.py`, not an addition bolted onto an
   unrelated existing file.
3. **Compose with existing abstractions.** New retrieval patterns should
   accept a `VectorStore` / `Embedder` (or wrap an existing retriever, the
   way `RerankedRetriever` wraps `HybridRetriever`) rather than reinventing
   embedding or storage. New eval/observability patterns should operate on
   `RetrievalResult` / `LLMResponse` so they work with any retriever or
   provider.
4. **Add `python/tests/test_<pattern>.py`** with the same test-file header
   used everywhere else:

   ```python
   import sys, os
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
   ```

   Cover the happy path, at least one edge case (empty input, missing
   metadata, disabled/unavailable optional dependency), and any explicitly
   security- or correctness-relevant default (e.g. fail-closed behavior).
   Use `EmbeddingProvider.MOCK` and an in-memory Chroma collection so tests
   run offline and deterministically.
5. **Document it in `README.md`** as a new numbered pattern, following the
   existing structure: a short code example, then a **Problem → Solution →
   Impact**-style callout or explanatory paragraph, consistent with how the
   other nine patterns are written. If your pattern is a new top-level
   capability (not a Staff-level addition), also update the `Project
   Structure` tree and the `mermaid` System Design diagram if it changes the
   flow.
6. Run `pytest python/tests/ -v --tb=short` and confirm the full suite
   (existing + new tests) passes before opening a PR.

## Pull requests

- Keep PRs scoped to one pattern or one fix — easier to review, easier to
  revert if something's wrong.
- Describe the tradeoff your pattern makes (every pattern in this repo has
  one — see "Design Decisions & Trade-offs" in `README.md` for the existing
  examples) so reviewers understand *when* to reach for it, not just *that*
  it exists.
- No secrets, API keys, or real production data in code, tests, or fixtures
  — everything here must run with mock providers and local data only.
