"""
Agentic RAG — self-correcting retrieval loop.

Standard RAG retrieves once and generates. If the context is insufficient,
the answer is wrong or incomplete with no recovery path.

Agentic RAG adds a feedback loop:
  1. Retrieve context
  2. Ask: "Is this context sufficient to answer the question?"
  3. If yes → generate answer
  4. If no → reformulate query and retrieve again (max 3 iterations)

This mirrors how a human researcher works — if the first search doesn't
yield enough, they refine the search before writing.

When to use: Questions requiring comprehensive answers, technical deep-dives,
anything where a single retrieval might miss key context.
Not ideal for: Simple factual lookups where one retrieval is always enough.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..providers.llm import LLMProvider, LLMResponse
from ..chunking.strategies import Chunk
from .pipeline import RetrievalResult, VectorRetriever


@dataclass
class AgentStep:
    iteration: int
    query: str
    context_sufficient: bool
    retrieval: RetrievalResult
    reformulation: Optional[str] = None


@dataclass
class AgenticRAGResult:
    original_query: str
    final_answer: str
    steps: list[AgentStep]
    total_iterations: int
    converged: bool
    total_latency_ms: float
    metadata: dict = field(default_factory=dict)

    @property
    def strategy(self) -> str:
        return "agentic"

    def __repr__(self) -> str:
        return (
            f"AgenticRAGResult(query={self.original_query!r}, "
            f"iterations={self.total_iterations}, "
            f"converged={self.converged}, "
            f"answer_chars={len(self.final_answer)})"
        )


SUFFICIENCY_PROMPT = (
    "You are evaluating whether retrieved context is sufficient to answer a question.\n\n"
    "Question: {query}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Is this context sufficient to provide a complete, accurate answer? "
    "Answer only YES or NO, then on the next line if NO explain what is missing "
    "and suggest a better search query (2-8 words)."
)

GENERATION_PROMPT = (
    "Answer the following question using only the provided context. "
    "Be specific and accurate. If the context doesn't fully answer the question, say so.\n\n"
    "Question: {query}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)

REFORMULATE_PROMPT = (
    "The current search query did not return sufficient context to answer the question.\n\n"
    "Original question: {query}\n"
    "Current query: {current_query}\n"
    "What is missing: {missing}\n\n"
    "Write a better search query (2-8 words) that would find the missing information:"
)


class AgenticRetriever:
    """
    Self-correcting RAG with iterative query reformulation.

    Retrieves → evaluates sufficiency → reformulates if needed → repeats.
    Stops when context is sufficient or max_iterations is reached.

    Usage:
        agent = AgenticRetriever(llm=llm, retriever=retriever, max_iterations=3)
        agent.index(chunks)
        result = agent.run("Explain the tradeoffs between BM25 and dense retrieval")
        print(result.final_answer)
        print(f"Converged in {result.total_iterations} iterations")
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: VectorRetriever,
        max_iterations: int = 3,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.top_k = top_k

    def index(self, chunks: list[Chunk]) -> None:
        self.retriever.index(chunks)

    def run(self, query: str) -> AgenticRAGResult:
        import time
        start = time.time()

        steps: list[AgentStep] = []
        current_query = query
        converged = False
        final_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            retrieval = self.retriever.retrieve(current_query, top_k=self.top_k)
            context = retrieval.context(n=self.top_k)

            sufficient, missing = self._check_sufficiency(query, context)

            step = AgentStep(
                iteration=iteration,
                query=current_query,
                context_sufficient=sufficient,
                retrieval=retrieval,
            )

            if sufficient or iteration == self.max_iterations:
                final_answer = self._generate(query, context)
                converged = sufficient
                steps.append(step)
                break

            new_query = self._reformulate(query, current_query, missing)
            step.reformulation = new_query
            steps.append(step)
            current_query = new_query

        return AgenticRAGResult(
            original_query=query,
            final_answer=final_answer,
            steps=steps,
            total_iterations=len(steps),
            converged=converged,
            total_latency_ms=(time.time() - start) * 1000,
            metadata={"max_iterations": self.max_iterations},
        )

    def _check_sufficiency(self, query: str, context: str) -> tuple[bool, str]:
        prompt = SUFFICIENCY_PROMPT.format(query=query, context=context[:3000])
        response = self.llm.complete(prompt)
        content = response.content.strip()
        lines = content.splitlines()
        verdict = lines[0].strip().upper() if lines else "NO"
        missing = lines[1].strip() if len(lines) > 1 else ""
        return verdict.startswith("YES"), missing

    def _generate(self, query: str, context: str) -> str:
        prompt = GENERATION_PROMPT.format(query=query, context=context)
        response = self.llm.complete(prompt)
        return response.content.strip()

    def _reformulate(self, original: str, current: str, missing: str) -> str:
        prompt = REFORMULATE_PROMPT.format(
            query=original, current_query=current, missing=missing
        )
        response = self.llm.complete(prompt)
        return response.content.strip()
