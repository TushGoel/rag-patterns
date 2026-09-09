"""
Guardrails — input and output validation for RAG pipelines.

RAG failures aren't always "wrong answer" — two failure modes are easy to
miss because they don't look like errors at all:

  - Input: a crafted query tries to override system instructions (prompt
    injection), or is simply too long to process safely or cheaply.
  - Output: the response contains PII pulled from retrieved context, or the
    model silently refused/deflected instead of answering — which looks
    identical to a successful call unless something is watching for it.

These checks run outside the LLM call — before a query becomes a prompt, and
after a response comes back — so they catch what the model itself has no
incentive (or ability) to flag on its own.

Deliberately regex/heuristic-based, not a full NER or classifier pipeline —
good enough to catch common shapes offline, with no model calls and no
external services. Swap in a real PII/injection classifier behind the same
`InputGuardrail` / `OutputGuardrail` interface for production-grade coverage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InputVerdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class OutputVerdict(str, Enum):
    ALLOW = "allow"       # clean, nothing flagged
    REDACTED = "redacted"  # PII found and stripped; response still returned
    FLAGGED = "flagged"    # refusal/deflection detected — surface it, don't return silently


@dataclass
class InputCheckResult:
    verdict: InputVerdict
    reasons: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.verdict == InputVerdict.ALLOW

    def __repr__(self) -> str:
        return f"InputCheckResult(verdict={self.verdict.value}, reasons={self.reasons})"


@dataclass
class OutputCheckResult:
    verdict: OutputVerdict
    text: str                 # sanitized text — safe to return to the caller
    original_text: str
    pii_found: list[str] = field(default_factory=list)
    refusal_detected: bool = False
    reasons: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"OutputCheckResult(verdict={self.verdict.value}, "
            f"pii_found={self.pii_found}, refusal_detected={self.refusal_detected})"
        )


# Prompt injection heuristics — common phrasing used to try to override
# system instructions or extract hidden context.
INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) instructions",
    r"disregard (all|any|the)? ?(previous|prior|above) (instructions|rules|prompt)",
    r"you are now (in )?(developer|admin|god|dan) mode",
    r"reveal (your|the) system prompt",
    r"forget (everything|all) (you('ve| have) been told|prior context)",
    r"act as (if you (are|were)|an?) (unrestricted|jailbroken|uncensored)",
    r"pretend (you have|to have) no (restrictions|rules|filters)",
    r"\bdan\b.{0,20}\bmode\b",
]

# PII heuristics, checked (and redacted) in this order so longer matches
# (credit card) are consumed before shorter overlapping ones (phone).
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){15,16}\b|\b(?:\d{4}[ -]?){3}\d{1,4}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}

# Refusal/deflection phrasing — catches silent failures where the model
# declined to answer but nothing downstream would otherwise notice.
REFUSAL_PATTERNS = [
    r"i('m| am) sorry,? but i (can('t|not)|won't)",
    r"i can('t|not) (help|assist) with that",
    r"as an ai( language model)?,? i (can('t|not)|don't)",
    r"i('m| am) not able to (provide|answer|help)",
    r"i don't have (access to|enough) (information|context) to answer",
]


class InputGuardrail:
    """
    Validate a query before it reaches retrieval/generation.

    Checks:
      - max length (cheap cost/DoS control)
      - prompt injection heuristics (regex over common override phrasing)

    Usage:
        guardrail = InputGuardrail(max_length=2000)
        result = guardrail.check(user_query)
        if not result.allowed:
            return refuse(result.reasons)
    """

    def __init__(
        self,
        max_length: int = 4000,
        injection_patterns: Optional[list[str]] = None,
    ) -> None:
        self.max_length = max_length
        self.injection_patterns = injection_patterns or INJECTION_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.injection_patterns]

    def check(self, query: str) -> InputCheckResult:
        if not query or not query.strip():
            return InputCheckResult(verdict=InputVerdict.BLOCK, reasons=["empty query"])

        reasons = []
        matched = []

        if len(query) > self.max_length:
            reasons.append(f"query length {len(query)} exceeds max_length {self.max_length}")

        for pattern, compiled in zip(self.injection_patterns, self._compiled):
            if compiled.search(query):
                matched.append(pattern)

        if matched:
            reasons.append(f"matched {len(matched)} prompt-injection pattern(s)")

        verdict = InputVerdict.BLOCK if reasons else InputVerdict.ALLOW
        return InputCheckResult(verdict=verdict, reasons=reasons, matched_patterns=matched)


class OutputGuardrail:
    """
    Validate and sanitize a response before it reaches the caller.

    Checks:
      - PII detection/redaction (email, SSN, credit card, phone). The
        response is still returned, with matches replaced by a
        `[REDACTED_<TYPE>]` tag.
      - refusal-pattern detection. Flags responses that silently declined to
        answer, so an unhelpful "successful" call doesn't look identical to
        a genuinely good one in logs or eval.

    Usage:
        guardrail = OutputGuardrail()
        result = guardrail.check(llm_response.content)
        print(result.text)               # sanitized text, safe to return
        if result.refusal_detected:
            log_silent_failure(result)
    """

    def __init__(
        self,
        pii_patterns: Optional[dict] = None,
        refusal_patterns: Optional[list[str]] = None,
    ) -> None:
        self.pii_patterns = pii_patterns or PII_PATTERNS
        self.refusal_patterns = refusal_patterns or REFUSAL_PATTERNS
        self._compiled_pii = {name: re.compile(p) for name, p in self.pii_patterns.items()}
        self._compiled_refusal = [re.compile(p, re.IGNORECASE) for p in self.refusal_patterns]

    def check(self, text: str) -> OutputCheckResult:
        pii_found = []
        redacted = text

        for name, compiled in self._compiled_pii.items():
            if compiled.search(redacted):
                pii_found.append(name)
                redacted = compiled.sub(f"[REDACTED_{name.upper()}]", redacted)

        refusal_detected = any(compiled.search(text) for compiled in self._compiled_refusal)

        reasons = []
        if pii_found:
            reasons.append(f"redacted {len(pii_found)} PII type(s): {', '.join(pii_found)}")
        if refusal_detected:
            reasons.append("response matches a refusal/deflection pattern")

        # Refusal takes priority: a flagged silent failure is more actionable
        # than "we also redacted some PII" — both get surfaced in `reasons`.
        if refusal_detected:
            verdict = OutputVerdict.FLAGGED
        elif pii_found:
            verdict = OutputVerdict.REDACTED
        else:
            verdict = OutputVerdict.ALLOW

        return OutputCheckResult(
            verdict=verdict,
            text=redacted,
            original_text=text,
            pii_found=pii_found,
            refusal_detected=refusal_detected,
            reasons=reasons,
        )


class Guardrails:
    """
    Convenience wrapper bundling input + output validation for one RAG call.

    Usage:
        guardrails = Guardrails(max_input_length=2000)

        input_result = guardrails.check_input(query)
        if not input_result.allowed:
            return refuse(input_result.reasons)

        response = llm.complete(query, context=context)
        output_result = guardrails.check_output(response.content)
        if output_result.verdict == OutputVerdict.FLAGGED:
            log_silent_failure(output_result)
        return output_result.text
    """

    def __init__(
        self,
        max_input_length: int = 4000,
        injection_patterns: Optional[list[str]] = None,
        pii_patterns: Optional[dict] = None,
        refusal_patterns: Optional[list[str]] = None,
    ) -> None:
        self.input_guardrail = InputGuardrail(max_length=max_input_length, injection_patterns=injection_patterns)
        self.output_guardrail = OutputGuardrail(pii_patterns=pii_patterns, refusal_patterns=refusal_patterns)

    def check_input(self, query: str) -> InputCheckResult:
        return self.input_guardrail.check(query)

    def check_output(self, text: str) -> OutputCheckResult:
        return self.output_guardrail.check(text)
