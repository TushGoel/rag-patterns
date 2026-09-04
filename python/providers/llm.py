"""
Multi-model LLM provider abstraction.

Swap between OpenAI, Anthropic, AWS Bedrock, and Ollama (local)
via a single config change — no code changes downstream.

All providers return a uniform LLMResponse so retrieval, eval,
and observability work identically regardless of provider.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: Optional[str] = None
    region: Optional[str] = None       # Bedrock
    base_url: Optional[str] = None     # Ollama


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider:
    """
    Provider-agnostic LLM interface.

    Usage:
        config = LLMConfig(provider=Provider.ANTHROPIC, model="claude-3-5-sonnet-20241022")
        llm = LLMProvider(config)
        response = llm.complete("What is RAG?", context="RAG stands for...")
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(self, query: str, context: str = "", system: str = "") -> LLMResponse:
        prompt = self._build_prompt(query, context, system)
        start = time.time()

        if self.config.provider == Provider.OPENAI:
            return self._openai(prompt, start)
        elif self.config.provider == Provider.ANTHROPIC:
            return self._anthropic(prompt, start)
        elif self.config.provider == Provider.BEDROCK:
            return self._bedrock(prompt, start)
        elif self.config.provider == Provider.OLLAMA:
            return self._ollama(prompt, start)
        elif self.config.provider == Provider.GEMINI:
            return self._gemini(prompt, start)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _build_prompt(self, query: str, context: str, system: str) -> str:
        if context:
            return f"Context:\n{context}\n\nQuestion: {query}"
        return query

    def _openai(self, prompt: str, start: float) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = OpenAI(api_key=self.config.api_key)
        resp = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=self.config.model,
            provider=Provider.OPENAI,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=(time.time() - start) * 1000,
            cost_usd=self._openai_cost(resp.usage.prompt_tokens, resp.usage.completion_tokens),
        )

    def _anthropic(self, prompt: str, start: float) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.Anthropic(api_key=self.config.api_key)
        resp = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            content=resp.content[0].text if resp.content else "",
            model=self.config.model,
            provider=Provider.ANTHROPIC,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.time() - start) * 1000,
            cost_usd=self._anthropic_cost(resp.usage.input_tokens, resp.usage.output_tokens),
        )

    def _bedrock(self, prompt: str, start: float) -> LLMResponse:
        try:
            import boto3, json
        except ImportError:
            raise ImportError("pip install boto3")

        client = boto3.client("bedrock-runtime", region_name=self.config.region or "us-east-1")
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = client.invoke_model(modelId=self.config.model, body=body)
        result = json.loads(resp["body"].read())
        return LLMResponse(
            content=result["content"][0]["text"] if result.get("content") else "",
            model=self.config.model,
            provider=Provider.BEDROCK,
            input_tokens=result.get("usage", {}).get("input_tokens", 0),
            output_tokens=result.get("usage", {}).get("output_tokens", 0),
            latency_ms=(time.time() - start) * 1000,
        )

    def _ollama(self, prompt: str, start: float) -> LLMResponse:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        base = self.config.base_url or "http://localhost:11434"
        resp = requests.post(
            f"{base}/api/generate",
            json={"model": self.config.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            content=data.get("response", ""),
            model=self.config.model,
            provider=Provider.OLLAMA,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=(time.time() - start) * 1000,
        )

    def _gemini(self, prompt: str, start: float) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        genai.configure(api_key=self.config.api_key)
        model = genai.GenerativeModel(self.config.model or "gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else ""
        usage = resp.usage_metadata if hasattr(resp, "usage_metadata") else None
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return LLMResponse(
            content=text,
            model=self.config.model or "gemini-1.5-flash",
            provider=Provider.GEMINI,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            cost_usd=self._gemini_cost(input_tokens, output_tokens),
        )

    def _openai_cost(self, input_tokens: int, output_tokens: int) -> float:
        # gpt-4o pricing as reference
        return (input_tokens * 2.5 + output_tokens * 10) / 1_000_000

    def _anthropic_cost(self, input_tokens: int, output_tokens: int) -> float:
        # claude-3-5-sonnet pricing as reference
        return (input_tokens * 3.0 + output_tokens * 15) / 1_000_000

    def _gemini_cost(self, input_tokens: int, output_tokens: int) -> float:
        # gemini-1.5-flash pricing as reference
        return (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000
