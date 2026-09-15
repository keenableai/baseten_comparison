"""Vendor-native web search + fetch: Anthropic and OpenAI APIs called directly, no Baseten."""

import os
import sys
import time
from collections import Counter

import fire
import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

from baseten_comparison.agent_runner import AgentAnswer, run_agent_benchmark
from baseten_comparison.benchmarks import BENCHMARKS
from baseten_comparison.harness import (
    CACHE_PRICES,
    DEFAULT_GRADER_MODEL,
    resolve_model_prices,
    search_cost,
)

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5", "openai": "gpt-5.6-terra"}
API_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
EFFORTS = ("low", "medium", "high")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
MAX_OUTPUT_TOKENS = 16384
# Long server-tool loops stop with pause_turn; resending the turn continues them.
MAX_PAUSE_TURNS = 5
ANTHROPIC_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def model_cost(model: str, prices: tuple[float, float] | None, tokens: dict) -> float | None:
    """tokens: uncached input, cache_read, cache_write, output. Cache rates fall back to input."""
    if prices is None:
        return None
    in_price, out_price = prices
    read_price, write_price = CACHE_PRICES.get(model, (in_price, in_price))
    return (
        tokens["input"] * in_price
        + tokens["cache_read"] * read_price
        + tokens["cache_write"] * write_price
        + tokens["output"] * out_price
    ) / 1e6


def anthropic_tokens(usage: dict) -> dict:
    # input_tokens excludes cache reads and writes; they are reported separately.
    return {
        "input": usage["input_tokens"],
        "cache_read": usage.get("cache_read_input_tokens") or 0,
        "cache_write": usage.get("cache_creation_input_tokens") or 0,
        "output": usage["output_tokens"],
    }


def openai_tokens(usage: dict) -> dict:
    # input_tokens includes cache reads and writes; split them out.
    details = usage.get("input_tokens_details") or {}
    cache_read = details.get("cached_tokens", 0)
    cache_write = details.get("cache_write_tokens", 0)
    return {
        "input": usage["input_tokens"] - cache_read - cache_write,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "output": usage["output_tokens"],
    }


class AnthropicNative:
    def __init__(self, api_key: str, model: str, effort: str, prices) -> None:
        self.model, self.effort, self.prices = model, effort, prices
        self.client = Anthropic(api_key=api_key, timeout=300.0, max_retries=8)

    def answer(self, question: str, system: str) -> AgentAnswer:
        start = time.perf_counter()
        messages = [{"role": "user", "content": question}]
        usage: Counter[str] = Counter()
        pause_turns = 0
        while True:
            final = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                tools=[
                    {"type": "web_search_20260318", "name": "web_search"},
                    {"type": "web_fetch_20260318", "name": "web_fetch"},
                ],
                messages=messages,
            )
            u = final.usage.model_dump(mode="json", warnings=False)
            usage.update({k: u.get(k) or 0 for k in ANTHROPIC_USAGE_KEYS})
            server = u.get("server_tool_use") or {}
            usage.update(
                web_search=server.get("web_search_requests", 0),
                web_fetch=server.get("web_fetch_requests", 0),
            )
            if final.stop_reason != "pause_turn" or pause_turns == MAX_PAUSE_TURNS:
                break
            pause_turns += 1
            messages = [*messages, {"role": "assistant", "content": final.content}]
        latency = time.perf_counter() - start
        response = final.model_dump(mode="json", warnings=False)
        response["usage"].update({k: usage[k] for k in ANTHROPIC_USAGE_KEYS})
        response["pause_turns"] = pause_turns
        calls = {
            "anthropic__web_search": usage["web_search"],
            "anthropic__web_fetch": usage["web_fetch"],
        }
        return AgentAnswer(
            predicted="\n".join(b.text for b in final.content if b.type == "text").strip(),
            response=response,
            latency_s=latency,
            model_usd=model_cost(self.model, self.prices, anthropic_tokens(response["usage"])),
            search_usd=search_cost(calls),
            search_calls=calls,
        )


class OpenAINative:
    def __init__(self, api_key: str, model: str, effort: str, prices) -> None:
        self.model, self.effort, self.prices = model, effort, prices
        self.http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=300.0)

    def answer(self, question: str, system: str) -> AgentAnswer:
        body = {
            "model": self.model,
            "reasoning": {"effort": self.effort},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "instructions": system,
            "input": question,
            # OpenAI's web_search tool also opens pages (open_page / find_in_page actions).
            "tools": [{"type": "web_search"}],
        }
        start = time.perf_counter()
        for attempt in range(8):
            resp = self.http.post(OPENAI_RESPONSES_URL, json=body)
            if (resp.status_code != 429 and resp.status_code < 500) or attempt == 7:
                break
            time.sleep(min(30.0, 2.0**attempt))
        latency = time.perf_counter() - start
        resp.raise_for_status()
        response = resp.json()
        if response.get("status") != "completed":
            raise RuntimeError(
                f"response {response.get('id')} {response.get('status')}: "
                f"{response.get('error') or response.get('incomplete_details')}"
            )
        output = response.get("output") or []
        predicted = "\n".join(
            part.get("text") or ""
            for item in output
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        ).strip()
        calls = Counter(
            f"openai__{(item.get('action') or {}).get('type') or 'unknown'}"
            for item in output
            if item.get("type") == "web_search_call"
        )
        return AgentAnswer(
            predicted=predicted,
            response=response,
            latency_s=latency,
            model_usd=model_cost(self.model, self.prices, openai_tokens(response["usage"])),
            search_usd=search_cost(calls),
            search_calls=dict(calls),
        )


CLIENTS = {"anthropic": AnthropicNative, "openai": OpenAINative}


def run(
    benchmark: str,
    provider: str,
    n: int = 10,
    model: str | None = None,
    effort: str = "medium",
    grader_model: str = DEFAULT_GRADER_MODEL,
    output: str | None = None,
    seed: int = 0,
    concurrency: int = 4,
    model_input_price: float | None = None,
    model_output_price: float | None = None,
    resume: bool = False,
) -> None:
    load_dotenv()
    if benchmark not in BENCHMARKS:
        sys.exit(f"unknown benchmark {benchmark!r}; choose from {sorted(BENCHMARKS)}")
    if provider not in CLIENTS:
        sys.exit(f"unknown provider {provider!r}; choose from {sorted(CLIENTS)}")
    if effort not in EFFORTS:
        sys.exit(f"unknown effort {effort!r}; choose from {EFFORTS}")
    for var in (API_KEYS[provider], "OPENROUTER_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"export {var} first")
    if n < 1 or concurrency < 1:
        sys.exit("n and concurrency must be >= 1")

    model = model or DEFAULT_MODELS[provider]
    prices = resolve_model_prices(model, model_input_price, model_output_price)
    if prices is None:
        print(f"WARNING: no price for {model}; model cost will be null")

    output = output or f"results/{benchmark}_{provider}_{model.replace('/', '_')}_{effort}.jsonl"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    agent = CLIENTS[provider](os.environ[API_KEYS[provider]], model, effort, prices)
    run_agent_benchmark(
        BENCHMARKS[benchmark],
        run_meta={
            "benchmark": benchmark,
            "provider": provider,
            "model": model,
            "effort": effort,
            "grader_model": grader_model,
        },
        answer=agent.answer,
        n=n,
        grader_model=grader_model,
        output=output,
        seed=seed,
        concurrency=concurrency,
        resume=resume,
    )


def main() -> None:
    fire.Fire(run)
