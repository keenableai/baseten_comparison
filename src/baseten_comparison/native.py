"""Vendor-native web search + fetch: Anthropic and OpenAI APIs called directly, no Baseten."""

import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import fire
import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

from baseten_comparison.benchmarks import BENCHMARKS, Benchmark
from baseten_comparison.client import system_prompt
from baseten_comparison.harness import (
    DEFAULT_GRADER_MODEL,
    grade_answer,
    load_kept_rows,
    print_summary,
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


class AnthropicNative:
    def __init__(self, api_key: str, model: str, effort: str) -> None:
        self.model, self.effort = model, effort
        self.client = Anthropic(api_key=api_key, timeout=300.0, max_retries=8)

    def answer(self, question: str, system: str) -> tuple[str, dict, float, Counter[str]]:
        start = time.perf_counter()
        messages = [{"role": "user", "content": question}]
        usage = Counter()
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
            server = final.usage.server_tool_use
            usage.update(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                web_search=server.web_search_requests if server else 0,
                web_fetch=server.web_fetch_requests if server else 0,
            )
            if final.stop_reason != "pause_turn" or pause_turns == MAX_PAUSE_TURNS:
                break
            pause_turns += 1
            messages = [*messages, {"role": "assistant", "content": final.content}]
        latency = time.perf_counter() - start
        predicted = "\n".join(b.text for b in final.content if b.type == "text").strip()
        response = final.model_dump(mode="json", warnings=False)
        response["usage"].update(
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]
        )
        response["pause_turns"] = pause_turns
        calls = Counter(
            {
                "anthropic__web_search": usage["web_search"],
                "anthropic__web_fetch": usage["web_fetch"],
            }
        )
        return predicted, response, latency, calls


class OpenAINative:
    def __init__(self, api_key: str, model: str, effort: str) -> None:
        self.model, self.effort = model, effort
        self.http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=300.0)

    def answer(self, question: str, system: str) -> tuple[str, dict, float, Counter[str]]:
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
        return predicted, response, latency, calls


CLIENTS = {"anthropic": AnthropicNative, "openai": OpenAINative}


def run_benchmark(
    benchmark: Benchmark,
    n: int,
    provider: str,
    model: str,
    effort: str,
    grader_model: str,
    output: str,
    seed: int,
    concurrency: int,
    model_prices: tuple[float, float] | None,
    resume: bool,
) -> None:
    system = system_prompt(benchmark.system_suffix)
    run_meta = {
        "benchmark": benchmark.name,
        "provider": provider,
        "model": model,
        "effort": effort,
        "grader_model": grader_model,
    }

    samples = benchmark.load(n, seed)
    current = {s["question"] for s in samples}
    done = load_kept_rows(output, run_meta) if resume else {}
    done = {q: r for q, r in done.items() if q in current}
    todo = [(i, s) for i, s in enumerate(samples) if s["question"] not in done]
    print(f"{len(done)} rows kept, {len(todo)} to run")

    agent = CLIENTS[provider](os.environ[API_KEYS[provider]], model, effort)
    grader = httpx.Client(
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=180.0
    )
    write_lock = threading.Lock()

    def process(idx_sample: tuple[int, dict]) -> dict:
        idx, sample = idx_sample
        question, target = sample["question"], sample["target"]
        record = {
            "index": idx,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "question": question,
            "target": target,
            "metadata": sample["metadata"],
            **run_meta,
            "predicted": None,
            "grade": "ERROR",
            "model_response": None,
            "grader_response": None,
            "latency": {"answer_s": None, "grade_s": None, "total_s": None},
            "search_calls": None,
            "cost": {"model_usd": None, "search_usd": None, "grader_usd": None},
            "error": None,
        }
        total_start = time.perf_counter()
        try:
            predicted, model_resp, answer_s, calls = agent.answer(question, system)
            usage = model_resp.get("usage") or {}
            record.update(predicted=predicted, model_response=model_resp, search_calls=calls)
            record["latency"]["answer_s"] = round(answer_s, 3)
            record["cost"]["search_usd"] = search_cost(calls)
            if model_prices:
                in_price, out_price = model_prices
                record["cost"]["model_usd"] = (
                    usage.get("input_tokens", 0) * in_price
                    + usage.get("output_tokens", 0) * out_price
                ) / 1e6

            grade, grader_resp, grade_s = grade_answer(
                grader, grader_model, question, target, predicted
            )
            record.update(grade=grade, grader_response=grader_resp)
            record["latency"]["grade_s"] = round(grade_s, 3)
            record["cost"]["grader_usd"] = (grader_resp.get("usage") or {}).get("cost")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency"]["total_s"] = round(time.perf_counter() - total_start, 3)
        with write_lock:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{idx + 1}/{len(samples)}] {record['grade']} "
                f"({record['latency']['total_s']}s): {question[:80]}"
            )
        return record

    run_start = time.perf_counter()
    with open(output, "w", encoding="utf-8") as out:
        for rec in done.values():
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            records = [*done.values(), *pool.map(process, todo)]

    print(f"\n{len(records)} samples → {output} in {time.perf_counter() - run_start:.0f}s wall")
    print_summary(records)
    if benchmark.extra_summary:
        benchmark.extra_summary(records)


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
    model_prices = resolve_model_prices(model, model_input_price, model_output_price)
    if model_prices is None:
        print(f"WARNING: no price for {model}; model cost will be null")

    output = output or f"results/{benchmark}_{provider}_{model.replace('/', '_')}_{effort}.jsonl"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    run_benchmark(
        BENCHMARKS[benchmark],
        n=n,
        provider=provider,
        model=model,
        effort=effort,
        grader_model=grader_model,
        output=output,
        seed=seed,
        concurrency=concurrency,
        model_prices=model_prices,
        resume=resume,
    )


def main() -> None:
    fire.Fire(run)
