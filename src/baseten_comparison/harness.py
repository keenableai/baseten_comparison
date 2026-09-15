import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

import httpx
from anthropic import Anthropic

from baseten_comparison.benchmarks import Benchmark

DEFAULT_BASE_URL = "https://inference.baseten.co"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_GRADER_MODEL = "openai/gpt-5.5"

SERVER_TOOLS_BY_PROVIDER = {
    "exa": ["baseten__exa__web_search_exa", "baseten__exa__web_fetch_exa"],
    "keenable": ["baseten__keenable__search_web_pages", "baseten__keenable__fetch_page_content"],
    "parallel": ["baseten__parallel__web_search"],
    "youcom": ["baseten__youcom__you-search", "baseten__youcom__you-contents"],
}
SERVER_TOOLS_BY_PROVIDER["all"] = [t for ts in SERVER_TOOLS_BY_PROVIDER.values() for t in ts]

SYSTEM = (
    "You are a precise assistant with web search. "
    f"Today is {date.today().isoformat()}. "
    "Search the web to verify facts before answering. "
    "Answer in terse TL;DR style: lead with the answer, no filler."
)

GRADER_TEMPLATE = """\
Your job is to grade a predicted answer to a question against the gold target.

Grade the predicted answer as one of:
A: CORRECT — fully contains the gold target's important information, no contradictions. \
Hedging is fine if the correct answer is clearly stated. Minor wording/order/capitalization \
differences and semantically equivalent numbers (with tolerance for rounding) are fine.
B: INCORRECT — contains any factual statement contradicting the gold target, even if hedged.
C: NOT_ATTEMPTED — does not give the gold target, but also does not contradict it \
(e.g. declines, says it cannot find the answer).

Question: {question}
Gold target: {target}
Predicted answer: {predicted}

Reply with exactly one letter: A, B, or C. No other text.
"""

GRADE_NAMES = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
ERROR_GRADES = ("ERROR", "GRADER_ERROR")

MODEL_PRICES = {
    "deepseek-ai/DeepSeek-V4-Pro": (1.74, 3.48),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "moonshotai/Kimi-K3": (3.00, 15.00),
    "moonshotai/Kimi-K2.6": (0.95, 4.00),
    "zai-org/GLM-5.2": (1.40, 4.40),
    "zai-org/GLM-5.2-Fast": (2.10, 6.60),
    "zai-org/GLM-5.3-Fast": (2.10, 6.60),
    "openai/gpt-oss-120b": (0.10, 0.50),
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": (0.60, 2.40),
}

SEARCH_PRICES_PER_1K = {
    "baseten__keenable__search_web_pages": 4.00,
    "baseten__keenable__fetch_page_content": 4.00,
    "baseten__exa__web_search_exa": 7.00,
    "baseten__exa__web_fetch_exa": 1.00,
    "baseten__parallel__web_search": 5.00,
    "baseten__youcom__you-search": 5.00,
    "baseten__youcom__you-contents": 1.00,
}


def count_search_calls(content: list[dict]) -> Counter[str]:
    return Counter(
        b.get("name") or "unknown" for b in content if b.get("type", "").endswith("tool_use")
    )


def search_cost(calls: dict[str, int]) -> float:
    return sum(n * SEARCH_PRICES_PER_1K.get(name, 0.0) / 1000 for name, n in calls.items())


def resolve_model_prices(
    model: str, input_override: float | None, output_override: float | None
) -> tuple[float, float] | None:
    table_in, table_out = MODEL_PRICES.get(model, (None, None))
    in_price = table_in if input_override is None else input_override
    out_price = table_out if output_override is None else output_override
    if in_price is None or out_price is None:
        return None
    return in_price, out_price


def make_client(api_key: str) -> Anthropic:
    return Anthropic(
        api_key=api_key,
        base_url=os.environ.get("BASETEN_BASE_URL", DEFAULT_BASE_URL),
        default_headers={"Authorization": f"Bearer {api_key}", "x-baseten-server-tools": "true"},
        timeout=180.0,
        max_retries=8,
    )


def answer_question(
    client: Anthropic, model: str, server_tools: list[str], system: str, question: str
) -> tuple[str, dict, float]:
    start = time.perf_counter()
    final = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        tools=[{"type": t} for t in server_tools],
        extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        messages=[{"role": "user", "content": question}],
    )
    latency = time.perf_counter() - start
    predicted = "\n".join(b.text for b in final.content if b.type == "text").strip()
    return predicted, final.model_dump(mode="json", warnings=False), latency


def parse_grade(content: str) -> str:
    return GRADE_NAMES.get(content.strip().rstrip("."), "GRADER_ERROR")


def grade_answer(
    grader: httpx.Client, grader_model: str, question: str, target: str, predicted: str
) -> tuple[str, dict, float]:
    start = time.perf_counter()
    resp = grader.post(
        OPENROUTER_URL,
        json={
            "model": grader_model,
            "messages": [
                {
                    "role": "user",
                    "content": GRADER_TEMPLATE.format(
                        question=question, target=target, predicted=predicted
                    ),
                }
            ],
            "max_tokens": 2048,
            "temperature": 0,
            "usage": {"include": True},
        },
    )
    latency = time.perf_counter() - start
    resp.raise_for_status()
    body = resp.json()
    grade = parse_grade(body["choices"][0]["message"].get("content") or "")
    return grade, body, latency


def load_kept_rows(path: str, run_meta: dict) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    kept: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping unreadable line {lineno} in {path}")
                continue
            if r.get("grade") not in ERROR_GRADES and all(
                r.get(k) == v for k, v in run_meta.items()
            ):
                kept[r["question"]] = r
    return kept


def run_benchmark(
    benchmark: Benchmark,
    n: int,
    provider: str,
    model: str,
    grader_model: str,
    output: str,
    seed: int,
    concurrency: int,
    model_prices: tuple[float, float] | None,
    resume: bool,
) -> None:
    server_tools = SERVER_TOOLS_BY_PROVIDER[provider]
    system = SYSTEM + benchmark.system_suffix
    run_meta = {
        "benchmark": benchmark.name,
        "provider": provider,
        "model": model,
        "grader_model": grader_model,
    }

    samples = benchmark.load(n, seed)
    done = load_kept_rows(output, run_meta) if resume else {}
    todo = [(i, s) for i, s in enumerate(samples) if s["question"] not in done]
    print(f"{len(done)} rows kept, {len(todo)} to run")

    client = make_client(os.environ["BASETEN_API_KEY"])
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
            predicted, model_resp, answer_s = answer_question(
                client, model, server_tools, system, question
            )
            usage = model_resp.get("usage") or {}
            calls = count_search_calls(model_resp.get("content") or [])
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


def print_summary(records: list[dict]) -> None:
    grades = [r["grade"] for r in records]
    total = len(grades)
    counts = Counter(grades)
    for grade, count in sorted(counts.items()):
        print(f"  {grade}: {count} ({count / total:.0%})")
    correct = counts.get("CORRECT", 0)
    attempted = correct + counts.get("INCORRECT", 0)
    if attempted:
        print(f"  accuracy (attempted): {correct / attempted:.0%}")

    usages = [(r["model_response"] or {}).get("usage") or {} for r in records]
    in_tok = sum(u.get("input_tokens", 0) for u in usages)
    out_tok = sum(u.get("output_tokens", 0) for u in usages)
    print(f"  model tokens: {in_tok} in / {out_tok} out")
    for key, label in (("model_usd", "model"), ("grader_usd", "grader")):
        costs = [r["cost"][key] for r in records if r["cost"][key] is not None]
        if costs:
            print(f"  {label} cost: ${sum(costs):.4f}")

    call_counts: Counter[str] = Counter()
    for r in records:
        call_counts.update(r["search_calls"] or {})
    if call_counts:
        breakdown = ", ".join(
            f"{name.split('__')[-1]}: {n}" for name, n in sorted(call_counts.items())
        )
        print(f"  search calls: {sum(call_counts.values())} ({breakdown})")
        print(f"  search cost: ${search_cost(call_counts):.4f}")
        unpriced = sorted(set(call_counts) - set(SEARCH_PRICES_PER_1K))
        if unpriced:
            print(f"  WARNING: no price for tools: {unpriced}")

    latencies = sorted(
        r["latency"]["answer_s"] for r in records if r["latency"]["answer_s"] is not None
    )
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p90 = latencies[int(len(latencies) * 0.9)]
        print(f"  answer latency: p50 {p50:.1f}s / p90 {p90:.1f}s / max {latencies[-1]:.1f}s")
