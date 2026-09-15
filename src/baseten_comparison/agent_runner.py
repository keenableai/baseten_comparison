"""Shared benchmark loop for standalone agent APIs (Exa Agent, Parallel Task). No Baseten."""

import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from baseten_comparison.benchmarks import Benchmark
from baseten_comparison.client import system_prompt
from baseten_comparison.harness import grade_answer, load_kept_rows


@dataclass
class AgentAnswer:
    predicted: str
    response: dict
    latency_s: float
    model_usd: float | None = None
    search_usd: float | None = None
    search_calls: dict[str, int] | None = None


AnswerFn = Callable[[str, str], AgentAnswer]


def run_agent_benchmark(
    benchmark: Benchmark,
    run_meta: dict,
    answer: AnswerFn,
    n: int,
    grader_model: str,
    output: str,
    seed: int,
    concurrency: int,
    resume: bool,
) -> None:
    system = system_prompt(benchmark.system_suffix)

    samples = benchmark.load(n, seed)
    current = {s["question"] for s in samples}
    done = load_kept_rows(output, run_meta) if resume else {}
    done = {q: r for q, r in done.items() if q in current}
    todo = [(i, s) for i, s in enumerate(samples) if s["question"] not in done]
    print(f"{len(done)} rows kept, {len(todo)} to run")

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
            a = answer(question, system)
            record.update(
                predicted=a.predicted, model_response=a.response, search_calls=a.search_calls
            )
            record["latency"]["answer_s"] = round(a.latency_s, 3)
            record["cost"].update(model_usd=a.model_usd, search_usd=a.search_usd)

            grade, grader_resp, grade_s = grade_answer(
                grader, grader_model, question, target, a.predicted
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
    counts = Counter(r["grade"] for r in records)
    total = len(records)
    for grade, count in sorted(counts.items()):
        print(f"  {grade}: {count} ({count / total:.0%})")
    correct = counts.get("CORRECT", 0)
    attempted = correct + counts.get("INCORRECT", 0)
    if attempted:
        print(f"  accuracy (attempted): {correct / attempted:.0%}")

    call_counts: Counter[str] = Counter()
    for r in records:
        call_counts.update(r["search_calls"] or {})
    if call_counts:
        print(f"  search calls: {dict(sorted(call_counts.items()))}")
    agent_usd = sum((r["cost"]["model_usd"] or 0) + (r["cost"]["search_usd"] or 0) for r in records)
    print(f"  agent cost: ${agent_usd:.4f}")
    grader_usd = [r["cost"]["grader_usd"] for r in records if r["cost"]["grader_usd"] is not None]
    if grader_usd:
        print(f"  grader cost: ${sum(grader_usd):.4f}")

    latencies = sorted(
        r["latency"]["answer_s"] for r in records if r["latency"]["answer_s"] is not None
    )
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p90 = latencies[int(len(latencies) * 0.9)]
        print(f"  answer latency: p50 {p50:.1f}s / p90 {p90:.1f}s / max {latencies[-1]:.1f}s")
