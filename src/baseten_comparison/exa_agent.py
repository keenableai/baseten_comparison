"""Exa Agent (api.exa.ai/agent/runs) on the same benchmarks. Standalone: no Baseten involved."""

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
from dotenv import load_dotenv

from baseten_comparison.benchmarks import BENCHMARKS, Benchmark
from baseten_comparison.client import system_prompt
from baseten_comparison.harness import DEFAULT_GRADER_MODEL, grade_answer, load_kept_rows

EXA_URL = "https://api.exa.ai"
PROVIDER = "exa-agent"
POLL_INTERVAL_S = 2.0


class ExaAgent:
    def __init__(self, api_key: str, effort: str) -> None:
        self.effort = effort
        self.http = httpx.Client(
            base_url=EXA_URL,
            headers={"x-api-key": api_key, "Exa-Beta": "agent-2026-05-07"},
            timeout=60.0,
        )

    def answer(self, question: str, system: str) -> tuple[str, dict, float]:
        start = time.perf_counter()
        run = self._create(question, system)
        while run["status"] in ("queued", "running"):
            time.sleep(POLL_INTERVAL_S)
            run = self._get(run["id"])
        latency = time.perf_counter() - start
        if run["status"] != "completed":
            raise RuntimeError(f"run {run['id']} {run['status']}: {run.get('error')}")
        predicted = ((run.get("output") or {}).get("text") or "").strip()
        return predicted, run, latency

    def _create(self, question: str, system: str) -> dict:
        body = {"query": question, "systemPrompt": system, "effort": self.effort}
        # 429 = CONCURRENCY_LIMIT_REACHED; slots only free up as runs finish, so wait and retry.
        for attempt in range(12):
            resp = self.http.post("/agent/runs", json=body)
            if resp.status_code != 429:
                break
            time.sleep(min(30.0, 2.0**attempt))
        resp.raise_for_status()
        return resp.json()

    def _get(self, run_id: str) -> dict:
        resp = self.http.get(f"/agent/runs/{run_id}")
        resp.raise_for_status()
        return resp.json()


def run_benchmark(
    benchmark: Benchmark,
    n: int,
    effort: str,
    grader_model: str,
    output: str,
    seed: int,
    concurrency: int,
    resume: bool,
) -> None:
    system = system_prompt(benchmark.system_suffix)
    model = f"{PROVIDER}-{effort}"
    run_meta = {
        "benchmark": benchmark.name,
        "provider": PROVIDER,
        "model": model,
        "grader_model": grader_model,
    }

    samples = benchmark.load(n, seed)
    done = load_kept_rows(output, run_meta) if resume else {}
    todo = [(i, s) for i, s in enumerate(samples) if s["question"] not in done]
    print(f"{len(done)} rows kept, {len(todo)} to run")

    agent = ExaAgent(os.environ["EXA_API_KEY"], effort)
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
            predicted, run, answer_s = agent.answer(question, system)
            cost = run["costDollars"]
            record.update(
                predicted=predicted,
                model_response=run,
                search_calls={"exa_agent_search": run["usage"]["searches"]},
            )
            record["latency"]["answer_s"] = round(answer_s, 3)
            record["cost"].update(model_usd=cost["agentCompute"], search_usd=cost["search"])

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
    counts = Counter(r["grade"] for r in records)
    total = len(records)
    for grade, count in sorted(counts.items()):
        print(f"  {grade}: {count} ({count / total:.0%})")
    correct = counts.get("CORRECT", 0)
    attempted = correct + counts.get("INCORRECT", 0)
    if attempted:
        print(f"  accuracy (attempted): {correct / attempted:.0%}")

    runs = [r["model_response"] for r in records if r["model_response"]]
    acu = sum(r["usage"]["agentComputeUnits"] for r in runs)
    searches = sum(r["usage"]["searches"] for r in runs)
    total_usd = sum(r["costDollars"]["total"] for r in runs)
    print(f"  agent compute units: {acu:.1f}, searches: {searches}")
    print(f"  agent cost: ${total_usd:.4f}")
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


def run(
    benchmark: str,
    n: int = 10,
    effort: str = "low",
    grader_model: str = DEFAULT_GRADER_MODEL,
    output: str | None = None,
    seed: int = 0,
    concurrency: int = 2,
    resume: bool = False,
) -> None:
    load_dotenv()
    if benchmark not in BENCHMARKS:
        sys.exit(f"unknown benchmark {benchmark!r}; choose from {sorted(BENCHMARKS)}")
    for var in ("EXA_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"export {var} first")
    if n < 1 or concurrency < 1:
        sys.exit("n and concurrency must be >= 1")

    run_benchmark(
        BENCHMARKS[benchmark],
        n=n,
        effort=effort,
        grader_model=grader_model,
        output=output or f"{benchmark}_{PROVIDER}_{effort}.jsonl",
        seed=seed,
        concurrency=concurrency,
        resume=resume,
    )


def main() -> None:
    fire.Fire(run)
