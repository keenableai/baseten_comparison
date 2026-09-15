"""Parallel Task API (api.parallel.ai/v1/tasks/runs) on the same benchmarks. No Baseten involved."""

import os
import sys
import time

import fire
import httpx
from dotenv import load_dotenv

from baseten_comparison.agent_runner import AgentAnswer, run_agent_benchmark
from baseten_comparison.benchmarks import BENCHMARKS
from baseten_comparison.harness import DEFAULT_GRADER_MODEL

PARALLEL_URL = "https://api.parallel.ai"
PROVIDER = "parallel-task"
# $/1k successful runs, docs.parallel.ai/getting-started/pricing, Sep 2026. Failed runs are free.
PROCESSOR_PRICES_PER_1K = {
    "lite": 5.0,
    "base": 10.0,
    "core": 25.0,
    "core2x": 50.0,
    "pro": 100.0,
    "ultra": 300.0,
}
# The result endpoint blocks, then returns 408 if the run is still active after `timeout` seconds.
RESULT_WAIT_S = 600
MAX_RUN_S = 3600.0


class ParallelTask:
    def __init__(self, api_key: str, processor: str) -> None:
        self.processor = processor
        self.http = httpx.Client(
            base_url=PARALLEL_URL,
            headers={"x-api-key": api_key},
            timeout=httpx.Timeout(60.0, read=RESULT_WAIT_S + 60.0),
        )

    def answer(self, question: str, system: str) -> AgentAnswer:
        start = time.perf_counter()
        run_id = self._create(question, system)["run_id"]
        result = self._result(run_id, start)
        return AgentAnswer(
            predicted=str(result["output"]["content"]).strip(),
            response=result,
            latency_s=time.perf_counter() - start,
            model_usd=PROCESSOR_PRICES_PER_1K[self.processor] / 1000,
        )

    def _create(self, question: str, system: str) -> dict:
        body = {
            "processor": self.processor,
            "input": question,
            # No system prompt in the Task API; the output description carries the instructions.
            "task_spec": {"output_schema": {"type": "text", "description": system}},
        }
        return self._request("POST", "/v1/tasks/runs", json=body, retry_on=(429,), attempts=12)

    def _result(self, run_id: str, start: float) -> dict:
        while True:
            remaining = MAX_RUN_S - (time.perf_counter() - start)
            if remaining <= 0:
                raise TimeoutError(f"run {run_id} still active after {MAX_RUN_S:.0f}s")
            try:
                resp = self.http.get(
                    f"/v1/tasks/runs/{run_id}/result",
                    params={"timeout": int(min(RESULT_WAIT_S, remaining))},
                )
            except httpx.TransportError:
                # Long-held connection dropped mid-wait; the run keeps going server-side.
                time.sleep(2.0)
                continue
            if resp.status_code == 408:
                continue
            if resp.status_code >= 500:
                time.sleep(2.0)
                continue
            if resp.status_code == 404:
                run = self.http.get(f"/v1/tasks/runs/{run_id}").json()
                raise RuntimeError(f"run {run_id} {run.get('status')}: {run.get('error')}")
            resp.raise_for_status()
            return resp.json()

    def _request(self, method: str, path: str, *, retry_on, attempts: int, **kwargs) -> dict:
        for attempt in range(attempts):
            resp = self.http.request(method, path, **kwargs)
            if resp.status_code not in retry_on or attempt == attempts - 1:
                break
            time.sleep(min(30.0, 2.0**attempt))
        resp.raise_for_status()
        return resp.json()


def run(
    benchmark: str,
    n: int = 10,
    processor: str = "base",
    grader_model: str = DEFAULT_GRADER_MODEL,
    output: str | None = None,
    seed: int = 0,
    concurrency: int = 4,
    resume: bool = False,
) -> None:
    load_dotenv()
    if benchmark not in BENCHMARKS:
        sys.exit(f"unknown benchmark {benchmark!r}; choose from {sorted(BENCHMARKS)}")
    if processor not in PROCESSOR_PRICES_PER_1K:
        sys.exit(f"unknown processor {processor!r}; choose from {sorted(PROCESSOR_PRICES_PER_1K)}")
    for var in ("PARALLEL_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"export {var} first")
    if n < 1 or concurrency < 1:
        sys.exit("n and concurrency must be >= 1")

    output = output or f"results/{benchmark}_{PROVIDER}_{processor}.jsonl"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    agent = ParallelTask(os.environ["PARALLEL_API_KEY"], processor)
    run_agent_benchmark(
        BENCHMARKS[benchmark],
        run_meta={
            "benchmark": benchmark,
            "provider": PROVIDER,
            "model": f"{PROVIDER}-{processor}",
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
