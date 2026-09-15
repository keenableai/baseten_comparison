"""Exa Agent (api.exa.ai/agent/runs) on the same benchmarks. Standalone: no Baseten involved."""

import os
import sys
import time

import fire
import httpx
from dotenv import load_dotenv

from baseten_comparison.agent_runner import AgentAnswer, run_agent_benchmark
from baseten_comparison.benchmarks import BENCHMARKS
from baseten_comparison.harness import DEFAULT_GRADER_MODEL

EXA_URL = "https://api.exa.ai"
PROVIDER = "exa-agent"
# Fixed-price tiers only; auto/max are metered up to $5/$20 per request.
EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
POLL_INTERVAL_S = 2.0
MAX_RUN_S = 900.0


class ExaAgent:
    def __init__(self, api_key: str, effort: str) -> None:
        self.effort = effort
        self.http = httpx.Client(
            base_url=EXA_URL,
            headers={"x-api-key": api_key, "Exa-Beta": "agent-2026-05-07"},
            timeout=60.0,
        )

    def answer(self, question: str, system: str) -> AgentAnswer:
        start = time.perf_counter()
        run = self._create(question, system)
        while run["status"] in ("queued", "running"):
            if time.perf_counter() - start > MAX_RUN_S:
                raise TimeoutError(f"run {run['id']} still {run['status']} after {MAX_RUN_S:.0f}s")
            time.sleep(POLL_INTERVAL_S)
            run = self._get(run["id"])
        latency = time.perf_counter() - start
        if run["status"] != "completed":
            raise RuntimeError(f"run {run['id']} {run['status']}: {run.get('error')}")
        cost = run["costDollars"]
        return AgentAnswer(
            predicted=((run.get("output") or {}).get("text") or "").strip(),
            response=run,
            latency_s=latency,
            model_usd=cost["agentCompute"],
            search_usd=cost["search"],
            search_calls={"exa_agent_search": run["usage"]["searches"]},
        )

    def _create(self, question: str, system: str) -> dict:
        body = {"query": question, "systemPrompt": system, "effort": self.effort}
        # 429 = CONCURRENCY_LIMIT_REACHED; slots only free up as runs finish, so wait and retry.
        return self._request("POST", "/agent/runs", json=body, retry_on=(429,), attempts=12)

    def _get(self, run_id: str) -> dict:
        # Polling hits transient 500s; the run keeps going server-side, so just retry.
        return self._request("GET", f"/agent/runs/{run_id}", retry_on=range(500, 600), attempts=6)

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
    if effort not in EFFORTS:
        sys.exit(f"unknown effort {effort!r}; choose from {EFFORTS}")
    for var in ("EXA_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"export {var} first")
    if n < 1 or concurrency < 1:
        sys.exit("n and concurrency must be >= 1")

    output = output or f"results/{benchmark}_{PROVIDER}_{effort}.jsonl"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    agent = ExaAgent(os.environ["EXA_API_KEY"], effort)
    run_agent_benchmark(
        BENCHMARKS[benchmark],
        run_meta={
            "benchmark": benchmark,
            "provider": PROVIDER,
            "model": f"{PROVIDER}-{effort}",
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
