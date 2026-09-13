import os
import sys

import fire
from dotenv import load_dotenv

from baseten_comparison.benchmarks import BENCHMARKS
from baseten_comparison.harness import (
    DEFAULT_GRADER_MODEL,
    DEFAULT_MODEL,
    MODEL_PRICES,
    SERVER_TOOLS_BY_PROVIDER,
    run_benchmark,
)


def run(
    benchmark: str,
    n: int = 10,
    provider: str = "keenable",
    model: str | None = None,
    grader_model: str = DEFAULT_GRADER_MODEL,
    output: str | None = None,
    seed: int = 0,
    concurrency: int = 2,
    model_input_price: float | None = None,
    model_output_price: float | None = None,
    resume: bool = False,
) -> None:
    load_dotenv()
    if benchmark not in BENCHMARKS:
        sys.exit(f"unknown benchmark {benchmark!r}; choose from {sorted(BENCHMARKS)}")
    if provider != "all" and provider not in SERVER_TOOLS_BY_PROVIDER:
        sys.exit(f"unknown provider {provider!r}; choose from {[*SERVER_TOOLS_BY_PROVIDER, 'all']}")
    for var in ("BASETEN_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"export {var} first")

    if n < 1 or concurrency < 1:
        sys.exit("n and concurrency must be >= 1")

    model = model or os.environ.get("BASETEN_MODEL") or DEFAULT_MODEL
    table_in, table_out = MODEL_PRICES.get(model, (None, None))
    in_price = model_input_price if model_input_price is not None else table_in
    out_price = model_output_price if model_output_price is not None else table_out
    model_prices = (in_price, out_price) if in_price is not None and out_price is not None else None
    output = output or f"{benchmark}_{provider}_{model.replace('/', '_')}.jsonl"

    run_benchmark(
        BENCHMARKS[benchmark],
        n=n,
        provider=provider,
        model=model,
        grader_model=grader_model,
        output=output,
        seed=seed,
        concurrency=concurrency,
        model_prices=model_prices,
        resume=resume,
    )


def main() -> None:
    fire.Fire(run)
