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
    """Run N benchmark samples through a Baseten model + search provider and grade them.

    Args:
        benchmark: simpleqa (1000 rows) or omniscience (600 rows).
        n: number of samples.
        provider: one of exa/keenable/parallel/youcom, or "all".
        model: Baseten model slug; defaults to $BASETEN_MODEL or DeepSeek-V4-Pro.
        grader_model: OpenRouter model slug for grading.
        output: output JSONL path; defaults to <benchmark>_<provider>_<model tail>.jsonl.
        seed: sampling seed.
        concurrency: parallel questions in flight. Baseten 429s above ~12 total.
        model_input_price: Baseten $/M input tokens; overrides the built-in price table.
        model_output_price: Baseten $/M output tokens.
        resume: keep non-error rows already in `output`; rerun only the rest.
    """
    load_dotenv()
    if benchmark not in BENCHMARKS:
        sys.exit(f"unknown benchmark {benchmark!r}; choose from {sorted(BENCHMARKS)}")
    if provider != "all" and provider not in SERVER_TOOLS_BY_PROVIDER:
        sys.exit(f"unknown provider {provider!r}; choose from {[*SERVER_TOOLS_BY_PROVIDER, 'all']}")
    for var in ("BASETEN_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"export {var} first")

    model = model or os.environ.get("BASETEN_MODEL") or DEFAULT_MODEL
    if model_input_price is not None and model_output_price is not None:
        model_prices = (model_input_price, model_output_price)
    else:
        model_prices = MODEL_PRICES.get(model)
    output = output or f"{benchmark}_{provider}_{model.rsplit('/', 1)[-1]}.jsonl"

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
