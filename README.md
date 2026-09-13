# baseten-comparison

Compare web-search providers exposed as Baseten server tools (Keenable, Exa, Parallel, You.com)
on SimpleQA-Verified and AA-Omniscience. A Baseten-hosted model answers each question with
search; GPT-5.5 (via OpenRouter) grades the answer against the gold target.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in BASETEN_API_KEY and OPENROUTER_API_KEY
```

## Run

```bash
uv run bench simpleqa --n 1000 --provider keenable --model deepseek-ai/DeepSeek-V4-Pro --concurrency 3
uv run bench omniscience --n 600 --provider exa --model deepseek-ai/DeepSeek-V4-Flash-0731 --concurrency 3
uv run bench -- --help
```

Output is one JSONL row per question (`<benchmark>_<provider>_<model>.jsonl` by default) with the
grade, full model response, grader response, latency, search calls and cost.

Baseten returns 429s when more than roughly 12 requests are in flight across all runs. Rate-limited
rows are retried with backoff; anything still failing is written with `grade: ERROR`. Fill those
rows with `--resume` on the same output path.

## Prices

Per-token model prices and per-call search prices live in `src/baseten_comparison/harness.py`.
Keenable has no public per-request price; $4/1k is an internal figure applied to search and fetch.
