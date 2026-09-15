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

- Model: Baseten serverless $/M tokens, July 2026 list prices (models.dev). DeepSeek-V4-Flash-0731
  and GLM-5.2-Fast have no published Baseten rate; official/Fireworks list prices are used.
  DeepSeek-V4.1-Flash and GLM-5.3-Fast use Baseten model-library list prices, Sep 2026.
- Search, $/1k calls, Aug 2026: Exa $7 search / $1 contents (exa.ai/pricing), Parallel $5
  (docs.parallel.ai), You.com $5 search / $1 contents (you.com/pricing). Keenable has no public
  per-request price; $4/1k is an internal figure applied to search and fetch.

Pass `--model_input_price` / `--model_output_price` to override the table for one run.

## CLI flags

`uv run bench -- --help` lists them. `--resume` keeps rows whose question, benchmark, provider,
model and grader match the current run; everything else is rerun.

## Results

Keenable vs Exa, run 2026-09-14. Grader GPT-5.5, seed 0, costs in USD for the full run using
the price table above. Also at https://paste.keenable.ai/simpleqa-keenable-vs-exa-v41-flash-glm53-fast.
Earlier DeepSeek-V4-Pro / V4-Flash results: https://paste.keenable.ai/simpleqa-keenable-vs-exa-deepseek-v4.

### SimpleQA-Verified (1000 questions per cell)

#### DeepSeek-V4.1-Flash

| | Keenable | Exa |
| --- | --- | --- |
| Accuracy | **96.8%** | **96.5%** |
| Correct / Incorrect / Not attempted | 968 / 30 / 2 | 965 / 35 / 0 |
| Search calls (search + fetch) | 1949 + 79 | 1738 + 63 |
| Search cost | $8.11 | $12.23 |
| Model cost | $3.21 | $2.98 |
| Model + search cost | $11.32 | $15.21 |
| Answer latency p50 / p90 | 2.1s / 11.4s | 4.0s / 10.5s |

#### GLM-5.3-Fast

| | Keenable | Exa |
| --- | --- | --- |
| Accuracy | **97.5%** | **96.7%** |
| Correct / Incorrect / Not attempted | 975 / 25 / 0 | 967 / 33 / 0 |
| Search calls (search + fetch) | 1394 + 85 | 1257 + 41 |
| Search cost | $5.92 | $8.84 |
| Model cost | $16.68 | $14.11 |
| Model + search cost | $22.60 | $22.95 |
| Answer latency p50 / p90 | 3.1s / 5.0s | 6.2s / 16.1s |

### AA-Omniscience (600 questions per cell, full public set)

Omniscience index = 100 × (correct − incorrect) / total. Declining costs nothing; a wrong answer costs as much as a right one earns.

#### DeepSeek-V4.1-Flash

| | Keenable | Exa |
| --- | --- | --- |
| Omniscience index | **+76.2** | **+77.0** |
| Accuracy | 85.0% | 86.7% |
| Correct / Incorrect / Not attempted | 510 / 53 / 37 | 520 / 58 / 22 |
| Accuracy on attempted | 90.6% | 90.0% |
| Search calls (search + fetch) | 2243 + 458 | 1920 + 268 |
| Search cost | $10.80 | $13.71 |
| Model cost | $4.85 | $6.11 |
| Model + search cost | $15.65 | $19.82 |
| Answer latency p50 / p90 | 5.6s / 30.1s | 8.3s / 36.1s |

#### GLM-5.3-Fast

| | Keenable | Exa |
| --- | --- | --- |
| Omniscience index | **+73.7** | **+75.8** |
| Accuracy | 85.8% | 87.5% |
| Correct / Incorrect / Not attempted | 515 / 73 / 12 | 525 / 70 / 5 |
| Accuracy on attempted | 87.6% | 88.2% |
| Search calls (search + fetch) | 1689 + 503 | 1294 + 202 |
| Search cost | $8.77 | $9.26 |
| Model cost | $26.79 | $28.40 |
| Model + search cost | $35.56 | $37.66 |
| Answer latency p50 / p90 | 5.1s / 58.7s | 14.8s / 51.4s |

#### Omniscience by domain (correct out of 100 / index)

| Domain | V4.1-Flash Keenable | V4.1-Flash Exa | GLM-5.3-Fast Keenable | GLM-5.3-Fast Exa |
| --- | --- | --- | --- | --- |
| Finance | 86 / +77 | 90 / +81 | 90 / +82 | 92 / +85 |
| Health | 83 / +68 | 79 / +61 | 75 / +50 | 77 / +55 |
| Humanities and Social Sciences | 78 / +64 | 84 / +73 | 84 / +71 | 86 / +74 |
| Law | 95 / +94 | 99 / +98 | 93 / +88 | 96 / +92 |
| Science, Engineering and Mathematics | 75 / +65 | 72 / +56 | 81 / +66 | 77 / +55 |
| Software Engineering | 93 / +89 | 96 / +93 | 92 / +85 | 97 / +94 |

### Notes

- SimpleQA: Keenable leads by 3 correct on V4.1-Flash and 8 on GLM-5.3-Fast. Keenable is 2-3s faster at p50 and 30% cheaper on search.
- Omniscience: Exa leads by 1-2 index points. The gap comes from fewer declines (22 vs 37, 5 vs 12), not from better accuracy on attempted questions, which is within 1 point either way.
- Keenable makes 15-30% more searches and 1.7-2.5x more fetches, yet still costs less per run on SimpleQA and about the same on Omniscience.
- Exa is 1.5-3x slower at p50 across all four cells. GLM on Exa Omniscience had one 1210s outlier.
- V4.1-Flash misspelled the Keenable tool name in ~15 calls per run; those are excluded from counts and search cost.
- Baseten returned 429s on the V4.1-Flash runs; resume passes at lower concurrency filled every row. All cells have zero errors.
