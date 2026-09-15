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

Output is one JSONL row per question (`results/<benchmark>_<provider>_<model>.jsonl` by default, tracked with git LFS) with the
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

## Single call

```python
import os
from baseten_comparison.client import answer_question, make_client

client = make_client(os.environ["BASETEN_API_KEY"])
predicted, response, latency_s = answer_question(
    client, "zai-org/GLM-5.3-Fast", "keenable", "Who wrote Ys Origin's soundtrack?"
)
```

Prompts live in `src/baseten_comparison/prompts/*.jinja`.

## CLI flags

`uv run bench -- --help` lists them. `--resume` keeps rows whose question, benchmark, provider,
model and grader match the current run; everything else is rerun.

## Results

Keenable vs Exa vs You.com, run 2026-09-14/15. Grader GPT-5.5, seed 0, costs in USD for the full run using
the price table above. Raw rows are in `results/`. Earlier DeepSeek-V4-Pro / V4-Flash results:
https://paste.keenable.ai/simpleqa-keenable-vs-exa-deepseek-v4.

### SimpleQA-Verified (1000 questions per cell)

#### DeepSeek-V4.1-Flash

| | Keenable | Exa | You.com |
| --- | --- | --- | --- |
| Accuracy | **96.8%** | **96.5%** | **97.3%** |
| Correct / Incorrect / Not attempted | 968 / 30 / 2 | 965 / 35 / 0 | 973 / 27 / 0 |
| Search calls (search + fetch) | 1949 + 78 | 1738 + 63 | 1913 + 38 |
| Search cost | $8.11 | $12.23 | $9.60 |
| Model cost | $3.21 | $2.98 | $6.48 |
| Model + search cost | $11.32 | $15.21 | $16.09 |
| Answer latency p50 / p90 | 2.1s / 11.4s | 4.0s / 10.5s | 4.6s / 26.3s |

#### GLM-5.3-Fast

| | Keenable | Exa | You.com |
| --- | --- | --- | --- |
| Accuracy | **97.5%** | **96.7%** | **97.6%** |
| Correct / Incorrect / Not attempted | 975 / 25 / 0 | 967 / 33 / 0 | 976 / 21 / 3 |
| Search calls (search + fetch) | 1394 + 85 | 1257 + 41 | 1258 + 43 |
| Search cost | $5.92 | $8.84 | $6.33 |
| Model cost | $16.68 | $14.11 | $37.02 |
| Model + search cost | $22.60 | $22.95 | $43.35 |
| Answer latency p50 / p90 | 3.1s / 5.0s | 6.2s / 16.1s | 4.1s / 24.6s |

### AA-Omniscience (600 questions per cell, full public set)

Omniscience index = 100 × (correct − incorrect) / total. Declining costs nothing; a wrong answer costs as much as a right one earns.

#### DeepSeek-V4.1-Flash

| | Keenable | Exa | You.com |
| --- | --- | --- | --- |
| Omniscience index | **+76.2** | **+77.0** | **+76.5** |
| Accuracy | 85.0% | 86.7% | 87.0% |
| Correct / Incorrect / Not attempted | 510 / 53 / 37 | 520 / 58 / 22 | 522 / 63 / 15 |
| Accuracy on attempted | 90.6% | 90.0% | 89.2% |
| Search calls (search + fetch) | 2243 + 457 | 1920 + 268 | 2093 + 249 |
| Search cost | $10.80 | $13.71 | $10.71 |
| Model cost | $4.85 | $6.11 | $8.06 |
| Model + search cost | $15.65 | $19.82 | $18.78 |
| Answer latency p50 / p90 | 5.6s / 30.1s | 8.3s / 36.1s | 9.1s / 39.4s |

| Domain (correct / index) | Keenable | Exa | You.com |
| --- | --- | --- | --- |
| Finance | 86 / +77 | 90 / +81 | 90 / +82 |
| Health | 83 / +68 | 79 / +61 | 84 / +68 |
| Humanities and Social Sciences | 78 / +64 | 84 / +73 | 83 / +68 |
| Law | 95 / +94 | 99 / +98 | 96 / +92 |
| Science Engineering and Mathematics | 75 / +65 | 72 / +56 | 74 / +59 |
| Software Engineering | 93 / +89 | 96 / +93 | 95 / +90 |

#### GLM-5.3-Fast

| | Keenable | Exa | You.com |
| --- | --- | --- | --- |
| Omniscience index | **+73.7** | **+75.8** | **+74.3** |
| Accuracy | 85.8% | 87.5% | 85.3% |
| Correct / Incorrect / Not attempted | 515 / 73 / 12 | 525 / 70 / 5 | 512 / 66 / 22 |
| Accuracy on attempted | 87.6% | 88.2% | 88.6% |
| Search calls (search + fetch) | 1689 + 503 | 1294 + 202 | 1507 + 212 |
| Search cost | $8.77 | $9.26 | $7.75 |
| Model cost | $26.79 | $28.40 | $83.72 |
| Model + search cost | $35.56 | $37.66 | $91.47 |
| Answer latency p50 / p90 | 5.1s / 58.7s | 14.8s / 51.4s | 6.2s / 37.0s |

| Domain (correct / index) | Keenable | Exa | You.com |
| --- | --- | --- | --- |
| Finance | 90 / +82 | 92 / +85 | 88 / +79 |
| Health | 75 / +50 | 77 / +55 | 78 / +58 |
| Humanities and Social Sciences | 84 / +71 | 86 / +74 | 78 / +59 |
| Law | 93 / +88 | 96 / +92 | 93 / +88 |
| Science Engineering and Mathematics | 81 / +66 | 77 / +55 | 80 / +72 |
| Software Engineering | 92 / +85 | 97 / +94 | 95 / +90 |
