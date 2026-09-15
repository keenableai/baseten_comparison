# baseten-comparison

Compare web-search providers exposed as Baseten server tools (Keenable, Exa, Parallel, You.com)
on SimpleQA-Verified and AA-Omniscience. A Baseten-hosted model answers each question with
search; GPT-5.5 (via OpenRouter) grades the answer against the gold target.

`bench-exa-agent` runs the same benchmarks against [Exa Agent](https://exa.ai/docs/reference/agent-api-guide)
directly (api.exa.ai, no Baseten model) with the same system prompt and grader.

`bench-native` runs them against a vendor's own model and built-in search tools: Anthropic
(`claude-sonnet-5` with `web_search` + `web_fetch`) or OpenAI (`gpt-5.6-terra` with `web_search`,
whose `open_page` action fetches pages). Same system prompt and grader.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in BASETEN_API_KEY, OPENROUTER_API_KEY (and EXA_API_KEY for Exa Agent,
                       # ANTHROPIC_API_KEY / OPENAI_API_KEY for bench-native)
```

## Run

```bash
uv run bench simpleqa --n 1000 --provider keenable --model deepseek-ai/DeepSeek-V4-Pro --concurrency 3
uv run bench omniscience --n 600 --provider exa --model deepseek-ai/DeepSeek-V4-Flash-0731 --concurrency 3
uv run bench -- --help
uv run bench-exa-agent simpleqa --n 1000 --effort low --concurrency 8
uv run bench-exa-agent -- --help
uv run bench-native simpleqa anthropic --n 1000 --effort medium --concurrency 8
uv run bench-native omniscience openai --n 600 --effort medium --concurrency 8
uv run bench-native -- --help
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

Exa Agent has a fixed price per request by `--effort` (low $0.025, medium $0.10, high $0.50; Sep 2026).
The harness records `costDollars` as returned by the API; `usage.searches` and compute units are the
tier's nominal allocation, not the work actually done, so they are constant per request.

`bench-native` uses vendor list prices, Sep 2026: Sonnet 5 $2/$10 per M tokens, gpt-5.6-terra
$2/$12. Both vendors bill $10 per 1k searches; Anthropic `web_fetch` and OpenAI `open_page` /
`find_in_page` cost tokens only; the ~1% of OpenAI `web_search_call` items that carry no action are
counted as searches. `--effort` maps to Anthropic `output_config.effort` with adaptive
thinking, and to OpenAI `reasoning.effort`. Cached input tokens are billed at the full input rate in
the table, so model cost is an upper bound.

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

Keenable vs Exa vs You.com vs Parallel, run 2026-09-14/15. Grader GPT-5.5, seed 0, costs in USD for the full run using
the price table above. Raw rows are in `results/`. Earlier DeepSeek-V4-Pro / V4-Flash results:
https://paste.keenable.ai/simpleqa-keenable-vs-exa-deepseek-v4.

### SimpleQA-Verified (1000 questions per cell)

#### DeepSeek-V4.1-Flash

| | Keenable | Exa | You.com | Parallel |
| --- | --- | --- | --- | --- |
| Accuracy | **96.8%** | **96.5%** | **97.3%** | **93.2%** |
| Correct / Incorrect / Not attempted | 968 / 30 / 2 | 965 / 35 / 0 | 973 / 27 / 0 | 932 / 29 / 39 |
| Search calls (search + fetch) | 1949 + 78 | 1738 + 63 | 1913 + 38 | 1074 + 0 |
| Search cost | $8.11 | $12.23 | $9.60 | $5.37 |
| Model cost | $3.21 | $2.98 | $6.48 | $1.75 |
| Model + search cost | $11.32 | $15.21 | $16.09 | $7.12 |
| Answer latency p50 / p90 | 2.1s / 11.4s | 4.0s / 10.5s | 4.6s / 26.3s | 4.3s / 24.1s |

#### GLM-5.3-Fast

| | Keenable | Exa | You.com | Parallel |
| --- | --- | --- | --- | --- |
| Accuracy | **97.5%** | **96.7%** | **97.6%** | **97.4%** |
| Correct / Incorrect / Not attempted | 975 / 25 / 0 | 967 / 33 / 0 | 976 / 21 / 3 | 974 / 26 / 0 |
| Search calls (search + fetch) | 1394 + 85 | 1257 + 41 | 1258 + 43 | 1085 + 0 |
| Search cost | $5.92 | $8.84 | $6.33 | $5.42 |
| Model cost | $16.68 | $14.11 | $37.02 | $11.08 |
| Model + search cost | $22.60 | $22.95 | $43.35 | $16.51 |
| Answer latency p50 / p90 | 3.1s / 5.0s | 6.2s / 16.1s | 4.1s / 24.6s | 4.0s / 29.1s |

### AA-Omniscience (600 questions per cell, full public set)

Omniscience index = 100 × (correct − incorrect) / total. Declining costs nothing; a wrong answer costs as much as a right one earns.

#### DeepSeek-V4.1-Flash

| | Keenable | Exa | You.com | Parallel |
| --- | --- | --- | --- | --- |
| Omniscience index | **+76.2** | **+77.0** | **+76.5** | **+72.5** |
| Accuracy | 85.0% | 86.7% | 87.0% | 80.5% |
| Correct / Incorrect / Not attempted | 510 / 53 / 37 | 520 / 58 / 22 | 522 / 63 / 15 | 483 / 48 / 69 |
| Accuracy on attempted | 90.6% | 90.0% | 89.2% | 91.0% |
| Search calls (search + fetch) | 2243 + 457 | 1920 + 268 | 2093 + 249 | 1153 + 0 |
| Search cost | $10.80 | $13.71 | $10.71 | $5.76 |
| Model cost | $4.85 | $6.11 | $8.06 | $2.59 |
| Model + search cost | $15.65 | $19.82 | $18.78 | $8.36 |
| Answer latency p50 / p90 | 5.6s / 30.1s | 8.3s / 36.1s | 9.1s / 39.4s | 6.2s / 36.7s |

| Domain (correct / index) | Keenable | Exa | You.com | Parallel |
| --- | --- | --- | --- | --- |
| Finance | 86 / +77 | 90 / +81 | 90 / +82 | 80 / +70 |
| Health | 83 / +68 | 79 / +61 | 84 / +68 | 77 / +64 |
| Humanities and Social Sciences | 78 / +64 | 84 / +73 | 83 / +68 | 80 / +74 |
| Law | 95 / +94 | 99 / +98 | 96 / +92 | 87 / +84 |
| Science Engineering and Mathematics | 75 / +65 | 72 / +56 | 74 / +59 | 68 / +57 |
| Software Engineering | 93 / +89 | 96 / +93 | 95 / +90 | 91 / +86 |

#### GLM-5.3-Fast

| | Keenable | Exa | You.com | Parallel |
| --- | --- | --- | --- | --- |
| Omniscience index | **+73.7** | **+75.8** | **+74.3** | **+75.2** |
| Accuracy | 85.8% | 87.5% | 85.3% | 86.3% |
| Correct / Incorrect / Not attempted | 515 / 73 / 12 | 525 / 70 / 5 | 512 / 66 / 22 | 518 / 67 / 15 |
| Accuracy on attempted | 87.6% | 88.2% | 88.6% | 88.5% |
| Search calls (search + fetch) | 1689 + 503 | 1294 + 202 | 1507 + 212 | 1100 + 0 |
| Search cost | $8.77 | $9.26 | $7.75 | $5.50 |
| Model cost | $26.79 | $28.40 | $83.72 | $20.20 |
| Model + search cost | $35.56 | $37.66 | $91.47 | $25.70 |
| Answer latency p50 / p90 | 5.1s / 58.7s | 14.8s / 51.4s | 6.2s / 37.0s | 6.0s / 36.0s |

| Domain (correct / index) | Keenable | Exa | You.com | Parallel |
| --- | --- | --- | --- | --- |
| Finance | 90 / +82 | 92 / +85 | 88 / +79 | 91 / +83 |
| Health | 75 / +50 | 77 / +55 | 78 / +58 | 79 / +59 |
| Humanities and Social Sciences | 84 / +71 | 86 / +74 | 78 / +59 | 85 / +72 |
| Law | 93 / +88 | 96 / +92 | 93 / +88 | 90 / +81 |
| Science Engineering and Mathematics | 81 / +66 | 77 / +55 | 80 / +72 | 78 / +66 |
| Software Engineering | 92 / +85 | 97 / +94 | 95 / +90 | 95 / +90 |

### Exa Agent (standalone, no Baseten model)

Run 2026-09-15 with `bench-exa-agent`, same questions (seed 0), same system prompt and grader.
Effort is Exa's fixed-price tier; cost is `costDollars` summed over the run. Raw rows are in
`results/` (LFS). Reference cells from the tables above are repeated for comparison.

#### SimpleQA-Verified (1000 questions)

| | Exa Agent low | Exa Agent medium | Exa Agent high | Keenable + GLM-5.3-Fast | Exa + GLM-5.3-Fast |
| --- | --- | --- | --- | --- | --- |
| Accuracy | **97.0%** | **96.3%** | **96.8%** | **97.5%** | **96.7%** |
| Correct / Incorrect / Not attempted | 970 / 29 / 1 | 963 / 36 / 1 | 968 / 31 / 1 | 975 / 25 / 0 | 967 / 33 / 0 |
| Cost | $25.00 | $100.00 | $500.00 | $22.60 | $22.95 |
| Answer latency p50 / p90 | 9.1s / 15.7s | 13.6s / 22.4s | 11.4s / 18.1s | 3.1s / 5.0s | 6.2s / 16.1s |

#### AA-Omniscience (600 questions)

| | Exa Agent low | Exa Agent medium | Exa Agent high | Keenable + V4.1-Flash | Exa + V4.1-Flash |
| --- | --- | --- | --- | --- | --- |
| Omniscience index | **+73.0** | **+78.0** | **+80.8** | **+76.2** | **+77.0** |
| Accuracy | 86.2% | 88.8% | 90.2% | 85.0% | 86.7% |
| Correct / Incorrect / Not attempted | 517 / 79 / 4 | 533 / 65 / 2 | 541 / 56 / 2 | 510 / 53 / 37 | 520 / 58 / 22 |
| Accuracy on attempted | 86.7% | 89.1% | 90.6% | 90.6% | 90.0% |
| Cost | $15.00 | $60.00 | $300.00 | $15.65 | $19.82 |
| Answer latency p50 / p90 | 11.3s / 20.3s | 17.9s / 35.9s | 15.8s / 51.4s | 5.6s / 30.1s | 8.3s / 36.1s |

| Domain (correct / index) | Exa Agent low | Exa Agent medium | Exa Agent high |
| --- | --- | --- | --- |
| Finance | 90 / +80 | 93 / +86 | 92 / +85 |
| Health | 78 / +57 | 80 / +60 | 87 / +75 |
| Humanities and Social Sciences | 85 / +70 | 89 / +79 | 85 / +71 |
| Law | 94 / +89 | 94 / +89 | 98 / +96 |
| Science, Engineering and Mathematics | 72 / +46 | 78 / +56 | 80 / +60 |
| Software Engineering | 98 / +96 | 99 / +98 | 99 / +98 |

Exa returned 500s for about 15% of medium-tier requests during one window on 2026-09-15; those rows
were rerun with `--resume`. Low and high had no agent errors. One high-tier Omniscience row is
`GRADER_ERROR` in the raw data: GPT-5.5's content filter refused the grading prompt. The answer
(72) matches the target; it is not counted as correct above.

### Vendor-native search (Sonnet 5, gpt-5.6-terra)

Run 2026-09-15 with `bench-native --effort medium`, same questions (seed 0), same system prompt and
grader. Sonnet 5 uses Anthropic's `web_search` + `web_fetch` server tools with adaptive thinking;
gpt-5.6-terra uses OpenAI's `web_search` tool (its `open_page` action is the fetch) with reasoning
effort medium. Cost is model tokens + billed searches at the vendor list prices above. Raw rows are
in `results/` (LFS).

#### SimpleQA-Verified (1000 questions)

| | gpt-5.6-terra | Sonnet 5 | Exa Agent medium | Keenable + GLM-5.3-Fast |
| --- | --- | --- | --- | --- |
| Accuracy | **96.5%** | **94.2%** | **96.3%** | **97.5%** |
| Correct / Incorrect / Not attempted | 965 / 34 / 1 | 942 / 53 / 5 | 963 / 36 / 1 | 975 / 25 / 0 |
| Search calls (search + fetch) | 1057 + 35 | 2500 + 244 | n/a | 1394 + 85 |
| Search cost | $10.57 | $25.00 | n/a | $5.92 |
| Model cost | $22.54 | $91.63 | n/a | $16.68 |
| Model + search cost | $33.11 | $116.63 | $100.00 | $22.60 |
| Answer latency p50 / p90 | 4.6s / 6.9s | 17.0s / 25.9s | 13.6s / 22.4s | 3.1s / 5.0s |

#### AA-Omniscience (600 questions)

| | gpt-5.6-terra | Sonnet 5 | Exa Agent medium | Keenable + V4.1-Flash |
| --- | --- | --- | --- | --- |
| Omniscience index | **+81.3** | **+69.2** | **+78.0** | **+76.2** |
| Accuracy | 90.5% | 82.8% | 88.8% | 85.0% |
| Correct / Incorrect / Not attempted | 543 / 55 / 2 | 497 / 82 / 21 | 533 / 65 / 2 | 510 / 53 / 37 |
| Accuracy on attempted | 90.8% | 85.8% | 89.1% | 90.6% |
| Search calls (search + fetch) | 791 + 113 | 2503 + 578 | n/a | 2243 + 457 |
| Search cost | $7.91 | $25.03 | n/a | $10.80 |
| Model cost | $19.30 | $133.10 | n/a | $4.85 |
| Model + search cost | $27.21 | $158.13 | $60.00 | $15.65 |
| Answer latency p50 / p90 | 5.3s / 12.9s | 18.0s / 82.8s | 17.9s / 35.9s | 5.6s / 30.1s |

| Domain (correct / index) | gpt-5.6-terra | Sonnet 5 |
| --- | --- | --- |
| Finance | 86 / +74 | 83 / +71 |
| Health | 82 / +64 | 74 / +50 |
| Humanities and Social Sciences | 94 / +88 | 77 / +61 |
| Law | 97 / +94 | 91 / +84 |
| Science, Engineering and Mathematics | 88 / +76 | 79 / +63 |
| Software Engineering | 96 / +92 | 93 / +86 |

gpt-5.6-terra fetch counts are `open_page` + `find_in_page` (96 + 17 on Omniscience, 32 + 3 on
SimpleQA); the search count includes 9 Omniscience `web_search_call` items that carried no action.
No run had an API error.

Sonnet 5's model cost is dominated by input tokens: each server-tool round trip re-reads the whole
context, so a 25-search answer bills ~65M input tokens per 600 questions. Six Sonnet 5 Omniscience
rows were rerun: four stopped with `pause_turn` (Anthropic pauses long tool loops and expects the
client to resend the turn, which the runner now does) and two spent the whole 8192-token output cap
on thinking (the cap is now 16384). One SimpleQA row spent 32 minutes in SDK retries before answering.
