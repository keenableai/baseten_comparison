"""Benchmark definitions: where the CSV lives and which columns hold the question and answer."""

import csv
import io
import random
from collections.abc import Callable
from dataclasses import dataclass

import httpx


def omniscience_index(grades: list[str]) -> float:
    """AA-Omniscience headline metric: (+1 correct, -1 incorrect, 0 declined) × 100."""
    return 100.0 * (grades.count("CORRECT") - grades.count("INCORRECT")) / len(grades)


def omniscience_summary(records: list[dict]) -> None:
    grades = [r["grade"] for r in records]
    print(f"  omniscience index: {omniscience_index(grades):+.1f}")
    by_domain: dict[str, list[str]] = {}
    for r in records:
        by_domain.setdefault(r["metadata"].get("domain") or "?", []).append(r["grade"])
    if len(by_domain) > 1:
        print("  by domain:")
        for domain, dg in sorted(by_domain.items()):
            print(
                f"    {domain}: {dg.count('CORRECT')}/{len(dg)} correct, "
                f"index {omniscience_index(dg):+.1f}"
            )


@dataclass(frozen=True)
class Benchmark:
    name: str
    url: str
    question_key: str
    answer_key: str
    system_suffix: str = ""
    extra_summary: Callable[[list[dict]], None] | None = None

    def load(self, n: int, seed: int) -> list[dict]:
        """Return n shuffled rows as {question, target, metadata}."""
        resp = httpx.get(self.url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(resp.text)))
        random.Random(seed).shuffle(rows)
        return [
            {
                "question": row[self.question_key],
                "target": row[self.answer_key],
                "metadata": {
                    k: v for k, v in row.items() if k not in (self.question_key, self.answer_key)
                },
            }
            for row in rows[:n]
        ]


BENCHMARKS = {
    "simpleqa": Benchmark(
        name="simpleqa",
        url=(
            "https://huggingface.co/datasets/google/simpleqa-verified"
            "/resolve/main/simpleqa_verified.csv"
        ),
        question_key="problem",
        answer_key="answer",
    ),
    "omniscience": Benchmark(
        name="omniscience",
        url=(
            "https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public"
            "/resolve/main/AA-Omniscience_dataset_public.csv"
        ),
        question_key="question",
        answer_key="answer",
        system_suffix=" If you cannot find the answer, say so plainly instead of guessing.",
        extra_summary=omniscience_summary,
    ),
}
