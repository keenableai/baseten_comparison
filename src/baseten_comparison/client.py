import os
import time
from datetime import date

from anthropic import Anthropic

from baseten_comparison.prompts import render

DEFAULT_BASE_URL = "https://inference.baseten.co"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

SERVER_TOOLS_BY_PROVIDER = {
    "exa": ["baseten__exa__web_search_exa", "baseten__exa__web_fetch_exa"],
    "keenable": ["baseten__keenable__search_web_pages", "baseten__keenable__fetch_page_content"],
    "parallel": ["baseten__parallel__web_search"],
    "youcom": ["baseten__youcom__you-search", "baseten__youcom__you-contents"],
}
SERVER_TOOLS_BY_PROVIDER["all"] = [t for ts in SERVER_TOOLS_BY_PROVIDER.values() for t in ts]


def system_prompt(suffix: str = "") -> str:
    return render("system", today=date.today().isoformat(), suffix=suffix)


def make_client(api_key: str) -> Anthropic:
    return Anthropic(
        api_key=api_key,
        base_url=os.environ.get("BASETEN_BASE_URL", DEFAULT_BASE_URL),
        default_headers={"Authorization": f"Bearer {api_key}", "x-baseten-server-tools": "true"},
        timeout=180.0,
        max_retries=8,
    )


def answer_question(
    client: Anthropic, model: str, provider: str, question: str, system: str | None = None
) -> tuple[str, dict, float]:
    start = time.perf_counter()
    final = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system or system_prompt(),
        tools=[{"type": t} for t in SERVER_TOOLS_BY_PROVIDER[provider]],
        extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        messages=[{"role": "user", "content": question}],
    )
    latency = time.perf_counter() - start
    predicted = "\n".join(b.text for b in final.content if b.type == "text").strip()
    return predicted, final.model_dump(mode="json", warnings=False), latency
