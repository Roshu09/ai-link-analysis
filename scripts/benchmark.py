"""Measure real speed, token usage and output reliability of Gemini models on real websites.

Usage (from the project root, with GEMINI_API_KEY set in .env):

    python scripts/benchmark.py --urls https://www.python.org https://www.shopify.com https://www.notion.so
    python scripts/benchmark.py --models gemini-3.5-flash-lite,gemini-3.1-flash-lite --urls ...

For every (model, website) pair it runs the REAL pipeline (fetch -> LLM -> validation) and records:
  * latency of the LLM call,
  * prompt / output tokens reported by the API,
  * whether the answer passed our validation on the first try,
  * the estimated paid-tier cost (free tier costs $0, but this shows what production would cost).

It prints a markdown table to paste into README.md ("measured" results) and saves each full
JSON response under docs/samples/ so you can attach real sample responses to your submission.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.schemas import LLMOutput  # noqa: E402
from app.services.analysis_service import AnalysisService  # noqa: E402
from app.services.llm_service import GeminiClient  # noqa: E402
from app.services.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from app.services.website_extractor import WebsiteExtractor  # noqa: E402
from app.utils.validators import normalize_url  # noqa: E402

# USD per 1M tokens (input, output), paid tier. Re-check against Google's pricing page before quoting.
PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


def cost(model: str, tokens_in: int | None, tokens_out: int | None) -> float | None:
    if model not in PRICES or tokens_in is None or tokens_out is None:
        return None
    price_in, price_out = PRICES[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


async def run(models: list[str], urls: list[str], save_dir: Path) -> None:
    base = get_settings()
    extractor = WebsiteExtractor(base)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fetch each site once so every model sees identical input.
    sites = {}
    for raw in urls:
        url = normalize_url(raw)
        try:
            sites[url] = await extractor.extract(url)
        except Exception as exc:  # keep going if one site is unreachable
            print(f"! skipping {url}: {exc}")

    rows = []
    for model in models:
        settings = base.model_copy(update={"gemini_model": model, "gemini_fallback_model": ""})
        client = GeminiClient(settings)
        service = AnalysisService(extractor, client, settings)
        latencies, ok_first_try, costs = [], 0, []

        for url, content in sites.items():
            prompt = build_user_prompt(content.to_prompt_text(settings.max_content_chars), content.final_url)
            try:
                result = await client.generate_json(SYSTEM_PROMPT, prompt)
            except Exception as exc:
                print(f"! {model} failed on {url}: {exc}")
                continue
            latencies.append(result.latency_ms / 1000)
            try:
                output = LLMOutput.model_validate(result.data)
                service._normalize_counts(output)
                service._check_quality(output)
                ok_first_try += 1
            except Exception as exc:
                print(f"  {model} / {url}: failed validation: {str(exc)[:120]}")
            c = cost(model, result.prompt_tokens, result.output_tokens)
            if c is not None:
                costs.append(c)
            print(f"  {model:24s} {url:40s} {result.latency_ms/1000:5.1f}s  in={result.prompt_tokens} out={result.output_tokens}")
            slug = re.sub(r"[^a-z0-9]+", "-", f"{model}-{url}".lower()).strip("-")
            (save_dir / f"{slug}.json").write_text(json.dumps(result.data, indent=2, ensure_ascii=False))
            rows.append((model, result))

        if latencies:
            model_rows = [r for m, r in rows if m == model]
            avg_in = statistics.mean(r.prompt_tokens or 0 for r in model_rows)
            avg_out = statistics.mean(r.output_tokens or 0 for r in model_rows)
            print(
                f"\n| {model} | {statistics.mean(latencies):.1f}s | {max(latencies):.1f}s | "
                f"{avg_in:.0f} | {avg_out:.0f} | {ok_first_try}/{len(sites)} | "
                f"{'$%.4f' % statistics.mean(costs) if costs else 'n/a'} |\n"
            )

    print("Header for the table above:")
    print("| Model | Avg latency | Slowest | Avg input tokens | Avg output tokens | Valid first try | Est. paid cost / analysis |")
    print("|---|---|---|---|---|---|---|")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urls", nargs="+", required=True, help="websites to test")
    parser.add_argument(
        "--models",
        default="gemini-3.5-flash-lite,gemini-3.1-flash-lite",
        help="comma-separated Gemini model IDs",
    )
    parser.add_argument("--save-dir", default="docs/samples", type=Path)
    args = parser.parse_args()
    asyncio.run(run([m.strip() for m in args.models.split(",") if m.strip()], args.urls, args.save_dir))
