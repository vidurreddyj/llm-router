"""
Evaluation harness: this is what turns the router from a demo into
evidence. It runs the same benchmark set through three conditions:

  1. always_cheap   -- every query goes to the cheap model
  2. always_strong  -- every query goes to the strong model
  3. router         -- your routing + caching + escalation logic

...and measures cost, latency, and quality for each, then plots a
cost-vs-quality Pareto chart. The pitch of the whole project is: "the
router gets close to always_strong's quality at close to always_cheap's
cost."

Quality is scored with an LLM-as-judge (the strong model grades each
response 1-5 for correctness/completeness, no reference answer needed).
This is a common, pragmatic approach when you don't have hand-written
gold answers for every query -- note in your writeup that a stronger
version would use human-labeled or reference-based scoring for higher
confidence in the numbers.

Usage:
    export GOOGLE_API_KEY=...   # free key from https://aistudio.google.com
    python -m eval.run_eval

Note on rate limits: this project uses Gemini's free tier, which caps
requests per minute (roughly 10-15 RPM depending on model/account as of
2026 -- check https://ai.google.dev/gemini-api/docs/rate-limits for current
numbers). RATE_LIMIT_DELAY_S below adds a small pause between calls so a
full eval run (which makes ~2-3x len(benchmark) calls, since every answer
also gets a judge call) doesn't get throttled. Increase it if you see 429
errors.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import LLMRouter, clients

BENCHMARK_PATH = Path(__file__).parent / "benchmark.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
RATE_LIMIT_DELAY_S = 3.0


def load_benchmark():
    with open(BENCHMARK_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_quality(query: str, response: str) -> float:
    """LLM-as-judge: rate response quality 1-5, return normalized 0-1."""
    prompt = (
        "Grade this AI assistant answer strictly, on a 1-5 scale:\n"
        "5 = fully correct, complete, and well-explained\n"
        "4 = correct but missing minor detail or nuance\n"
        "3 = partially correct, or correct but poorly explained\n"
        "2 = significant errors or major omissions\n"
        "1 = wrong or unhelpful\n\n"
        "Do not default to 5 -- most real answers have at least a minor "
        "gap and should score 4 or lower unless truly flawless. "
        "Respond with ONLY the number, nothing else.\n\n"
        f"Question: {query}\n\nAnswer: {response}"
    )
    result = clients.call_model("judge", prompt)
    try:
        raw = "".join(c for c in result.text if c.isdigit() or c == ".")
        score = float(raw.split(".")[0] + ("." + raw.split(".")[1] if "." in raw else ""))
    except (ValueError, IndexError):
        score = 3.0  # neutral fallback if parsing fails
    return max(1.0, min(5.0, score)) / 5.0


def run_condition_fixed(model_key: str, benchmark: list) -> dict:
    """Run every query through a single fixed model (no routing/caching)."""
    print(f"  ({len(benchmark)} queries x 2 calls each [answer + judge], "
          f"~{len(benchmark) * 2 * RATE_LIMIT_DELAY_S:.0f}s minimum)")
    total_cost = 0.0
    latencies = []
    quality_scores = []

    for item in benchmark:
        result = clients.call_model(model_key, item["query"])
        time.sleep(RATE_LIMIT_DELAY_S)
        total_cost += result.cost_usd
        latencies.append(result.latency_s)
        quality_scores.append(judge_quality(item["query"], result.text))
        print(f"  [{item['id']:>2}/{len(benchmark)}] done "
              f"(cost=${result.cost_usd:.5f}, latency={result.latency_s:.2f}s)",
              flush=True)

    return {
        "condition": f"always_{model_key}",
        "total_cost_usd": round(total_cost, 5),
        "avg_latency_s": round(statistics.mean(latencies), 3),
        "avg_quality": round(statistics.mean(quality_scores), 3),
        "n": len(benchmark),
    }


def run_condition_router(benchmark: list) -> dict:
    """Run every query through the router (cache + classify + route +
    escalate)."""
    router = LLMRouter()
    total_cost = 0.0
    latencies = []
    quality_scores = []
    cheap_count = strong_count = cache_count = escalated_count = 0

    for item in benchmark:
        trace = router.route(item["query"])
        time.sleep(RATE_LIMIT_DELAY_S)
        total_cost += trace.cost_usd
        latencies.append(trace.latency_s)
        quality_scores.append(judge_quality(item["query"], trace.response))

        if trace.cache_hit:
            cache_count += 1
        elif trace.model_used == "cheap":
            cheap_count += 1
        elif trace.model_used == "strong":
            strong_count += 1
        if trace.escalated:
            escalated_count += 1

        tag = "CACHE" if trace.cache_hit else trace.model_used.upper()
        print(f"  [{item['id']:>2}/{len(benchmark)}] {tag:>6} "
              f"(cost=${trace.cost_usd:.5f})", flush=True)

    return {
        "condition": "router",
        "total_cost_usd": round(total_cost, 5),
        "avg_latency_s": round(statistics.mean(latencies), 3),
        "avg_quality": round(statistics.mean(quality_scores), 3),
        "n": len(benchmark),
        "routed_to_cheap": cheap_count,
        "routed_to_strong": strong_count,
        "cache_hits": cache_count,
        "escalations": escalated_count,
        "cache_stats": router.cache.stats(),
    }


def plot_pareto(results: list, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"always_cheap": "#888888", "always_strong": "#1f6feb", "router": "#2ea043"}
    for r in results:
        ax.scatter(r["total_cost_usd"], r["avg_quality"], s=140,
                    color=colors.get(r["condition"], "black"), zorder=3)
        ax.annotate(r["condition"], (r["total_cost_usd"], r["avg_quality"]),
                     textcoords="offset points", xytext=(8, 6), fontsize=10)

    ax.set_xlabel("Total cost (USD) for benchmark run")
    ax.set_ylabel("Avg quality score (0-1, LLM-judged)")
    ax.set_title("Cost vs. Quality: Router vs. Fixed Model Baselines")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit(
            "Set GOOGLE_API_KEY in your environment before running the eval. "
            "Get a free key (no credit card) at https://aistudio.google.com"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    benchmark = load_benchmark()
    print(f"Loaded {len(benchmark)} benchmark queries.\n")

    results = []

    print("Running always_cheap...")
    results.append(run_condition_fixed("cheap", benchmark))

    print("Running always_strong...")
    results.append(run_condition_fixed("strong", benchmark))

    print("Running router...")
    results.append(run_condition_router(benchmark))

    report_path = RESULTS_DIR / "results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {report_path}")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['condition']:>14}: cost=${r['total_cost_usd']:.5f}  "
              f"avg_quality={r['avg_quality']:.3f}  "
              f"avg_latency={r['avg_latency_s']:.2f}s")

    cheap = next(r for r in results if r["condition"] == "always_cheap")
    strong = next(r for r in results if r["condition"] == "always_strong")
    router = next(r for r in results if r["condition"] == "router")

    if strong["total_cost_usd"] > 0:
        cost_saved_pct = 100 * (1 - router["total_cost_usd"] / strong["total_cost_usd"])
        quality_retained_pct = 100 * (router["avg_quality"] / strong["avg_quality"])
        print(f"\nRouter saved {cost_saved_pct:.1f}% of cost vs. always_strong, "
              f"while retaining {quality_retained_pct:.1f}% of its quality.")

    plot_pareto(results, RESULTS_DIR / "pareto.png")


if __name__ == "__main__":
    main()
