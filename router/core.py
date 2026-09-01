"""
LLMRouter: the piece that ties everything together.

Flow for each query:
  1. Check the semantic cache -- return immediately on a hit (free, instant).
  2. Score the query's complexity with the heuristic classifier.
  3. Route: complexity below threshold -> cheap model, else -> strong model.
  4. If the cheap model was used, check its self-reported confidence.
     Low confidence -> escalate and re-answer with the strong model.
  5. Store the final answer in the cache and return a full trace of what
     happened (useful for logging/eval, not just the answer text).
"""

from dataclasses import dataclass, field
from typing import Optional

from . import classifier, clients, config
from .cache import SemanticCache


@dataclass
class RouteTrace:
    query: str
    cache_hit: bool
    complexity_score: float
    model_used: str          # "cheap" or "strong" or "cache"
    escalated: bool
    confidence: Optional[float]
    cost_usd: float
    latency_s: float
    response: str


class LLMRouter:
    def __init__(
        self,
        complexity_threshold: float = config.COMPLEXITY_ROUTE_THRESHOLD,
        confidence_threshold: float = config.CONFIDENCE_ESCALATION_THRESHOLD,
        cache_similarity_threshold: float = config.CACHE_SIMILARITY_THRESHOLD,
        use_cache: bool = True,
    ):
        self.complexity_threshold = complexity_threshold
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        self.cache = SemanticCache(similarity_threshold=cache_similarity_threshold)

    def route(self, query: str) -> RouteTrace:
        # 1. Cache lookup
        if self.use_cache:
            hit = self.cache.lookup(query)
            if hit is not None:
                return RouteTrace(
                    query=query,
                    cache_hit=True,
                    complexity_score=-1.0,
                    model_used="cache",
                    escalated=False,
                    confidence=None,
                    cost_usd=0.0,
                    latency_s=0.0,
                    response=hit.response,
                )

        # 2. Classify complexity
        complexity = classifier.score(query)

        # 3. Route
        escalated = False
        if complexity >= self.complexity_threshold:
            result = clients.call_model("strong", query)
        else:
            result = clients.call_model("cheap", query, ask_confidence=True)
            # 4. Escalate if the cheap model wasn't confident
            if (
                result.confidence is not None
                and result.confidence < self.confidence_threshold
            ):
                escalated = True
                strong_result = clients.call_model("strong", query)
                total_cost = result.cost_usd + strong_result.cost_usd
                total_latency = result.latency_s + strong_result.latency_s
                result = strong_result
                result.cost_usd = total_cost
                result.latency_s = total_latency

        # 5. Cache the final answer
        if self.use_cache:
            self.cache.store(query, result.text, result.model_key)

        return RouteTrace(
            query=query,
            cache_hit=False,
            complexity_score=complexity,
            model_used=result.model_key,
            escalated=escalated,
            confidence=result.confidence,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
            response=result.text,
        )
