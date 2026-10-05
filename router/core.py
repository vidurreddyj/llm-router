"""
Main routing module for the LLM Router.

Ties together the semantic cache, complexity classifier,
LLM clients, and configuration settings.

For each query, the router:
    1. Checks the semantic cache for a similar previous query.
    2. Calculates the complexity of the query.
    3. Routes the query to either the cheap or strong model.
    4. Escalates from the cheap model to the strong model if confidence is low.
    5. Stores the final response in the cache.
    6. Returns a RouteTrace describing what happened.
"""

from dataclasses import dataclass, field
from typing import Optional

from . import classifier, clients, config
from .cache import SemanticCache


@dataclass
class RouteTrace:
    """
    Stores information about how a single query was processed.

        query: The original query entered by the user.
        cache_hit: Whether the response was retrieved from the semantic cache.
        complexity_score: The complexity score assigned to the query.
            A value of -1.0 means complexity was not calculated because
            the response came from the cache.
        model_used: The source of the final response. This will be
            "cheap", "strong", or "cache".
        escalated: Whether the query was first sent to the cheap model
            and then escalated to the strong model.
        confidence: The confidence score reported by the model, if available.
        cost_usd: The total estimated cost of processing the query.
        latency_s: The total time spent waiting for model responses.
        response: The final response returned to the user.
    """

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
    """
    Controls the complete LLM routing process.

    The router first checks whether a similar query already exists in
    the semantic cache. If not, it calculates the query's complexity
    and uses that score to choose between the cheap and strong models.

    If the cheap model is selected but reports low confidence, the
    router escalates the query to the strong model.

    The final response is stored in the semantic cache for possible
    reuse on future queries.
    """
    def __init__(
        self,
        complexity_threshold: float = config.COMPLEXITY_ROUTE_THRESHOLD,
        confidence_threshold: float = config.CONFIDENCE_ESCALATION_THRESHOLD,
        cache_similarity_threshold: float = config.CACHE_SIMILARITY_THRESHOLD,
        use_cache: bool = True,
        ):
        """
        Initializes an LLMRouter.

        Args:
            complexity_threshold: Minimum complexity score required to
                route a query directly to the strong model.
            confidence_threshold: Minimum acceptable confidence from the
                cheap model. A lower confidence causes escalation.
            cache_similarity_threshold: Minimum similarity required for
                a previous query to count as a semantic cache hit.
            use_cache: Whether semantic caching should be enabled.
        """

        self.complexity_threshold = complexity_threshold
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        self.cache = SemanticCache(similarity_threshold=cache_similarity_threshold)

    def route(self, query: str) -> RouteTrace:
        """
        Processes and routes a user query.

        Args:
            query: The user's input query.

        Returns:
            A RouteTrace containing the final response and information
            about how the query was processed.
        """
        # Step 1: Check the semantic cache before calling an LLM.
        if self.use_cache:
            hit = self.cache.lookup(query)

            # If a sufficiently similar query exists, immediately return
            # its cached response without classifying or calling an LLM.
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

        # Step 2: Calculate the complexity of the query.
        complexity = classifier.score(query)

        escalated = False

        # Step 3: Route complex queries directly to strong or weaker model
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

                # Re-answer the query using the strong model.
                strong_result = clients.call_model("strong", query)
                total_cost = result.cost_usd + strong_result.cost_usd
                total_latency = result.latency_s + strong_result.latency_s
                result = strong_result
                result.cost_usd = total_cost
                result.latency_s = total_latency

        # Step 5: Store the final response in the semantic cache.
        # This may be an answer from either the cheap or strong model.
        if self.use_cache:
            self.cache.store(query, result.text, result.model_key)

        # Step 6: Return the final response along with a trace
        # describing how the router handled the query.
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
