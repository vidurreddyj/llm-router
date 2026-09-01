"""
Central configuration: which models are "cheap" vs "strong", their pricing,
and the thresholds that control routing/escalation/caching behavior.

Pricing is $ per 1M tokens (input, output) at Google's current *paid* list
rates -- used only to calculate what the run *would* have cost, so the
eval's cost numbers stay meaningful. Both models below are free to
actually call (within rate limits) via a Google AI Studio API key.

Model names and free-tier availability change often on Gemini -- if a
call fails with a 404/"no longer available" error, check
https://ai.google.dev/gemini-api/docs/models (or the error message
itself, which usually names the replacement model directly) and update
the "name" fields below.
"""

MODELS = {
    "cheap": {
        "name": "gemini-3.5-flash-lite",
        "price_in_per_m": 0.30,
        "price_out_per_m": 2.50,
    },
    "strong": {
        "name": "gemini-3.5-flash",
        "price_in_per_m": 1.50,
        "price_out_per_m": 9.00,
    },
    "judge": {
        "name": "gemini-3.1-flash-lite",
        "price_in_per_m": 0.25,
        "price_out_per_m": 1.50,
    },
}

# Complexity score (0-1) at or above which a query is routed straight to
# the strong model instead of the cheap one.
COMPLEXITY_ROUTE_THRESHOLD = 0.55

# If the cheap model self-reports a confidence below this, escalate the
# query to the strong model and use that answer instead.
CONFIDENCE_ESCALATION_THRESHOLD = 0.6

# Cosine similarity above which a new query is considered a "hit" against
# something already in the semantic cache.
CACHE_SIMILARITY_THRESHOLD = 0.90

MAX_TOKENS = 512
