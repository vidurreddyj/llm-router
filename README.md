# LLM Router

An intelligent LLM routing system that reduces API costs by dynamically sending each request to the **cheapest model capable of handling it effectively**, rather than routing every request to the most expensive model.

The system combines **semantic caching, query complexity classification, dynamic model routing, confidence-based escalation, and automated evaluation** to balance **cost, latency, and response quality**.

---

## Overview

Using a powerful LLM for every request can be unnecessarily expensive. Many queries are simple enough to be handled by a smaller, cheaper model, while more complex queries benefit from a stronger model.

LLM Router automatically decides which model should handle each request.

```text
                     ┌───────────────┐
                     │     Query     │
                     └───────┬───────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Semantic Cache  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                  Cache             Cache
                   Hit               Miss
                    │                 │
                    ▼                 ▼
                 Return      ┌─────────────────┐
                 Cached      │ Complexity Check │
                 Answer      └────────┬────────┘
                                      │
                              ┌───────┴───────┐
                              │               │
                            Easy             Hard
                              │               │
                              ▼               ▼
                        Cheap Model      Strong Model
                              │
                              ▼
                       Confidence Check
                              │
                       Low confidence?
                              │
                             Yes
                              ▼
                        Strong Model
                              │
                              ▼
                         Cache Result
                              │
                              ▼
                            Return
```

---

## Features

### Semantic Cache

Before making an API request, the router checks whether a sufficiently similar query has already been answered.

If a match is found, the cached response is returned immediately, avoiding another LLM call.

This reduces:

- API usage
- Response latency
- Cost for repeated or similar requests

---

### Complexity Classifier

Each incoming query is classified as **easy** or **hard** using lightweight heuristics.

The classifier does not require an additional LLM call, allowing the routing decision itself to remain fast and inexpensive.

Examples:

```text
"What is the capital of France?"
        ↓
      EASY
        ↓
   Cheap Model
```

```text
"Compare optimistic and pessimistic concurrency control
and explain when each should be used."
        ↓
       HARD
        ↓
   Strong Model
```

---

### Dynamic Model Routing

After classification, the router selects the appropriate model:

```text
Easy query → Cheap model

Hard query → Strong model
```

This avoids paying for a stronger model when a cheaper model is sufficient.

---

### Confidence-Based Escalation

Routing is not necessarily final.

If the cheap model produces a response with insufficient confidence, the request can be **escalated to the stronger model**.

```text
Easy Query
    ↓
Cheap Model
    ↓
Confidence Check
    ↓
Low Confidence
    ↓
Strong Model
```

This provides a fallback mechanism for queries that initially appear simple but prove more difficult to answer reliably.

---

### Evaluation Harness

The project includes an automated evaluation pipeline that compares three routing strategies:

1. **Always Cheap** — every query uses the inexpensive model
2. **Always Strong** — every query uses the stronger model
3. **Router** — dynamically selects the model and uses caching/escalation

Each strategy is evaluated across:

- API cost
- Average latency
- Response quality
- Routing decisions
- Cache behavior

Run the evaluation with:

```bash
python -m eval.run_eval
```

Evaluation results are saved under:

```text
eval/results/
```

---

## Current Evaluation Results

The current benchmark contains **7 test queries**.

| Strategy       |         Cost | Avg. Quality | Avg. Latency |
| -------------- | -----------: | -----------: | -----------: |
| Always Cheap   |     $0.00440 |        0.771 |        5.14s |
| Always Strong  |     $0.00106 |        0.600 |        5.30s |
| **LLM Router** | **$0.00084** |    **0.571** |    **3.64s** |

In this evaluation, the router:

- Reduced cost by approximately **20.8% compared with always using the strong model**
- Correctly routed the benchmark queries according to their expected paths
- Sent easy queries to the cheap model
- Sent hard queries to the strong model
- Served duplicate queries through the cache
- Achieved lower average latency in this particular run

### Evaluation Limitation

The **quality scores should be interpreted cautiously**.

The current benchmark contains only **7 queries**, which is too small to make a statistically meaningful comparison of response quality. Quality scores varied noticeably between evaluation runs, and the current results should therefore be treated as an early functional benchmark rather than evidence that one routing strategy consistently produces better responses.

A more reliable evaluation would use **at least 25+ diverse queries**, ideally more, covering different difficulty levels and query categories.

The benchmark is currently kept small because the evaluation executes queries across multiple routing strategies and may also require additional LLM-based evaluation calls. Expanding from 7 to 25+ queries would therefore consume substantially more of the available **free API quota**.

Future evaluation work should expand the benchmark when additional API capacity is available.

---

## Setup

The project uses the **Google Gemini API**.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Create a Gemini API key through Google AI Studio.

Then set it as an environment variable.

**macOS / Linux**

```bash
export GOOGLE_API_KEY="your-api-key"
```

**Windows PowerShell**

```powershell
$env:GOOGLE_API_KEY="your-api-key"
```

The model configuration is located in:

```text
router/config.py
```

Gemini model IDs may change or be retired over time. If the API returns an error indicating that a model is unavailable, update the corresponding model ID in `router/config.py`.

---

## Usage

```python
from router import LLMRouter

router = LLMRouter()

trace = router.route(
    "What is the capital of France?"
)

print(trace.model_used)
print(trace.cost_usd)
print(trace.response)
```

Example routing result:

```text
model_used: cheap
```

Repeated or sufficiently similar requests may instead be returned directly from the semantic cache.

---

## Project Structure

```text
llm-router/
│
├── router/
│   ├── config.py
│   ├── classifier.py
│   ├── cache.py
│   ├── clients.py
│   └── core.py
│
├── eval/
│   ├── benchmark.jsonl
│   ├── run_eval.py
│   └── results/
│
├── requirements.txt
└── README.md
```

### Core Components

**`config.py`**
Model configuration, routing thresholds, and system settings.

**`classifier.py`**
Lightweight query-complexity classifier used to determine whether a request should use the cheap or strong model.

**`cache.py`**
Semantic caching layer for detecting and reusing answers to similar queries.

**`clients.py`**
Wrapper around the Gemini API and model interactions.

**`core.py`**
Main routing pipeline that connects classification, caching, model selection, escalation, and response tracing.

**`benchmark.jsonl`**
Evaluation dataset containing queries used to test routing behavior.

**`run_eval.py`**
Runs the benchmark against the always-cheap, always-strong, and dynamic-routing strategies and records their results.

---

## Design Goals

The project explores a practical problem in production LLM systems:

> **How can an application minimize inference cost without unnecessarily sacrificing response quality?**

Instead of treating model selection as a fixed configuration, the router treats it as a runtime decision.

The current implementation focuses on four optimization mechanisms:

```text
Caching
   +
Complexity Classification
   +
Dynamic Model Selection
   +
Confidence-Based Escalation
```

Together, these mechanisms provide a foundation for experimenting with **cost-aware LLM inference**.

---

## Future Improvements

Potential extensions include:

- Replace TF-IDF similarity with embedding-based semantic caching
- Train a learned complexity classifier instead of relying only on heuristics
- Expand the evaluation benchmark to **25+ queries**
- Evaluate performance across additional query categories and difficulty levels
- Add more models and support multi-tier routing
- Improve confidence calibration for escalation decisions
- Add persistent cache storage
- Implement rate limiting
- Add fair-share request queuing across users
- Track token usage and routing metrics over time
- Add configurable cost/quality optimization policies

---
