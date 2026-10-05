**LLM Cost Router**

An LLM routing system that reduces the cost of answering prompts by deciding when a cheaper model is sufficient and when a stronger model is needed.

The router combines semantic caching, heuristic complexity classification, and confidence-based escalation to avoid unnecessary calls to more expensive models.

**Running the Project**

Install the dependencies:

pip install -r requirements.txt

Set your Gemini API key.

PowerShell:

$env:GOOGLE_API_KEY="YOUR_API_KEY"

Then run:

python chat.py

Enter queries directly in the terminal. Type exit to stop the program.

## Running the Evaluation

The project includes a benchmark system for evaluating the router's performance. Test queries are stored in `eval/benchmark.jsonl`.

### 1. Navigate to the project folder

```powershell
cd C:\Users\HP\Downloads\llm-router
```

### 2. Set your Gemini API key

```powershell
$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

Replace `YOUR_API_KEY_HERE` with your Google Gemini API key.

### 3. Run the evaluation

```powershell
python -m eval.run_eval
```

The evaluation reads the predefined queries from `benchmark.jsonl` and tests the router's performance.

After the evaluation finishes, it generates:

- `results.json` — stores the numerical evaluation results, including metrics such as cost, latency, quality, routing decisions, and cache performance.
- `pareto.png` — visualizes the cost-versus-quality trade-off between the evaluated routing strategies.

To test different queries, edit the queries in `eval/benchmark.jsonl` and run the evaluation again.

**Current Constraints**

Gemini Free Tier

This project currently uses Gemini Flash-family models for both the cheap and strong routing options.

The project is being developed using Google's Gemini API free tier, which has more restrictive model availability and rate limits than paid usage. Because of this, the current "strong" model should be understood as stronger relative to the cheap model used by this prototype, rather than the strongest Gemini model available.

The routing architecture is model-independent enough that stronger or more expensive models could be substituted when higher API tiers are available.

Output Token Limit

Responses currently have a maximum output length of:

MAX_TOKENS = 2048

The limit helps control response length and token usage, which is especially important for a project focused on reducing LLM cost.

Very large questions may occasionally require more than this limit. The models are therefore prompted to provide complete but concise answers.

**Evaluation**

The router can also be evaluated using the predefined queries in:

eval/benchmark.jsonl

Run:

python -m eval.run_eval

The evaluation compares different routing strategies and measures factors such as cost, latency, quality, cache usage, and model selection.

**Goal**

The goal of the project is not simply to always use the cheapest model. Instead, it aims to use the least expensive option that can still provide an acceptable answer, while escalating difficult queries when necessary.
