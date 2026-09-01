"""
Complexity classifier: scores an incoming query from 0 (trivial) to 1 (hard),
without calling any LLM. This is what decides which model the router
sends the request to.

This is a heuristic (rule-based) classifier -- fast, free, and a reasonable
v1. A natural v2 upgrade is to replace `score()` with a small trained
classifier (e.g. logistic regression over embeddings, or few-shot labels
scored by the cheap model itself) without touching the rest of the router.
"""

import re

# Signals that a query likely needs deeper reasoning.
HARD_SIGNALS = [
    r"\bprove\b", r"\bderive\b", r"\bdebug\b", r"\boptimi[sz]e\b",
    r"\barchitecture\b", r"\bdesign\b", r"\btrade[- ]?off\b",
    r"\bwhy does\b", r"\bstep[- ]by[- ]step\b", r"\bcompare\b",
    r"\bwrite (a|an) (function|algorithm|program)\b",
    r"\bedge case\b", r"\bcomplexity\b", r"\bproof\b",
]

# Signals that a query is likely simple lookup/formatting/chit-chat.
EASY_SIGNALS = [
    r"\bwhat is\b", r"\bdefine\b", r"\btranslate\b", r"\bspell\b",
    r"\bsummarize\b(?! .*\band\b.*\banalyze\b)", r"\bcapital of\b",
    r"^hi\b", r"^hello\b", r"\bformat\b", r"\bconvert\b",
]

CODE_BLOCK_RE = re.compile(r"```|def |class |function\(|import ")
MATH_RE = re.compile(r"[=+\-*/^]{1}.*\d|\\frac|\\sum|integral|derivative")


def score(query: str) -> float:
    """Return a complexity score in [0, 1]. Higher = harder / needs a
    stronger model."""
    q = query.strip().lower()
    if not q:
        return 0.0

    s = 0.0

    # Length is a weak but useful signal -- longer asks tend to be more
    # involved than one-liners.
    word_count = len(q.split())
    if word_count > 60:
        s += 0.25
    elif word_count > 25:
        s += 0.12

    for pattern in HARD_SIGNALS:
        if re.search(pattern, q):
            s += 0.20
            break  # don't let repeated matches blow the score up

    for pattern in EASY_SIGNALS:
        if re.search(pattern, q):
            s -= 0.20
            break

    if CODE_BLOCK_RE.search(query):
        s += 0.25

    if MATH_RE.search(q):
        s += 0.15

    # Multi-part questions ("...and also...", multiple "?") tend to need
    # more reasoning to answer completely.
    if q.count("?") > 1 or " and " in q and word_count > 15:
        s += 0.1

    return max(0.0, min(1.0, s + 0.35))  # 0.35 baseline so score is centered
