"""
Complexity classifier for the LLM Router.

This module estimates how difficult a user query is before sending it
to an LLM. The router uses the resulting complexity score to decide
whether the query should be handled by the cheap model or the strong model.

The classifier is heuristic-based, meaning it uses manually defined rules
such as:
    - Query length
    - Words associated with difficult tasks
    - Words associated with simple tasks
    - Presence of code
    - Presence of math
    - Multiple questions

The score ranges from 0.0 to 1.0:
    Lower score -> simpler query
    Higher score -> more complex query
"""

import re

# Regular expression patterns that suggest a query requires deeper reasoning.
# Examples include requests to prove something, debug code, design a system,
# compare ideas, or write an algorithm.
HARD_SIGNALS = [
    r"\bprove\b", r"\bderive\b", r"\bdebug\b", r"\boptimi[sz]e\b",
    r"\barchitecture\b", r"\bdesign\b", r"\btrade[- ]?off\b",
    r"\bwhy does\b", r"\bstep[- ]by[- ]step\b", r"\bcompare\b",
    r"\bwrite (a|an) (function|algorithm|program)\b",
    r"\bedge case\b", r"\bcomplexity\b", r"\bproof\b",
]

# Regular expression patterns that suggest a query is relatively simple.
# These are commonly associated with definitions, basic lookups,
# formatting, conversion, or simple conversation.
EASY_SIGNALS = [
    r"\bwhat is\b", r"\bdefine\b", r"\btranslate\b", r"\bspell\b",
    r"\bsummarize\b(?! .*\band\b.*\banalyze\b)", r"\bcapital of\b",
    r"^hi\b", r"^hello\b", r"\bformat\b", r"\bconvert\b",
]

# Detects common signs that the query contains programming code.
# For example:
#     def calculate():
#     class Student:
#     import math
#
# Queries containing code receive a higher complexity score.
CODE_BLOCK_RE = re.compile(r"```|def |class |function\(|import ")

# Detects common signs of mathematical expressions or math-related tasks.
#
# This includes mathematical operators, numbers, fractions, sums,
# integrals, and derivatives.
MATH_RE = re.compile(r"[=+\-*/^]{1}.*\d|\\frac|\\sum|integral|derivative")


def score(query: str) -> float:
    """
    Calculates a complexity score for a user query.

    The score is determined using several heuristic signals:
        1. Query length
        2. Presence of hard-task keywords
        3. Presence of easy-task keywords
        4. Presence of programming code
        5. Presence of mathematical expressions
        6. Whether the query contains multiple parts

    The router can compare this score against its configured complexity
    threshold to decide whether to use the cheap or strong LLM.

    Args:
        query: The user's input query.

    Returns:
        A float between 0.0 and 1.0 representing query complexity.
        Higher values indicate that the query is likely to require
        a stronger model.
    """
    q = query.strip().lower()
    if not q:
        return 0.0

    # Stores adjustments to the query's complexity score.
    s = 0.0

    # LENGTH CHECK
    # Longer queries are treated as somewhat more complex because
    # they often contain more instructions or information.
    word_count = len(q.split())
    if word_count > 60:
        s += 0.25
    elif word_count > 25:
        s += 0.12

    # HARD SIGNAL CHECK
    # Search for words or phrases associated with difficult tasks.
    # If one is found, increase complexity by 0.20.
    #
    # The loop stops after the first match so a query containing
    # several hard keywords is not given repeated +0.20 increases.
    for pattern in HARD_SIGNALS:
        if re.search(pattern, q):
            s += 0.20
            break  # don't let repeated matches blow the score up

    # EASY SIGNAL CHECK
    # Search for words or phrases associated with simple tasks.
    # If one is found, reduce complexity by 0.20.
    for pattern in EASY_SIGNALS:
        if re.search(pattern, q):
            s -= 0.20
            break

    # CODE CHECK
    if CODE_BLOCK_RE.search(query):
        s += 0.25

    # MATH CHECK
    if MATH_RE.search(q):
        s += 0.15

    # MULTI-PART QUESTION CHECK
    # Queries containing multiple questions or longer requests joined
    # with "and" may require more reasoning to answer completely.
    if q.count("?") > 1 or " and " in q and word_count > 15:
        s += 0.1

    # FINAL SCORE
    # Start with a baseline complexity of 0.35 and add all adjustments. Stays between 0.0 and 1.0
    return max(0.0, min(1.0, s + 0.35))  # 0.35 baseline so score is centered
