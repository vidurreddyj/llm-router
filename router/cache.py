"""
Semantic cache: instead of exact-string matching, embed each query and
return a cached response for any new query that's similar enough to one
we've already answered. This is where a big chunk of real-world cost
savings comes from, since production traffic is full of near-duplicate
questions phrased slightly differently.

This implementation uses TF-IDF + cosine similarity so it has zero
external dependencies and zero latency cost -- good for a resume project
and for understanding the mechanics. For production quality, swap
`SemanticCache` to embed queries with a real embedding model (e.g.
sentence-transformers or an embeddings API) and store vectors in a proper
vector index (FAISS, Chroma, pgvector) instead of the in-memory matrix
recomputed here.
"""

from dataclasses import dataclass
from typing import Optional

import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CacheEntry:
    query: str
    response: str
    model_used: str


class SemanticCache:
    """
    NOTE on the vectorizer choice: an earlier version of this cache used
    TfidfVectorizer re-fit on only the cached queries. That has a subtle
    but serious bug -- if a new query contains a word never seen before
    (out-of-vocabulary), that word is silently dropped from its vector
    instead of being counted as "different." In testing, this made
    "capital of Germany" register as a 0.91 similarity match against a
    cached "capital of France" entry (the differentiating word just
    vanished, leaving only shared stopwords). A HashingVectorizer with a
    large fixed feature space avoids this: every word -- seen before or
    not -- deterministically hashes into the same fixed-size vector space,
    so new/unseen words still count as real signal that pulls similarity
    down. This is also closer to how you'd want caching to behave in
    production, where you can't keep re-fitting a vocabulary as traffic
    grows.
    """

    def __init__(self, similarity_threshold: float = 0.90, n_features: int = 2**14):
        self.similarity_threshold = similarity_threshold
        self.entries: list[CacheEntry] = []
        self._hasher = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None
        )
        self._tfidf = TfidfTransformer()
        self._matrix = None  # fitted lazily once we have >=1 entry
        self.hits = 0
        self.misses = 0

    def _vectorize_all(self):
        counts = self._hasher.transform([e.query for e in self.entries])
        self._matrix = self._tfidf.fit_transform(counts)

    def lookup(self, query: str) -> Optional[CacheEntry]:
        if not self.entries:
            self.misses += 1
            return None

        vec = self._tfidf.transform(self._hasher.transform([query]))
        sims = cosine_similarity(vec, self._matrix)[0]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])

        if best_sim >= self.similarity_threshold:
            self.hits += 1
            return self.entries[best_idx]

        self.misses += 1
        return None

    def store(self, query: str, response: str, model_used: str):
        self.entries.append(CacheEntry(query, response, model_used))
        self._vectorize_all()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "entries": len(self.entries),
        }
