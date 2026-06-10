"""Hybrid retrieval: vector search + BM25 keyword search with RRF fusion.

Provides :class:`HybridRetriever` which combines:

1. **Dense / vector** retrieval from the ChromaDB vector store using
   the same BGE-M3 embeddings as ingest.  Supports multi-query
   expansion via :class:`QueryRewriter` — the original query plus
   rewritten variants are each embedded and searched, with results
   merged by best similarity per chunk.

2. **Sparse / keyword** retrieval using a self-implemented BM25
   scorer with jieba tokenization for Chinese text.  The BM25 index
   is built (and can be rebuilt) from the full corpus held in the
   vector store.

3. **Fusion** via Reciprocal Rank Fusion (RRF) — no fragile score
   normalisation needed because RRF only considers result *ranks*.

Usage::

    from src.retrieval.retriever import HybridRetriever
    from src.retrieval.query_rewriter import QueryRewriter
    from src.knowledge.store import VectorStore
    from src.knowledge.embedder import Embedder

    retriever = HybridRetriever(
        vector_store=VectorStore(),
        embedder=Embedder(),
        query_rewriter=QueryRewriter(),
    )
    results = retriever.retrieve("YAG激光器怎么选型", top_k=5)
    for r in results:
        print(f"{r.score:.4f}  {r.text[:80]}...")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import jieba

if TYPE_CHECKING:
    from config import Settings
    from src.knowledge.embedder import Embedder
    from src.knowledge.store import VectorStore
    from src.retrieval.query_rewriter import QueryRewriter

logger = logging.getLogger("laser-rag.retriever")

# ---------------------------------------------------------------------------
#  output dataclass
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """A single chunk returned by the hybrid retriever.

    Attributes
    ----------
    score:        Fused RRF score (higher = better).
    vector_score: Cosine similarity from vector search (``None`` if the
                  chunk only appeared in keyword results).
    keyword_score: BM25 score from keyword search (``None`` if the
                   chunk only appeared in vector results).
    """

    id: str
    text: str
    filename: str = ""
    chunk_id: int = 0
    page: int | None = None
    score: float = 0.0          # fused / final score
    vector_score: float | None = None
    keyword_score: float | None = None
    metadata: dict = field(default_factory=dict)


# ===========================================================================
#  BM25 keyword searcher (jieba tokenization)
# ===========================================================================

class BM25Searcher:
    """Lightweight BM25 keyword search with jieba Chinese tokenization.

    Parameters
    ----------
    k1: Term-frequency saturation parameter (default 1.5).
    b:  Document-length normalisation parameter (default 0.75).

    Usage::

        bm25 = BM25Searcher()
        bm25.index([{"id": "a", "text": "..."}, {"id": "b", "text": "..."}])
        hits = bm25.search("激光器 选型", top_k=10)
        # → [(corpus_idx, bm25_score), ...]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        # corpus state — populated by .index()
        self._corpus: list[dict] = []
        self._tokenized_docs: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._doc_count: int = 0

    # ------------------------------------------------------------------
    #  index
    # ------------------------------------------------------------------
    def index(self, corpus: list[dict]) -> None:
        """Build / replace the BM25 index from *corpus*.

        Each dict must have at least ``"id"`` and ``"text"`` keys.
        """
        self._corpus = corpus
        self._doc_count = len(corpus)
        if self._doc_count == 0:
            self._tokenized_docs = []
            self._idf = {}
            self._avgdl = 0.0
            logger.info("BM25 index: empty corpus")
            return

        self._tokenized_docs = [self._tokenize(d["text"]) for d in corpus]
        self._avgdl = sum(len(t) for t in self._tokenized_docs) / self._doc_count
        self._idf = self._compute_idf()

        logger.info(
            "BM25 index built: docs=%d avg_len=%.1f vocab=%d",
            self._doc_count,
            self._avgdl,
            len(self._idf),
        )

    # ------------------------------------------------------------------
    #  search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``[(corpus_index, bm25_score), ...]`` sorted by score desc.

        Only entries with a score > 0 are returned.
        """
        if not self._corpus or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = []
        for doc_tokens in self._tokenized_docs:
            scores.append(self._bm25_score(query_tokens, doc_tokens))

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, s) for idx, s in indexed[:top_k] if s > 0]

    @property
    def is_empty(self) -> bool:
        return self._doc_count == 0

    # ------------------------------------------------------------------
    #  internal
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize *text* with jieba, dropping single-char and blank tokens."""
        tokens = jieba.lcut(text)
        return [t.strip() for t in tokens if len(t.strip()) >= 2]

    def _compute_idf(self) -> dict[str, float]:
        """Compute smoothed IDF for every term in the corpus."""
        df: dict[str, int] = {}
        for tokens in self._tokenized_docs:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        N = self._doc_count
        # BM25 smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        return {
            term: math.log((N - cnt + 0.5) / (cnt + 0.5) + 1.0)
            for term, cnt in df.items()
        }

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """Compute BM25 score for a single document."""
        score = 0.0
        doc_len = len(doc_tokens)
        dl_norm = doc_len / max(1.0, self._avgdl)

        # term frequency for this document
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        for term in query_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            f = tf.get(term, 0)
            numerator = f * (self.k1 + 1.0)
            denominator = f + self.k1 * (1.0 - self.b + self.b * dl_norm)
            score += idf * numerator / denominator

        return score


# ===========================================================================
#  HybridRetriever
# ===========================================================================

class HybridRetriever:
    """Hybrid retrieval: dense (vector) + sparse (BM25) → RRF fusion.

    Parameters
    ----------
    vector_store:  :class:`VectorStore` instance for dense retrieval.
    embedder:      :class:`Embedder` instance for query embedding.
    query_rewriter: Optional :class:`QueryRewriter` for multi-query expansion.
    rrf_k:         RRF constant — larger values flatten rank differences
                   (default 60, the standard choice).
    expand:        Multiplier applied to ``top_k`` for the initial
                   retrieval before fusion — e.g. with *expand* = 3
                   and ``top_k`` = 5, each sub-retriever fetches 15
                   candidates (default from ``RETRIEVAL_HYBRID_EXPAND``).
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        query_rewriter: QueryRewriter | None = None,
        *,
        rrf_k: int | None = None,
        expand: int | None = None,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        # lazy imports avoid circular dependency at module level
        if vector_store is None:
            from src.knowledge.store import VectorStore
            vector_store = VectorStore()
        if embedder is None:
            from src.knowledge.embedder import Embedder
            embedder = Embedder()

        self._store = vector_store
        self._embedder = embedder
        self._rewriter = query_rewriter

        self._rrf_k = rrf_k if rrf_k is not None else self._s.retrieval_hybrid_rrf_k
        self._expand = expand if expand is not None else self._s.retrieval_hybrid_expand

        # BM25 index — built lazily on first retrieve()
        self._bm25 = BM25Searcher()
        self._corpus: list[dict] = []       # parallel to BM25._corpus
        self._id_to_index: dict[str, int] = {}  # chunk_id → corpus index
        self._index_loaded: bool = False

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        use_rewriter: bool = True,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval for *query*.

        Parameters
        ----------
        query:         Natural-language query from the user.
        top_k:         Number of final results (default from config).
        use_rewriter:  If ``True`` and a :class:`QueryRewriter` was
                       provided, expand the query before retrieval.

        Returns
        -------
        list[RetrievalResult]
            Top-*k* chunks sorted by fused RRF score (descending).
            Returns an empty list when the index is empty or *query*
            is blank.
        """
        if top_k is None:
            top_k = self._s.retrieval_top_k
        if not query.strip():
            return []

        self._ensure_index()
        if not self._corpus:
            logger.warning("Retrieve called on empty corpus")
            return []

        # how many candidates to pull from each retriever
        candidate_k = top_k * self._expand

        # ---- 1. vector search (multi-query) ---------------------------
        queries = [query]
        if use_rewriter and self._rewriter is not None:
            try:
                queries = self._rewriter.rewrite(query)
            except Exception:
                logger.exception("Query rewriting failed — using original only")

        vector_ranked = self._vector_search(queries, candidate_k)

        # ---- 2. keyword search -----------------------------------------
        keyword_ranked = self._keyword_search(query, candidate_k)

        # ---- 3. RRF fusion --------------------------------------------
        fused = self._rrf_fusion(vector_ranked, keyword_ranked)

        # ---- 4. assemble results --------------------------------------
        results: list[RetrievalResult] = []
        vec_map = {idx: score for idx, score in vector_ranked}
        kw_map = {idx: score for idx, score in keyword_ranked}

        for idx, fused_score in fused[:top_k]:
            doc = self._corpus[idx]
            results.append(
                RetrievalResult(
                    id=doc["id"],
                    text=doc["text"],
                    filename=doc.get("filename", ""),
                    chunk_id=doc.get("chunk_id", 0),
                    page=doc.get("page"),
                    score=round(fused_score, 6),
                    vector_score=round(vec_map.get(idx), 6) if idx in vec_map else None,
                    keyword_score=round(kw_map.get(idx), 6) if idx in kw_map else None,
                    metadata=doc.get("metadata", {}),
                )
            )

        logger.info(
            "Hybrid retrieve %r → %d results (vec=%d kw=%d fused=%d)",
            query[:60],
            len(results),
            len(vector_ranked),
            len(keyword_ranked),
            len(fused),
        )
        return results

    def rebuild_index(self) -> None:
        """Rebuild the BM25 index from the current vector-store contents.

        Call this after ingesting new documents so that keyword
        search stays in sync with the vector store.
        """
        self._index_loaded = False
        self._ensure_index()
        logger.info("Hybrid index rebuilt — %d document(s)", len(self._corpus))

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the combined index."""
        self._ensure_index()
        return len(self._corpus)

    # ------------------------------------------------------------------
    #  index management
    # ------------------------------------------------------------------
    def _ensure_index(self) -> None:
        """Load corpus from the vector store and build BM25 if needed."""
        if self._index_loaded:
            return
        self._load_corpus()
        self._bm25.index(self._corpus)
        self._index_loaded = True

    def _load_corpus(self) -> None:
        """Pull all documents from ChromaDB into in-memory corpus."""
        raw = self._store._collection.get(
            include=["documents", "metadatas"],
        )
        ids: list[str] = raw.get("ids", [])
        docs: list[str] = raw.get("documents", [])
        metas: list[dict] = raw.get("metadatas", [])

        self._corpus = []
        self._id_to_index = {}

        for i, doc_id in enumerate(ids):
            meta = metas[i] if metas and i < len(metas) else {}
            self._corpus.append({
                "id": doc_id,
                "text": docs[i] if docs and i < len(docs) else "",
                "filename": str(meta.get("filename", "")),
                "chunk_id": int(meta.get("chunk_id", 0)),
                "page": meta.get("page"),
                "metadata": {
                    k: v for k, v in meta.items()
                    if k not in ("filename", "chunk_id", "page")
                },
            })
            self._id_to_index[doc_id] = i

        logger.info("Corpus loaded from store: %d chunk(s)", len(self._corpus))

    # ------------------------------------------------------------------
    #  vector search
    # ------------------------------------------------------------------
    def _vector_search(
        self, queries: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """Multi-query dense retrieval → ``[(corpus_idx, similarity), ...]``.

        Each query in *queries* is embedded and searched independently.
        When a chunk appears in multiple result sets the **best**
        similarity is kept.
        """
        best_sim: dict[int, float] = {}

        for q in queries:
            vectors = self._embedder.embed([q])
            if not vectors:
                continue

            raw_results = self._store.query(vectors[0], top_k=top_k)
            for r in raw_results:
                idx = self._id_to_index.get(r.id)
                if idx is None:
                    continue
                if idx not in best_sim or r.similarity > best_sim[idx]:
                    best_sim[idx] = r.similarity

        return sorted(best_sim.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    #  keyword search
    # ------------------------------------------------------------------
    def _keyword_search(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 sparse retrieval → ``[(corpus_idx, bm25_score), ...]``."""
        return self._bm25.search(query, top_k)

    # ------------------------------------------------------------------
    #  RRF fusion
    # ------------------------------------------------------------------
    def _rrf_fusion(
        self,
        vector_ranked: list[tuple[int, float]],
        keyword_ranked: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: combine two ranked lists into one.

        .. math::
            RRF(d) = \\sum_{r \\in R} \\frac{1}{k + rank_r(d)}

        where *k* is the RRF constant (default 60) and *rank_r(d)* is
        the 1-based rank of document *d* in result list *r*.
        """
        scores: dict[int, float] = {}
        k = float(self._rrf_k)

        for rank, (idx, _score) in enumerate(vector_ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

        for rank, (idx, _score) in enumerate(keyword_ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
