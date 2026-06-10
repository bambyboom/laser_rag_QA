from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import chromadb
from chromadb.api.types import Metadata, OneOrMany

from src.knowledge.embedder import EmbeddedChunk

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("laser-rag.store")


# ---------------------------------------------------------------------------
#  query output
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Single retrieval result from a vector search."""

    id: str
    text: str
    filename: str = ""
    chunk_id: int = 0
    page: int | None = None
    distance: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def similarity(self) -> float:
        """Cosine similarity = 1 − cosine distance."""
        return max(0.0, 1.0 - self.distance)


# ---------------------------------------------------------------------------
#  VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """Persistent ChromaDB wrapper with cosine similarity.

    Parameters
    ----------
    collection_name: ChromaDB collection name (default from config).
    persist_dir: Directory for ChromaDB's on-disk storage.

    Usage::

        store = VectorStore()
        store.add(embedded_chunks)
        results = store.query(embedding, top_k=5)
    """

    def __init__(
        self,
        *,
        collection_name: str | None = None,
        persist_dir: Path | str | None = None,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        _name = collection_name or self._s.chroma_collection_name
        _path = str(persist_dir or self._s.chroma_persist_dir)

        Path(_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=_path)
        self._collection = self._client.get_or_create_collection(
            name=_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "VectorStore ready — collection=%r path=%s count=%d",
            _name,
            _path,
            self.count(),
        )

    # ------------------------------------------------------------------
    #  add
    # ------------------------------------------------------------------
    def add(self, chunks: Sequence[EmbeddedChunk]) -> int:
        """Insert embedded chunks into the collection.

        Returns the number of chunks added.  Safe to call multiple
        times — duplicate IDs overwrite the previous entry (upsert).
        """
        if not chunks:
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, object]] = []

        for c in chunks:
            ids.append(c.id)
            embeddings.append(c.embedding)
            documents.append(c.text)

            meta: dict[str, object] = {
                "filename": c.filename,
                "chunk_id": c.chunk_id,
            }
            if c.page is not None:
                meta["page"] = c.page
            meta.update(c.metadata)
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Added %d chunk(s) — collection now has %d", len(ids), self.count())
        return len(ids)

    # ------------------------------------------------------------------
    #  query
    # ------------------------------------------------------------------
    def query(
        self,
        embedding: list[float],
        top_k: int | None = None,
    ) -> list[QueryResult]:
        """Return the *top_k* most similar chunks for *embedding*.

        Parameters
        ----------
        embedding: Query vector (same dimension as stored embeddings).
        top_k: Number of results (default from ``RETRIEVAL_TOP_K`` config).
        """
        if top_k is None:
            top_k = self._s.retrieval_top_k

        if self.count() == 0:
            logger.warning("Query on empty collection")
            return []

        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: list[QueryResult] = []
        # ChromaDB returns lists-of-lists (one inner list per query vector)
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            meta = metas[i] if metas and i < len(metas) else {}
            results.append(
                QueryResult(
                    id=doc_id,
                    text=docs[i] if docs and i < len(docs) else "",
                    filename=str(meta.get("filename", "")),
                    chunk_id=int(meta.get("chunk_id", 0)),
                    page=meta.get("page"),
                    distance=float(dists[i]) if dists and i < len(dists) else 0.0,
                    metadata={k: v for k, v in meta.items()
                              if k not in ("filename", "chunk_id", "page")},
                )
            )

        return results

    # ------------------------------------------------------------------
    #  collection management
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Number of documents currently stored."""
        return self._collection.count()

    def clear(self) -> None:
        """Remove all documents from the collection (keeps schema)."""
        before = self.count()
        if before == 0:
            return
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
        logger.info("Cleared %d document(s) from collection", before)

    def reset(self) -> None:
        """Delete the collection entirely and recreate it fresh."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection %r reset", name)

    def delete_by_filename(self, filename: str) -> int:
        """Remove all chunks originating from *filename*.  Returns count deleted."""
        before = self.count()
        self._collection.delete(where={"filename": filename})
        after = self.count()
        removed = before - after
        if removed > 0:
            logger.info("Deleted %d chunk(s) from %r", removed, filename)
        return removed

    def get_filenames(self) -> list[str]:
        """Return deduplicated list of filenames currently indexed."""
        all_meta = self._collection.get(include=["metadatas"])
        metadatas = all_meta.get("metadatas", [])
        filenames: set[str] = set()
        for m in metadatas:
            if m and "filename" in m:
                filenames.add(str(m["filename"]))
        return sorted(filenames)
