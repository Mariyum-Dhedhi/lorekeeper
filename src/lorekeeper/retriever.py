"""Hybrid retrieval: dense (Chroma + local fastembed) fused with BM25 via reciprocal rank fusion."""

import json
import re
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from lorekeeper.config import CACHE_DIR, INDEX_DIR, get_settings


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[Document]: ...


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings, free, no torch needed."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name, cache_dir=str(CACHE_DIR / "fastembed"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


_embeddings: FastEmbedEmbeddings | None = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(get_settings().embedding_model)
    return _embeddings


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge several ranked id lists. Items ranked high in any list float to the top."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR, k: int | None = None, hybrid: bool | None = None):
        from langchain_chroma import Chroma
        from rank_bm25 import BM25Okapi

        from lorekeeper.ingest import CHUNKS_FILE, COLLECTION

        s = get_settings()
        self.k = k or s.top_k
        self.hybrid = s.hybrid if hybrid is None else hybrid

        raw = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        self.chunks = {r["metadata"]["id"]: Document(page_content=r["text"], metadata=r["metadata"]) for r in raw}
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(self.chunks[i].page_content) for i in self.ids])
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(index_dir / "chroma"),
        )

    def dense_ids(self, query: str, n: int) -> list[str]:
        return [d.metadata["id"] for d in self.store.similarity_search(query, k=n)]

    def bm25_ids(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:n] if scores[i] > 0]

    def retrieve(self, query: str) -> list[Document]:
        pool = self.k * 3
        if self.hybrid:
            ids = reciprocal_rank_fusion([self.dense_ids(query, pool), self.bm25_ids(query, pool)])
        else:
            ids = self.dense_ids(query, self.k)
        return [self.chunks[i] for i in ids[: self.k]]


def load_retriever(k: int | None = None, hybrid: bool | None = None) -> HybridRetriever:
    """Load the index, building it first if it doesn't exist yet."""
    from lorekeeper.ingest import CHUNKS_FILE, build_index

    if not (INDEX_DIR / CHUNKS_FILE).exists():
        build_index()
    return HybridRetriever(k=k, hybrid=hybrid)
