"""Integration tests against a real PostgreSQL + pgvector.

Skipped unless DATABASE_URL is set, e.g.
    docker compose up -d db
    DATABASE_URL=postgresql://lorekeeper:lorekeeper@localhost:5432/lorekeeper uv run pytest tests/test_pgvector.py

A small hashing embedder stands in for fastembed, so no model download is needed.
"""

import hashlib
import math
import os

import pytest
from langchain_core.embeddings import Embeddings

from lorekeeper.ingest import load_chunks
from lorekeeper.pgstore import keyword_tsquery

DATABASE_URL = os.environ.get("DATABASE_URL")
TABLE = "test_chunks"
DIM = 64

needs_db = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")


class HashEmbeddings(Embeddings):
    """Bag-of-words hashed into a fixed-size unit vector."""

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_keyword_tsquery_ors_words():
    assert keyword_tsquery("Why do sky-whales migrate?") == "why | do | sky | whales | migrate"
    assert keyword_tsquery("Ferrow's ship") == "ferrows | ship"
    assert keyword_tsquery("?!") == ""


def test_table_name_is_checked():
    from lorekeeper.pgstore import _check_table

    with pytest.raises(ValueError):
        _check_table("chunks; DROP TABLE x")


@pytest.fixture(scope="module")
def retriever():
    from lorekeeper.pgstore import PgVectorRetriever, build_pg_index, connect

    chunks = load_chunks()
    n = build_pg_index(chunks, HashEmbeddings(), DATABASE_URL, table=TABLE)
    assert n == len(chunks)
    r = PgVectorRetriever(DATABASE_URL, HashEmbeddings(), k=4, table=TABLE)
    yield r
    r.conn.close()
    with connect(DATABASE_URL) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")


@needs_db
def test_all_chunks_stored(retriever):
    from lorekeeper.pgstore import table_size

    assert table_size(DATABASE_URL, TABLE) == len(load_chunks())


@needs_db
def test_keyword_search_finds_the_document(retriever):
    ids = retriever.keyword_ids("sky-whales migration", 5)
    assert ids and ids[0].startswith("sky_whales#")


@needs_db
def test_dense_search_returns_k(retriever):
    assert len(retriever.dense_ids("Who rules Veyra?", 4)) == 4


@needs_db
def test_hybrid_retrieve_returns_documents_with_metadata(retriever):
    docs = retriever.retrieve("What do sky-whales eat?")
    assert len(docs) == 4
    assert docs[0].metadata["source"] == "sky_whales"
    assert {"id", "source", "section", "title"} <= set(docs[0].metadata)


@needs_db
def test_dense_only_mode(retriever):
    retriever.hybrid = False
    try:
        assert len(retriever.retrieve("driftglass")) == 4
    finally:
        retriever.hybrid = True
