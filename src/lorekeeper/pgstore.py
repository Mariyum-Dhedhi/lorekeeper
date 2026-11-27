"""PostgreSQL + pgvector store, with hybrid search done inside the database.

Dense side: pgvector cosine distance on an HNSW index.
Keyword side: Postgres full-text search (tsvector + ts_rank_cd) on a GIN index.
The two rankings are fused with the same reciprocal rank fusion as the Chroma path.

Start the database:  docker compose up -d db
Build the table:     uv run python -m lorekeeper.ingest --store pgvector
"""

import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from lorekeeper.retriever import reciprocal_rank_fusion, tokenize

DEFAULT_TABLE = "chunks"


def connect(url: str):
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def _check_table(table: str) -> str:
    # Table names can't be bound as parameters, so only allow plain identifiers.
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", table):
        raise ValueError(f"invalid table name: {table!r}")
    return table


def create_schema(conn, dim: int, table: str = DEFAULT_TABLE) -> None:
    t = _check_table(table)
    conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.execute(
        f"""
        CREATE TABLE {t} (
            id        text PRIMARY KEY,
            source    text NOT NULL,
            section   text NOT NULL,
            title     text NOT NULL,
            content   text NOT NULL,
            embedding vector({int(dim)}) NOT NULL,
            tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        )
        """
    )
    conn.execute(f"CREATE INDEX {t}_embedding_hnsw ON {t} USING hnsw (embedding vector_cosine_ops)")
    conn.execute(f"CREATE INDEX {t}_tsv_gin ON {t} USING gin (tsv)")


def build_pg_index(chunks: list[Document], embeddings: Embeddings, url: str, table: str = DEFAULT_TABLE) -> int:
    import numpy as np

    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    with connect(url) as conn:
        create_schema(conn, dim=len(vectors[0]), table=table)
        with conn.cursor() as cur, conn.transaction():
            cur.executemany(
                f"INSERT INTO {_check_table(table)} (id, source, section, title, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (c.metadata["id"], c.metadata["source"], c.metadata["section"], c.metadata["title"],
                     c.page_content, np.asarray(v, dtype=np.float32))
                    for c, v in zip(chunks, vectors)
                ],
            )
    return len(chunks)


def table_size(url: str, table: str = DEFAULT_TABLE) -> int:
    """Number of rows in the table, or 0 if it doesn't exist yet."""
    with connect(url) as conn:
        exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]
        if exists is None:
            return 0
        return conn.execute(f"SELECT count(*) FROM {_check_table(table)}").fetchone()[0]


def keyword_tsquery(query: str) -> str:
    """OR the query's words together. plainto_tsquery ANDs them, which finds almost
    nothing for a full natural-language question."""
    words = [w.replace("'", "") for w in tokenize(query)]
    return " | ".join(w for w in words if w)


class PgVectorRetriever:
    def __init__(self, url: str, embeddings: Embeddings, k: int = 4, hybrid: bool = True, table: str = DEFAULT_TABLE):
        self.conn = connect(url)
        self.embeddings = embeddings
        self.k = k
        self.hybrid = hybrid
        self.table = _check_table(table)

    def dense_ids(self, query: str, n: int) -> list[str]:
        import numpy as np

        vec = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        rows = self.conn.execute(
            f"SELECT id FROM {self.table} ORDER BY embedding <=> %s LIMIT %s", (vec, n)
        ).fetchall()
        return [r[0] for r in rows]

    def keyword_ids(self, query: str, n: int) -> list[str]:
        tsq = keyword_tsquery(query)
        if not tsq:
            return []
        rows = self.conn.execute(
            f"""
            SELECT id FROM {self.table}, to_tsquery('english', %s) q
            WHERE tsv @@ q
            ORDER BY ts_rank_cd(tsv, q) DESC, id
            LIMIT %s
            """,
            (tsq, n),
        ).fetchall()
        return [r[0] for r in rows]

    def fetch(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        rows = self.conn.execute(
            f"SELECT id, source, section, title, content FROM {self.table} WHERE id = ANY(%s)", (ids,)
        ).fetchall()
        by_id = {
            r[0]: Document(page_content=r[4], metadata={"id": r[0], "source": r[1], "section": r[2], "title": r[3]})
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    def retrieve(self, query: str) -> list[Document]:
        pool = self.k * 3
        if self.hybrid:
            ids = reciprocal_rank_fusion([self.dense_ids(query, pool), self.keyword_ids(query, pool)])
        else:
            ids = self.dense_ids(query, self.k)
        return self.fetch(ids[: self.k])
