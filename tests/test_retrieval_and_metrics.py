import pytest

from lorekeeper.ingest import load_chunks, split_markdown_sections
from lorekeeper.judge import parse_score
from lorekeeper.metrics import hit_at_k, mean, mrr, recall_at_k
from lorekeeper.retriever import reciprocal_rank_fusion


def test_split_markdown_sections():
    title, sections = split_markdown_sections("# Title\n\n## A\nalpha\n\n## B\nbeta\nmore")
    assert title == "Title"
    assert sections == [("A", "alpha"), ("B", "beta\nmore")]


def test_chunks_keep_metadata():
    chunks = load_chunks()
    assert len(chunks) > 30
    sources = {c.metadata["source"] for c in chunks}
    assert {"overview", "sky_whales", "rulers"} <= sources
    ids = [c.metadata["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    for c in chunks:
        assert c.metadata["section"]
        assert c.page_content.startswith(c.metadata["title"])


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert fused[0] == "b"  # found by both retrievers beats top of only one
    assert set(fused) == {"a", "b", "c"}
    assert fused.index("a") < fused.index("c")  # rank 1 beats rank 2


def test_retrieval_metrics():
    retrieved = ["history", "history", "magic", "rulers"]
    assert hit_at_k(retrieved, ["rulers"]) == 1.0
    assert hit_at_k(retrieved, ["calendar"]) == 0.0
    assert recall_at_k(retrieved, ["magic", "calendar"]) == 0.5
    assert mrr(retrieved, ["magic"]) == pytest.approx(1 / 2)  # duplicates collapse
    assert mrr(retrieved, ["calendar"]) == 0.0
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def test_parse_score():
    assert parse_score('{"score": 1, "reason": "matches"}') == (1.0, "matches")
    assert parse_score('Sure! {"score": 0.5, "reason": "partial"} hope that helps')[0] == 0.5
    assert parse_score("score: 0")[0] == 0.0
    assert parse_score("gibberish")[0] == 0.0
