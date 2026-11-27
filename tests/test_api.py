"""API tests with a fake LLM and retriever: no model, no network, no database."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel

from lorekeeper import api
from lorekeeper.graph import build_graph
from test_graph import DOCS, FakeRetriever


class ListLog:
    def __init__(self):
        self.rows = []

    def log(self, result):
        self.rows.append(result)


GROUNDED = ["lore", "1", "ok", "Corvin Ashe did [rulers#corvin-ashe].", "yes"]


@pytest.fixture
def client_with():
    made = []

    def make(responses):
        log = ListLog()
        graph = build_graph(FakeListChatModel(responses=responses), FakeRetriever(DOCS))
        api.app.dependency_overrides[api.get_agent] = lambda: graph
        api.app.dependency_overrides[api.get_query_log] = lambda: log
        made.append(log)
        return TestClient(api.app), log

    yield make
    api.app.dependency_overrides.clear()


def test_health():
    r = TestClient(api.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vector_store"] in {"chroma", "pgvector"}


def test_ask_returns_answer_citations_and_steps(client_with):
    client, log = client_with(GROUNDED)
    r = client.post("/ask", json={"question": "Who negotiated the Treaty of Lanterns?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == ["rulers#corvin-ashe"]
    assert body["grounded"] is True and body["refused"] is False
    assert [p["cited"] for p in body["passages"]] == [True, False]
    assert {s["agent"] for s in body["steps"]} == {"orchestrator", "retrieval", "checking"}
    assert len(log.rows) == 1 and log.rows[0]["question"] == "Who negotiated the Treaty of Lanterns?"


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_ask_rejects_bad_questions(client_with, question):
    client, log = client_with(GROUNDED)
    assert client.post("/ask", json={"question": question}).status_code == 422
    assert log.rows == []


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_sends_steps_then_answer(client_with):
    client, log = client_with(GROUNDED)
    r = client.post("/ask/stream", json={"question": "Who negotiated the Treaty of Lanterns?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[-1] == "answer" and set(names[:-1]) == {"step"}
    nodes = [d["node"] for e, d in events if e == "step"]
    assert nodes == ["route", "retrieval_agent", "premise_check", "generate", "grounding_check"]
    assert events[-1][1]["citations"] == ["rulers#corvin-ashe"]
    assert len(log.rows) == 1


def test_stream_reports_errors(client_with):
    client, log = client_with([])  # the fake LLM has nothing to say -> raises
    events = parse_sse(client.post("/ask/stream", json={"question": "anything"}).text)
    assert events[-1][0] == "error"
    assert log.rows == []


def test_eval_runs(tmp_path, monkeypatch):
    run = {"config": {"top_k": 4}, "summary": {"overall": {"correctness": 0.9}, "by_type": {"single_hop": {"n": 1}}}}
    (tmp_path / "20260101-000000_test.json").write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(api, "EVAL_RESULTS", tmp_path)
    client = TestClient(api.app)

    runs = client.get("/eval/runs").json()
    assert [r["name"] for r in runs] == ["20260101-000000_test"]
    assert client.get("/eval/runs/20260101-000000_test").json()["overall"]["correctness"] == 0.9
    assert client.get("/eval/runs/missing").status_code == 404
    assert client.get("/eval/runs/..%2Fsecrets").status_code in {400, 404}
