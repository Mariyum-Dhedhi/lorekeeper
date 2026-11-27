"""HTTP API for the Lorekeeper.

Run:   uv run uvicorn lorekeeper.api:app --reload
Docs:  http://127.0.0.1:8000/docs
"""

import json
import re
import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from lorekeeper import __version__
from lorekeeper.config import EVAL_RESULTS, get_settings
from lorekeeper.graph import AgentState, ask, is_refusal, label, trace_steps
from lorekeeper.llm import provider_label

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
RUN_NAME = re.compile(r"[\w.-]{1,200}")


# ---------- request / response models ----------


class AskRequest(BaseModel):
    question: Question = Field(examples=["Who was Regent during the Stormwar?"])


class Passage(BaseModel):
    label: str = Field(description="Citation label, as used in the answer: source#section")
    source: str
    section: str
    text: str
    cited: bool


class Step(BaseModel):
    agent: str = Field(description="orchestrator, retrieval or checking")
    step: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    refused: bool
    grounded: bool | None
    premise: str | None = Field(description="Correction, if the question had a false assumption")
    route: str | None
    retrievals: int
    latency_s: float
    steps: list[Step]
    passages: list[Passage]


class Health(BaseModel):
    status: str
    version: str
    llm: str
    checker: str
    vector_store: str
    database: str = Field(description="ok, unreachable, or not used")


class EvalRunSummary(BaseModel):
    name: str
    config: dict[str, Any]
    overall: dict[str, Any]
    by_type: dict[str, dict[str, Any]]


# ---------- dependencies (overridden in tests) ----------


class QueryLog(Protocol):
    def log(self, result: dict) -> None: ...


class NoLog:
    def log(self, result: dict) -> None:
        pass


class PgQueryLog:
    """Writes each answered question to a `queries` table."""

    def __init__(self, url: str):
        from lorekeeper.pgstore import connect

        self.conn = connect(url)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id          bigserial PRIMARY KEY,
                asked_at    timestamptz NOT NULL DEFAULT now(),
                question    text NOT NULL,
                answer      text NOT NULL,
                citations   text[] NOT NULL,
                refused     boolean NOT NULL,
                grounded    boolean,
                latency_s   real NOT NULL,
                trace       jsonb NOT NULL
            )
            """
        )

    def log(self, result: dict) -> None:
        self.conn.execute(
            "INSERT INTO queries (question, answer, citations, refused, grounded, latency_s, trace) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (result["question"], result["answer"], result["citations"], result["refused"],
             result["grounded"], result["latency_s"], json.dumps(result["trace"])),
        )


@lru_cache
def get_agent():
    from lorekeeper.graph import load_agent

    return load_agent()


@lru_cache
def get_query_log() -> QueryLog:
    s = get_settings()
    return PgQueryLog(s.database_url) if s.vector_store == "pgvector" else NoLog()


def database_status() -> str:
    s = get_settings()
    if s.vector_store != "pgvector":
        return "not used"
    try:
        from lorekeeper.pgstore import connect

        with connect(s.database_url) as conn:
            conn.execute("SELECT 1")
        return "ok"
    except Exception:
        return "unreachable"


# ---------- app ----------

app = FastAPI(
    title="Lorekeeper API",
    version=__version__,
    description="A multi-agent RAG system over the fictional realm of Veyra. "
    "Answers cite their sources and refuse when the archives don't say.",
)


def to_response(result: dict) -> AskResponse:
    cited = set(result["citations"])
    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        citations=result["citations"],
        refused=result["refused"],
        grounded=result["grounded"],
        premise=result["premise"],
        route=result["route"],
        retrievals=result["attempts"],
        latency_s=result["latency_s"],
        steps=[Step(**s) for s in result["steps"]],
        passages=[
            Passage(label=label(d), source=d.metadata["source"], section=d.metadata["section"],
                    text=d.page_content, cited=label(d) in cited)
            for d in result["relevant_docs"]
        ],
    )


@app.get("/health", response_model=Health, tags=["meta"])
def health() -> Health:
    s = get_settings()
    return Health(
        status="ok",
        version=__version__,
        llm=provider_label(),
        checker=provider_label(s.checker_provider),
        vector_store=s.vector_store,
        database=database_status(),
    )


@app.post("/ask", response_model=AskResponse, tags=["agent"])
def ask_question(req: AskRequest, agent=Depends(get_agent), qlog: QueryLog = Depends(get_query_log)) -> AskResponse:
    """Ask the Lorekeeper a question and get the full answer at once."""
    result = ask(agent, req.question)
    qlog.log(result)
    return to_response(result)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_events(agent, qlog: QueryLog, question: str) -> Iterator[str]:
    start = time.perf_counter()
    state: AgentState = {"question": question, "trace": []}
    trace: list[str] = []
    try:
        for update in agent.stream({"question": question, "trace": []}, stream_mode="updates"):
            for node, delta in update.items():
                delta = delta or {}
                lines = delta.get("trace", [])
                trace += lines
                state.update({k: v for k, v in delta.items() if k != "trace"})
                yield sse("step", {"node": node, "steps": trace_steps(lines)})
    except Exception as e:  # e.g. the LLM is not running; the client still gets a clean end
        yield sse("error", {"detail": str(e)})
        return
    result = {
        "question": question,
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "route": state.get("route"),
        "docs": state.get("docs", []),
        "relevant_docs": state.get("relevant_docs", []),
        "grounded": state.get("grounded"),
        "premise": state.get("premise"),
        "attempts": state.get("attempts", 0),
        "trace": trace,
        "steps": trace_steps(trace),
        "refused": is_refusal(state.get("answer", "")),
        "latency_s": round(time.perf_counter() - start, 2),
    }
    qlog.log(result)
    yield sse("answer", to_response(result).model_dump())


@app.post("/ask/stream", tags=["agent"], response_class=StreamingResponse,
          responses={200: {"content": {"text/event-stream": {}},
                           "description": "`step` events as the graph runs, then one `answer` event"}})
def ask_stream(req: AskRequest, agent=Depends(get_agent), qlog: QueryLog = Depends(get_query_log)):
    """Server-Sent Events: one `step` event per graph step, then the final `answer`."""
    return StreamingResponse(stream_events(agent, qlog, req.question), media_type="text/event-stream")


def load_run(path) -> EvalRunSummary:
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalRunSummary(name=path.stem, config=data["config"], overall=data["summary"]["overall"],
                          by_type=data["summary"]["by_type"])


@app.get("/eval/runs", response_model=list[EvalRunSummary], tags=["eval"])
def eval_runs() -> list[EvalRunSummary]:
    """Saved eval runs, newest first."""
    if not EVAL_RESULTS.exists():
        return []
    return [load_run(p) for p in sorted(EVAL_RESULTS.glob("*.json"), reverse=True)]


@app.get("/eval/runs/{name}", response_model=EvalRunSummary, tags=["eval"])
def eval_run(name: str) -> EvalRunSummary:
    if not RUN_NAME.fullmatch(name) or ".." in name:
        raise HTTPException(400, "invalid run name")
    path = EVAL_RESULTS / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, f"no eval run named {name!r}")
    return load_run(path)
