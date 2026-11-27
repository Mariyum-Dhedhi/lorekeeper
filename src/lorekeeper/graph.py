"""The Lorekeeper as a small multi-agent LangGraph system.

Orchestrator:
  route -> (chitchat) respond_direct -> END
        -> (lore) retrieval_agent -> premise_check -> generate -> grounding_check -> END
                                                        ^               |
                                                        +---------------+
                                                   (not grounded: retry once, strict)

Retrieval agent (own subgraph): retrieve -> grade_docs -> (nothing relevant, once) rewrite_query -> retrieve
Checking agent (own subgraph):  check_premise | check_grounded, optionally on a separate model
"""

import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from lorekeeper import prompts
from lorekeeper.llm import chat
from lorekeeper.retriever import Retriever

MAX_RETRIEVALS = 2  # initial search + one rewrite

REFUSAL_MARKERS = (
    "no record",
    "not recorded",
    "isn't recorded",
    "is not recorded",
    "do not record",
    "don't record",
    "does not record",
    "no mention",
    "not mention",
    "not mentioned",
    "never mention",
    "do not contain",
    "don't contain",
    "does not contain",
    "cannot find",
    "can't find",
    "don't know",
    "do not know",
)

# Which agent owns each step. The orchestrator routes and writes the answer.
NODE_AGENT = {
    "route": "orchestrator",
    "respond_direct": "orchestrator",
    "generate": "orchestrator",
    "retrieve": "retrieval",
    "grade_docs": "retrieval",
    "rewrite_query": "retrieval",
    "check_premise": "checking",
    "check_grounded": "checking",
}


class AgentState(TypedDict, total=False):
    question: str
    query: str
    route: str
    docs: list[Document]
    relevant_docs: list[Document]
    answer: str
    citations: list[str]
    attempts: int
    premise: str | None  # correction text if the question has a false assumption
    grounded: bool | None
    regenerated: bool
    trace: Annotated[list[str], operator.add]


class RetrievalState(TypedDict, total=False):
    question: str
    query: str
    docs: list[Document]
    relevant_docs: list[Document]
    attempts: int
    trace: Annotated[list[str], operator.add]


class CheckState(TypedDict, total=False):
    task: str  # "premise" or "grounding"
    question: str
    docs: list[Document]
    answer: str
    premise: str | None
    grounded: bool | None
    trace: Annotated[list[str], operator.add]


def label(doc: Document) -> str:
    return f"{doc.metadata['source']}#{doc.metadata['id'].split('#')[1]}"


def format_passages(docs: list[Document]) -> str:
    return "\n\n".join(f"[{label(d)}]\n{d.page_content}" for d in docs)


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)


def parse_indices(text: str, n: int) -> list[int]:
    if "none" in text.lower() and not re.search(r"\d", text):
        return []
    return sorted({int(x) - 1 for x in re.findall(r"\d+", text) if 1 <= int(x) <= n})


def build_retrieval_agent(llm: BaseChatModel, retriever: Retriever):
    """Search, grade the passages, and rewrite the query once if nothing useful came back."""

    def retrieve(state: RetrievalState) -> dict:
        docs = retriever.retrieve(state["query"])
        attempts = state.get("attempts", 0) + 1
        return {"docs": docs, "attempts": attempts, "trace": [f"retrieve #{attempts}: {len(docs)} passages"]}

    def grade_docs(state: RetrievalState) -> dict:
        docs = state["docs"]
        numbered = "\n\n".join(f"Passage {i + 1}:\n{d.page_content}" for i, d in enumerate(docs))
        reply = chat(llm, prompts.GRADE, f"Question: {state['question']}\n\n{numbered}")
        keep = [docs[i] for i in parse_indices(reply, len(docs))]
        return {"relevant_docs": keep, "trace": [f"grade_docs: {len(keep)}/{len(docs)} relevant"]}

    def rewrite_query(state: RetrievalState) -> dict:
        q = chat(llm, prompts.REWRITE, state["question"]).strip().strip('"')
        return {"query": q or state["question"], "trace": [f"rewrite_query: {q!r}"]}

    def after_grade(state: RetrievalState) -> str:
        if state["relevant_docs"] or state["attempts"] >= MAX_RETRIEVALS:
            return END
        return "rewrite_query"

    g = StateGraph(RetrievalState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_docs", grade_docs)
    g.add_node("rewrite_query", rewrite_query)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", after_grade, ["rewrite_query", END])
    g.add_edge("rewrite_query", "retrieve")
    return g.compile()


def build_checking_agent(llm: BaseChatModel):
    """Checks the question before an answer is written, and the answer after it.

    It can run on a different model from the one that writes answers, so an answer
    is not only judged by the model that produced it.
    """

    def check_premise(state: CheckState) -> dict:
        user = f"Archive passages:\n\n{format_passages(state['docs'])}\n\nQuestion: {state['question']}"
        reply = chat(llm, prompts.PREMISE, user).strip()
        m = re.match(r"\W*false\W*(.*)", reply, re.I | re.S)
        correction = (m.group(1).strip() or "the question's assumption is wrong") if m else None
        if correction and is_refusal(correction):
            correction = None  # "the passages don't mention X" is missing info, not a false premise
        note = f"check_premise: FALSE ({correction[:80]})" if correction else "check_premise: ok"
        return {"premise": correction, "trace": [note]}

    def check_grounded(state: CheckState) -> dict:
        if is_refusal(state["answer"]):
            return {"grounded": True, "trace": ["check_grounded: refusal, nothing to check"]}
        user = f"PASSAGES:\n{format_passages(state['docs'])}\n\nANSWER:\n{state['answer']}"
        ok = chat(llm, prompts.GROUNDED, user).lower().startswith("y")
        return {"grounded": ok, "trace": [f"check_grounded: {'grounded' if ok else 'NOT grounded'}"]}

    def pick_task(state: CheckState) -> str:
        return "check_grounded" if state["task"] == "grounding" else "check_premise"

    g = StateGraph(CheckState)
    g.add_node("check_premise", check_premise)
    g.add_node("check_grounded", check_grounded)
    g.add_conditional_edges(START, pick_task, ["check_premise", "check_grounded"])
    g.add_edge("check_premise", END)
    g.add_edge("check_grounded", END)
    return g.compile()


def build_graph(llm: BaseChatModel, retriever: Retriever, checker_llm: BaseChatModel | None = None):
    """The orchestrator. It routes the message, hands the search to the retrieval agent,
    writes the answer, and has the checking agent review the question and the answer.
    `checker_llm` defaults to `llm`."""
    retrieval_agent = build_retrieval_agent(llm, retriever)
    checking_agent = build_checking_agent(checker_llm or llm)

    def route(state: AgentState) -> dict:
        reply = chat(llm, prompts.ROUTE, state["question"]).lower()
        r = "chitchat" if "chitchat" in reply and "lore" not in reply else "lore"
        return {"route": r, "query": state["question"], "attempts": 0, "trace": [f"route: {r}"]}

    def respond_direct(state: AgentState) -> dict:
        answer = chat(llm, prompts.CHITCHAT, state["question"])
        return {"answer": answer, "citations": [], "docs": [], "relevant_docs": [], "trace": ["respond_direct"]}

    def retrieval(state: AgentState) -> dict:
        # Each agent runs on its own state; only its results come back to the orchestrator.
        out = retrieval_agent.invoke({"question": state["question"], "query": state["query"], "attempts": 0, "trace": []})
        graded, docs = out.get("relevant_docs") or [], out.get("docs") or []
        # The grader's verdict decides whether to search again, but it may not *drop*
        # passages: small models often discard the second half of a multi-hop answer.
        # Graded-relevant passages go first, the rest follow.
        graded_ids = {d.metadata["id"] for d in graded}
        context = graded + [d for d in docs if d.metadata["id"] not in graded_ids]
        note = [] if graded else ["grade_docs: nothing relevant, keeping all passages"]
        return {
            "query": out["query"],
            "docs": docs,
            "relevant_docs": context,
            "attempts": out["attempts"],
            "trace": out["trace"] + note,
        }

    def premise_check(state: AgentState) -> dict:
        out = checking_agent.invoke(
            {"task": "premise", "question": state["question"], "docs": state["relevant_docs"], "trace": []}
        )
        return {"premise": out.get("premise"), "trace": out["trace"]}

    def generate(state: AgentState) -> dict:
        strict = state.get("grounded") is False
        docs = state["relevant_docs"]
        if state.get("premise"):
            system = prompts.CORRECT_PREMISE.format(correction=state["premise"])
        else:
            system = prompts.GENERATE
        system += prompts.STRICT_SUFFIX if strict else ""
        user = f"Archive passages:\n\n{format_passages(docs)}\n\nQuestion: {state['question']}"
        answer = chat(llm, system, user)
        valid = {label(d) for d in docs}
        citations = [c for c in dict.fromkeys(re.findall(r"\[([^\[\]]+#[^\[\]]+)\]", answer)) if c in valid]
        return {
            "answer": answer,
            "citations": citations,
            "regenerated": state.get("regenerated", False) or strict,
            "trace": [f"generate{' (strict)' if strict else ''}"],
        }

    def grounding_check(state: AgentState) -> dict:
        out = checking_agent.invoke(
            {"task": "grounding", "answer": state["answer"], "docs": state["relevant_docs"], "trace": []}
        )
        return {"grounded": out["grounded"], "trace": out["trace"]}

    def after_route(state: AgentState) -> str:
        return "respond_direct" if state["route"] == "chitchat" else "retrieval_agent"

    def after_check(state: AgentState) -> str:
        if not state["grounded"] and not state.get("regenerated"):
            return "generate"
        return END

    g = StateGraph(AgentState)
    for name, fn in [
        ("route", route),
        ("respond_direct", respond_direct),
        ("retrieval_agent", retrieval),
        ("premise_check", premise_check),
        ("generate", generate),
        ("grounding_check", grounding_check),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["respond_direct", "retrieval_agent"])
    g.add_edge("respond_direct", END)
    g.add_edge("retrieval_agent", "premise_check")
    g.add_edge("premise_check", "generate")
    g.add_edge("generate", "grounding_check")
    g.add_conditional_edges("grounding_check", after_check, ["generate", END])
    return g.compile()


def trace_steps(trace: list[str]) -> list[dict[str, str]]:
    """Tag each trace line with the agent that produced it."""
    steps = []
    for line in trace:
        m = re.match(r"[a-z_]+", line)
        steps.append({"agent": NODE_AGENT.get(m.group(0) if m else "", "orchestrator"), "step": line})
    return steps


def ask(graph, question: str) -> dict[str, Any]:
    """Run the agent and return a plain dict that's easy to display and log."""
    start = time.perf_counter()
    state = graph.invoke({"question": question, "trace": []})
    trace = state.get("trace", [])
    return {
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


def load_agent(k: int | None = None, hybrid: bool | None = None, provider=None):
    from lorekeeper.config import get_settings
    from lorekeeper.llm import get_checker_llm, get_llm
    from lorekeeper.retriever import load_retriever

    llm = get_llm(provider)
    checker = get_checker_llm() if get_settings().checker_provider else llm
    return build_graph(llm, load_retriever(k=k, hybrid=hybrid), checker_llm=checker)
