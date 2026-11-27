"""The Lorekeeper agent as a LangGraph state machine.

route -> (chitchat) respond_direct -> END
      -> (lore) retrieve -> grade_docs -> check_premise -> generate -> check_grounded -> END
                   ^           |                              ^              |
                   |  (nothing relevant, once)                +--------------+
                   +---- rewrite_query                   (not grounded: retry once, strict)
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


def build_graph(llm: BaseChatModel, retriever: Retriever):
    def route(state: AgentState) -> dict:
        reply = chat(llm, prompts.ROUTE, state["question"]).lower()
        r = "chitchat" if "chitchat" in reply and "lore" not in reply else "lore"
        return {"route": r, "query": state["question"], "attempts": 0, "trace": [f"route: {r}"]}

    def respond_direct(state: AgentState) -> dict:
        answer = chat(llm, prompts.CHITCHAT, state["question"])
        return {"answer": answer, "citations": [], "docs": [], "relevant_docs": [], "trace": ["respond_direct"]}

    def retrieve(state: AgentState) -> dict:
        docs = retriever.retrieve(state["query"])
        attempts = state.get("attempts", 0) + 1
        return {"docs": docs, "attempts": attempts, "trace": [f"retrieve #{attempts}: {len(docs)} passages"]}

    def grade_docs(state: AgentState) -> dict:
        docs = state["docs"]
        numbered = "\n\n".join(f"Passage {i + 1}:\n{d.page_content}" for i, d in enumerate(docs))
        reply = chat(llm, prompts.GRADE, f"Question: {state['question']}\n\n{numbered}")
        keep = [docs[i] for i in parse_indices(reply, len(docs))]
        return {"relevant_docs": keep, "trace": [f"grade_docs: {len(keep)}/{len(docs)} relevant"]}

    def rewrite_query(state: AgentState) -> dict:
        q = chat(llm, prompts.REWRITE, state["question"]).strip().strip('"')
        return {"query": q or state["question"], "trace": [f"rewrite_query: {q!r}"]}

    def context_docs(state: AgentState) -> list[Document]:
        # The grader's verdict decides whether to search again, but we don't let it *drop*
        # passages: small models often discard the second half of a multi-hop answer.
        # Graded-relevant passages go first, the rest follow.
        graded = state.get("relevant_docs") or []
        graded_ids = {d.metadata["id"] for d in graded}
        return graded + [d for d in state.get("docs") or [] if d.metadata["id"] not in graded_ids]

    def check_premise(state: AgentState) -> dict:
        docs = context_docs(state)
        user = f"Archive passages:\n\n{format_passages(docs)}\n\nQuestion: {state['question']}"
        reply = chat(llm, prompts.PREMISE, user).strip()
        m = re.match(r"\W*false\W*(.*)", reply, re.I | re.S)
        correction = (m.group(1).strip() or "the question's assumption is wrong") if m else None
        if correction and is_refusal(correction):
            correction = None  # "the passages don't mention X" is missing info, not a false premise
        note = f"check_premise: FALSE ({correction[:80]})" if correction else "check_premise: ok"
        return {
            "premise": correction,
            "relevant_docs": docs,
            "trace": ([] if state.get("relevant_docs") else ["grade_docs: nothing relevant, keeping all passages"]) + [note],
        }

    def generate(state: AgentState) -> dict:
        strict = state.get("grounded") is False
        docs = context_docs(state)
        note = []
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
            "relevant_docs": docs,
            "regenerated": state.get("regenerated", False) or strict,
            "trace": note + [f"generate{' (strict)' if strict else ''}"],
        }

    def check_grounded(state: AgentState) -> dict:
        if is_refusal(state["answer"]):
            return {"grounded": True, "trace": ["check_grounded: refusal, nothing to check"]}
        user = f"PASSAGES:\n{format_passages(state['relevant_docs'])}\n\nANSWER:\n{state['answer']}"
        ok = chat(llm, prompts.GROUNDED, user).lower().startswith("y")
        return {"grounded": ok, "trace": [f"check_grounded: {'grounded' if ok else 'NOT grounded'}"]}

    def after_route(state: AgentState) -> str:
        return "respond_direct" if state["route"] == "chitchat" else "retrieve"

    def after_grade(state: AgentState) -> str:
        if state["relevant_docs"] or state["attempts"] >= MAX_RETRIEVALS:
            return "check_premise"
        return "rewrite_query"

    def after_check(state: AgentState) -> str:
        if not state["grounded"] and not state.get("regenerated"):
            return "generate"
        return END

    g = StateGraph(AgentState)
    for name, fn in [
        ("route", route),
        ("respond_direct", respond_direct),
        ("retrieve", retrieve),
        ("grade_docs", grade_docs),
        ("rewrite_query", rewrite_query),
        ("check_premise", check_premise),
        ("generate", generate),
        ("check_grounded", check_grounded),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["respond_direct", "retrieve"])
    g.add_edge("respond_direct", END)
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", after_grade, ["check_premise", "rewrite_query"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("check_premise", "generate")
    g.add_edge("generate", "check_grounded")
    g.add_conditional_edges("check_grounded", after_check, ["generate", END])
    return g.compile()


def ask(graph, question: str) -> dict[str, Any]:
    """Run the agent and return a plain dict that's easy to display and log."""
    start = time.perf_counter()
    state = graph.invoke({"question": question, "trace": []})
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
        "trace": state.get("trace", []),
        "refused": is_refusal(state.get("answer", "")),
        "latency_s": round(time.perf_counter() - start, 2),
    }


def load_agent(k: int | None = None, hybrid: bool | None = None, provider=None):
    from lorekeeper.llm import get_llm
    from lorekeeper.retriever import load_retriever

    return build_graph(get_llm(provider), load_retriever(k=k, hybrid=hybrid))
