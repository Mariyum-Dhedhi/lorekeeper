"""Shared Streamlit helpers: cached agent/judge, answer rendering, and a live graph diagram."""

import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from lorekeeper.config import get_settings
from lorekeeper.graph import label
from lorekeeper.llm import get_judge_llm, provider_label

NODES = ["route", "respond_direct", "retrieve", "grade_docs", "rewrite_query", "check_premise", "generate", "check_grounded"]
EDGES = [
    ("START", "route", ""),
    ("route", "respond_direct", "chitchat"),
    ("route", "retrieve", "lore"),
    ("respond_direct", "END", ""),
    ("retrieve", "grade_docs", ""),
    ("grade_docs", "check_premise", "relevant"),
    ("check_premise", "generate", ""),
    ("grade_docs", "rewrite_query", "nothing"),
    ("rewrite_query", "retrieve", ""),
    ("generate", "check_grounded", ""),
    ("check_grounded", "END", "grounded"),
    ("check_grounded", "generate", "retry once"),
]
# The two agents the orchestrator hands work to (see lorekeeper.graph).
CLUSTERS = [
    ("retrieval", "Retrieval agent", ["retrieve", "grade_docs", "rewrite_query"]),
    ("checking", "Checking agent", ["check_premise", "check_grounded"]),
]


@st.cache_resource(show_spinner="Opening the archives (first run builds the index)...")
def get_agent():
    from lorekeeper.graph import load_agent

    return load_agent()


@st.cache_resource
def get_judge():
    return get_judge_llm()


@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


def sidebar_badge() -> None:
    s = get_settings()
    with st.sidebar:
        st.caption("Archive engine")
        st.code(f"LLM     {provider_label()}\nChecker {provider_label(s.checker_provider)}\nJudge   {provider_label(s.judge_provider)}\nTop-k   {s.top_k}\nSearch  {'hybrid (dense+BM25)' if s.hybrid else 'dense only'}", language=None)
        st.caption("Change these in `.env`")


def visited_nodes(trace: list[str]) -> list[str]:
    return [m.group(0) for t in trace if (m := re.match(r"[a-z_]+", t)) and m.group(0) in NODES]


def graph_dot(trace: list[str] | None = None) -> str:
    seen = set(visited_nodes(trace or []))
    lines = [
        "digraph G {",
        'rankdir=TB; bgcolor="transparent";',
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#94a3b8" fillcolor="#f1f5f9" fontcolor="#1e293b"];',
        'edge [color="#94a3b8" fontname="Helvetica" fontsize=9 fontcolor="#64748b"];',
        'START [shape=circle label="" width=0.2 fillcolor="#475569"]; END [shape=doublecircle label="" width=0.15 fillcolor="#475569"];',
    ]
    for n in NODES:
        if n in seen:
            lines.append(f'{n} [fillcolor="#fde68a" color="#d97706" penwidth=2];')
    for name, title, members in CLUSTERS:
        lines.append(
            f'subgraph cluster_{name} {{ label="{title}"; style="rounded,dashed"; color="#94a3b8"; '
            f'fontname="Helvetica"; fontsize=10; fontcolor="#64748b"; {"; ".join(members)}; }}'
        )
    for a, b, lab in EDGES:
        lines.append(f'{a} -> {b} [label="{lab}"];')
    lines.append("}")
    return "\n".join(lines)


def render_result(result: dict, show_graph: bool = True) -> None:
    """Sources + graph trace under an answer."""
    docs = result.get("relevant_docs") or []
    cited = set(result.get("citations") or [])
    if docs:
        with st.expander(f"Evidence: {len(docs)} passage(s) consulted"):
            for d in docs:
                tag = "cited" if label(d) in cited else "consulted"
                st.markdown(f"**`{label(d)}`** · _{tag}_")
                st.caption(d.page_content)
    with st.expander(f"How the Lorekeeper thought · {result.get('latency_s', '?')}s"):
        if show_graph:
            st.graphviz_chart(graph_dot(result.get("trace")))
        st.code("\n".join(result.get("trace", [])), language=None)
        if result.get("grounded") is False:
            st.warning("The groundedness check still had doubts about this answer.")
