from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from lorekeeper.graph import ask, build_graph, is_refusal, parse_indices


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.docs


DOCS = [
    Document(page_content="Rulers > Corvin Ashe\nCorvin Ashe negotiated the Treaty of Lanterns.",
             metadata={"id": "rulers#corvin-ashe#0", "source": "rulers", "section": "Corvin Ashe"}),
    Document(page_content="Culture > Food\nSaltbread is baked with seawater.",
             metadata={"id": "culture#food#0", "source": "culture", "section": "Food"}),
]


def run(responses, question="Who negotiated the Treaty of Lanterns?"):
    retriever = FakeRetriever(DOCS)
    graph = build_graph(FakeListChatModel(responses=responses), retriever)
    return ask(graph, question), retriever


def test_chitchat_skips_retrieval():
    res, retriever = run(["chitchat", "Fair tides, visitor."], "hello!")
    assert res["route"] == "chitchat"
    assert retriever.queries == []
    assert res["answer"] == "Fair tides, visitor."
    assert res["trace"] == ["route: chitchat", "respond_direct"]


def test_grounded_path():
    res, retriever = run(["lore", "1", "ok", "Corvin Ashe did [rulers#corvin-ashe].", "yes"])
    assert res["route"] == "lore"
    assert len(retriever.queries) == 1
    assert res["citations"] == ["rulers#corvin-ashe"]
    # graded-relevant passages first, the rest kept (so multi-hop facts aren't dropped)
    assert [d.metadata["source"] for d in res["relevant_docs"]] == ["rulers", "culture"]
    assert res["grounded"] is True
    assert res["trace"][-1] == "check_grounded: grounded"


def test_nothing_relevant_rewrites_then_refuses():
    res, retriever = run(["lore", "NONE", "regatta winner 1190", "NONE", "ok", "The archives hold no record of that."],
                         "Who won the regatta of 1190 TR?")
    assert retriever.queries == ["Who won the regatta of 1190 TR?", "regatta winner 1190"]
    assert res["attempts"] == 2
    assert res["refused"] is True
    assert any(t.startswith("rewrite_query") for t in res["trace"])


def test_ungrounded_answer_is_regenerated_once():
    res, _ = run(["lore", "1", "ok", "Ashe, riding a dragon [rulers#corvin-ashe].", "no",
                  "Corvin Ashe [rulers#corvin-ashe].", "no"])
    gens = [t for t in res["trace"] if t.startswith("generate")]
    assert gens == ["generate", "generate (strict)"]
    assert res["answer"] == "Corvin Ashe [rulers#corvin-ashe]."
    assert res["grounded"] is False  # still doubtful, but we stop after one retry


def test_invalid_citations_are_dropped():
    res, _ = run(["lore", "1,2", "ok", "Ashe [rulers#corvin-ashe] [made-up#thing].", "yes"])
    assert res["citations"] == ["rulers#corvin-ashe"]


def test_parse_indices():
    assert parse_indices("1, 3", 4) == [0, 2]
    assert parse_indices("NONE", 4) == []
    assert parse_indices("Passages 2 and 9", 4) == [1]


def test_is_refusal():
    assert is_refusal("The archives hold no record of that.")
    assert not is_refusal("Corvin Ashe negotiated it.")


def test_false_premise_switches_prompt():
    res, _ = run(["lore", "1", "false: Corvin Ashe negotiated it, not the Hollow Choir",
                  "Actually, it was Corvin Ashe [rulers#corvin-ashe].", "yes"],
                 "Why did the Hollow Choir negotiate the Treaty of Lanterns?")
    assert res["premise"].startswith("Corvin Ashe")
    assert any(t.startswith("check_premise: FALSE") for t in res["trace"])
    assert res["answer"].startswith("Actually")
    assert not res["refused"]


class CountingFake(FakeListChatModel):
    """A fake LLM that remembers which system prompts it was called with."""

    calls: list = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages[0].content[:40])
        return super().invoke(messages, *args, **kwargs)


def test_checking_agent_uses_its_own_model():
    from lorekeeper import prompts

    writer = CountingFake(responses=["lore", "1", "Corvin Ashe did [rulers#corvin-ashe]."], calls=[])
    checker = CountingFake(responses=["ok", "yes"], calls=[])
    graph = build_graph(writer, FakeRetriever(DOCS), checker_llm=checker)
    res = ask(graph, "Who negotiated the Treaty of Lanterns?")

    assert res["grounded"] is True
    assert res["citations"] == ["rulers#corvin-ashe"]
    assert checker.calls == [prompts.PREMISE[:40], prompts.GROUNDED[:40]]
    assert prompts.PREMISE[:40] not in writer.calls
    assert prompts.GROUNDED[:40] not in writer.calls
    assert len(writer.calls) == 3  # route, grade, generate


def test_steps_are_tagged_with_their_agent():
    res, _ = run(["lore", "1", "FALSE: Ashe never sailed.", "Actually, no [rulers#corvin-ashe].", "yes"])
    agents = [s["agent"] for s in res["steps"]]
    assert agents == ["orchestrator", "retrieval", "retrieval", "checking", "orchestrator", "checking"]
    assert res["premise"] == "Ashe never sailed."


def test_retrieval_agent_runs_on_its_own():
    from lorekeeper.graph import build_retrieval_agent

    agent = build_retrieval_agent(FakeListChatModel(responses=["NONE", "treaty lanterns", "2"]), FakeRetriever(DOCS))
    out = agent.invoke({"question": "q", "query": "q", "attempts": 0, "trace": []})
    assert out["attempts"] == 2
    assert out["query"] == "treaty lanterns"
    assert [d.metadata["source"] for d in out["relevant_docs"]] == ["culture"]
