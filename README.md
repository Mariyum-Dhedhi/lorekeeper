<div align="center">

# 📜 Lorekeeper

**A question-answering agent for an invented world. It cites the passage behind each answer, and says so when the documents don't cover a question.**

[![tests](https://github.com/Mariyum-Dhedhi/lorekeeper/actions/workflows/tests.yml/badge.svg)](https://github.com/Mariyum-Dhedhi/lorekeeper/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

[Quick start](#-quick-start) · [How it works](#-how-it-works) · [API](#-the-api) · [Evaluation](#-evaluation) · [Quiz Duel](#%EF%B8%8F-quiz-duel) · [Design notes](#-design-notes-and-limits)

<br>

<a href="docs/media/lorekeeper-launch.mp4"><img src="docs/media/lorekeeper-demo.gif" width="760" alt="Lorekeeper catching a false premise: asked why sky-whales migrate every seven years, it answers that the archives say every eleven years, with a citation"></a>

**▶️ [Watch the 22-second video, with sound](docs/media/lorekeeper-launch.mp4)**

</div>

---

The documents describe Veyra, an invented island realm of guilds, sky-whales and one ill-advised war. It is fictional on purpose: no model saw it during training, so the 40-question eval can only be passed by finding the right passage and answering from it, not from memory.

> [!TIP]
> No paid services are needed. The language model and the embeddings run locally (Ollama, fastembed), Groq's free tier is optional, and PostgreSQL runs in Docker.

## 📸 Screenshots

<table>
<tr>
<td width="50%"><img src="docs/images/chat.png" alt="The chat page: a corrected false premise with its citation, and the agent graph with the steps that ran highlighted"><br><sub><b>Chat.</b> Each answer shows the passages it used and a graph of the steps that ran.</sub></td>
<td width="50%"><img src="docs/images/quiz-duel.png" alt="The Quiz Duel start screen with scoring rules and ranks"><br><sub><b>Quiz Duel.</b> You against the agent: 10 questions, a 30-second clock and ranks.</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/images/eval-dashboard.png" alt="The Eval Dashboard showing 0.9 correctness, 1.0 hit@k and per-type bars"><br><sub><b>Eval Dashboard.</b> Pick a run, or two to compare.</sub></td>
<td width="50%"><img src="docs/images/api-docs.png" alt="FastAPI's generated docs listing the health, ask, stream and eval endpoints"><br><sub><b>API.</b> FastAPI's generated docs at <code>/docs</code>.</sub></td>
</tr>
</table>

## 🧭 What you can run

| I want to… | Do this | Needs |
|---|---|---|
| 💬 **Chat** with the agent and see each step it took | `uv run streamlit run app/Home.py` | Ollama |
| ⚔️ **Play** the Quiz Duel against it | Streamlit, then the *Quiz Duel* page | Ollama |
| 🔌 **Call it over HTTP** | `uv run uvicorn lorekeeper.api:app` → [`/docs`](http://127.0.0.1:8000/docs) | Ollama |
| 🐘 **Use PostgreSQL + pgvector** | `docker compose up -d db`, set `VECTOR_STORE=pgvector` | Docker |
| 🐳 **Run it all in containers** | `docker compose up --build` | Docker + Ollama |
| 📊 **Measure it** | `uv run python -m lorekeeper.eval` | Ollama |
| 🧪 **Run the tests** | `uv run pytest` | nothing (fake LLM) |

## 🚀 Quick start

```powershell
ollama pull qwen2.5:7b                      # 1. local LLM (skip if you have it)
uv sync                                     # 2. install
copy .env.example .env                      #    (cp on macOS/Linux)
uv run python -m lorekeeper.ingest          # 3. build the index (the app also does this on first run)
uv run streamlit run app/Home.py            # 4. play
```

The first run downloads the ~70 MB embedding model into `.cache/`.

<details>
<summary><b>🐘 With PostgreSQL + pgvector instead of Chroma</b></summary>

```powershell
docker compose up -d db                     # pgvector/pgvector:pg17 on 127.0.0.1:5432
# in .env:  VECTOR_STORE=pgvector
uv run python -m lorekeeper.ingest          # fills the `chunks` table
uv run streamlit run app/Home.py
```

Chroma stays the default, so nothing changes for anyone who doesn't use Docker.
</details>

<details>
<summary><b>🐳 Everything in Docker (API + database)</b></summary>

```powershell
docker compose up --build                   # API on http://127.0.0.1:8000/docs
```

The API container talks to Ollama on your host through `host.docker.internal`. Both ports are bound to `127.0.0.1` only.
</details>

<details>
<summary><b>⚡ With Groq (free tier, larger hosted models)</b></summary>

1. Get a free key at https://console.groq.com/keys.
2. In `.env`, set `LLM_PROVIDER=groq` and `GROQ_API_KEY=...`.
3. Check https://console.groq.com/docs/models for current model names, and set `GROQ_MODEL` if the default has been retired.

You can also keep the agent local and use Groq only for the parts that check its work:

| Setting | Effect |
|---|---|
| `CHECKER_PROVIDER=groq` | The **checking agent** runs on Groq, so answers aren't checked by the model that wrote them |
| `JUDGE_PROVIDER=groq` | The **eval judge** runs on Groq, so the agent isn't graded by its own model |

The free tier has rate limits. If a full eval run hits them, use `--limit`.
</details>

## 🧠 How it works

A main graph routes each message and writes the answer. It hands two jobs to separate agents, each a **LangGraph subgraph with its own state**:

```mermaid
flowchart TB
    Q([question]) --> route{route}
    route -- small talk --> direct[respond_direct] --> A([answer])
    route -- lore --> R

    subgraph R [🔎 Retrieval agent]
        direction TB
        retrieve["retrieve<br/>dense + keyword, RRF"] --> grade[grade_docs]
        grade -- "nothing relevant (once)" --> rewrite[rewrite_query] --> retrieve
    end

    R --> P
    subgraph P [🛡️ Checking agent: premise]
        premise[check_premise]
    end

    P --> gen["generate<br/>with citations"]
    gen --> G
    subgraph G [🛡️ Checking agent: grounding]
        grounded[check_grounded]
    end

    G -- grounded --> A
    G -- "not grounded (retry once)" --> gen

    classDef orch fill:#fef3c7,stroke:#d97706,color:#1e293b
    class route,direct,gen orch
```

| Agent | Node | What it does |
|---|---|---|
| Main graph | `route` | Decides whether the message is a lore question or small talk |
| 🔎 Retrieval | `retrieve` | Runs a vector search and a keyword search and merges the two rankings (reciprocal rank fusion) |
| 🔎 Retrieval | `grade_docs` | The LLM marks the useful passages. It decides whether to search again, but it can't *drop* passages, because small models discard the second half of multi-hop answers |
| 🔎 Retrieval | `rewrite_query` | If nothing is useful, rephrases the question and searches again (once) |
| 🛡️ Checking | `check_premise` | Catches false assumptions ("Why do sky-whales migrate every *seven* years?") and switches `generate` to a correction prompt |
| Main graph | `generate` | Answers only from the passages, with `[doc#section]` citations, or refuses: *"The archives hold no record of that."* |
| 🛡️ Checking | `check_grounded` | Checks that the passages support the answer. If not, the answer is written again once with a stricter prompt |

**Why separate agents?** The checking agent can run on a **different model** (`CHECKER_PROVIDER`), so an answer isn't only checked by the model that wrote it. Each agent also gets a small state with only what it needs. Every step is tagged with the part that ran it (`orchestrator` for the main graph, `retrieval` or `checking`), in the API and in the app, where the trace panel draws the two agents as boxes.

<details>
<summary><b>🖼️ The same graph as the app draws it</b> (amber = steps that ran for a false-premise question)</summary>
<br>
<img src="docs/images/agent-graph.png" width="560" alt="Graphviz drawing of the agent graph with the retrieval and checking agents as dashed clusters">
</details>

<details>
<summary><b>🗄️ The two vector stores</b></summary>

| | Chroma (default) | PostgreSQL + pgvector |
|---|---|---|
| Dense search | Chroma, embedded | `embedding <=> query` on an **HNSW** cosine index |
| Keyword search | BM25 in Python (`rank-bm25`) | Postgres **full-text search**: generated `tsvector` column, **GIN** index, `ts_rank_cd` |
| Fusion | reciprocal rank fusion | the same reciprocal rank fusion |
| Setup | none | `docker compose up -d db` |
| Also stores | | a `queries` log of every API question (answer, citations, latency, trace as `jsonb`) |

Embeddings are `BAAI/bge-small-en-v1.5` (384 dimensions) via fastembed. They run locally on ONNX, without PyTorch or an API.
</details>

## 🔌 The API

```powershell
uv run uvicorn lorekeeper.api:app --reload          # docs: http://127.0.0.1:8000/docs
```

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | version, LLM and checker models, vector store, database status |
| `POST` | `/ask` | the answer, citations, `refused` / `grounded` / `premise`, agent-tagged steps and passages |
| `POST` | `/ask/stream` | **Server-Sent Events**: one `step` event per graph step, then `answer` |
| `GET` | `/eval/runs`, `/eval/runs/{name}` | saved eval summaries |

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI
    participant O as Main graph
    participant R as Retrieval agent
    participant K as Checking agent
    participant DB as PostgreSQL + pgvector
    C->>API: POST /ask/stream (question)
    API->>O: run graph
    O->>R: search
    R->>DB: HNSW vector search + full-text search
    DB-->>R: ranked chunks (fused with RRF)
    R-->>O: passages
    API-->>C: event: step (retrieval)
    O->>K: premise check
    O->>O: generate answer with citations
    O->>K: grounding check
    API-->>C: event: answer
    API->>DB: INSERT INTO queries
```

<details>
<summary><b>Example: a false premise, streamed</b></summary>

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "content-type: application/json" \
  -d '{"question": "Why do the sky-whales migrate every seven years?"}'
```

```text
event: step
data: {"node": "route", "steps": [{"agent": "orchestrator", "step": "route: lore"}]}

event: step
data: {"node": "retrieval_agent", "steps": [{"agent": "retrieval", "step": "retrieve #1: 4 passages"}, {"agent": "retrieval", "step": "grade_docs: 1/4 relevant"}]}

event: step
data: {"node": "premise_check", "steps": [{"agent": "checking", "step": "check_premise: FALSE (The sky-whales migrate every eleven years, not every seven years.)"}]}

event: step
data: {"node": "generate", "steps": [{"agent": "orchestrator", "step": "generate"}]}

event: step
data: {"node": "grounding_check", "steps": [{"agent": "checking", "step": "check_grounded: grounded"}]}

event: answer
data: {"answer": "Actually, the archives clearly state that the Vael undertake the Great Passage ...", "citations": [...], ...}
```
</details>

<details>
<summary><b>Example: <code>POST /ask</code></b></summary>

```bash
curl -X POST http://127.0.0.1:8000/ask -H "content-type: application/json" \
  -d '{"question": "What is driftglass used for?"}'
```

```json
{
  "answer": "Driftglass is used to create lenses, lamps, and tether stones by the Glasswrights of Kessa Vel [sky_whales#driftglass].",
  "citations": ["sky_whales#driftglass"],
  "refused": false,
  "grounded": true,
  "retrievals": 1,
  "latency_s": 1.44,
  "steps": [{"agent": "orchestrator", "step": "route: lore"}, "..."],
  "passages": [{"label": "sky_whales#driftglass", "cited": true, "...": "..."}]
}
```

Questions are stripped and must be 1–1000 characters, or you get a `422`.
</details>

## 📊 Evaluation

`eval/dataset.jsonl` has **40 hand-written items** in four types:

| Type | Count | What it tests |
|---|---|---|
| `single_hop` | 15 | The fact is in one place |
| `multi_hop` | 10 | The answer needs two documents ("Which guild is based on the current Regent's home island?") |
| `unanswerable` | 8 | The lore never says, so the agent must **refuse** rather than make something up |
| `false_premise` | 7 | The question assumes something false, so the agent must **correct** it |

```powershell
uv run python -m lorekeeper.eval --limit 5            # smoke test
uv run python -m lorekeeper.eval                      # full run
uv run python -m lorekeeper.eval --no-hybrid          # dense-only, to compare
$env:VECTOR_STORE="pgvector"; uv run python -m lorekeeper.eval   # against Postgres
uv run python -m lorekeeper.eval --include-user       # + questions saved from the game
```

### What the eval caught while building this

All runs used `qwen2.5:7b` via Ollama, top-k 4 and hybrid retrieval, judged by the same model.

```mermaid
xychart-beta
    title "Correctness by version (40 items)"
    x-axis ["v1", "v2", "v3", "v4", "multi-agent", "pgvector"]
    y-axis "correctness" 0 --> 1
    bar [0.78, 0.76, 0.84, 0.86, 0.86, 0.90]
```

| Version | Change | Overall | Multi-hop | False premise |
|---|---|---|---|---|
| v1 | Baseline | 0.78 | 0.45 | 0.50 |
| v2 | Stop the grader dropping passages (it was discarding the second doc in multi-hop questions) | 0.76 | 0.70 | 0.07 |
| v3 | Add a `check_premise` node (the 7B model kept answering false premises with "no record") | 0.84 | 0.55 | 0.86 |
| v4 | Better refusal detection; "not mentioned" no longer counts as a false premise | 0.86 | 0.65 | 0.86 |
| multi-agent | Split into retrieval and checking agents (same model, so this checks the refactor changed nothing) | 0.86 | 0.65 | 0.86 |
| pgvector | Postgres: HNSW + full-text search instead of Chroma + BM25 | **0.90** | 0.70 | 0.86 |

<details>
<summary><b>Retrieval: Chroma vs pgvector</b></summary>

| Store | hit@4 | recall@4 | MRR | Faithfulness |
|---|---|---|---|---|
| Chroma + BM25 | 1.0 | 0.95 | 0.92 | 0.86 |
| pgvector + Postgres full-text | 1.0 | 0.92 | 0.97 | 0.89 |

Postgres full-text search ranks the first gold document higher (MRR), but finds the second multi-hop document a little less often (recall). Dense-only Chroma scored 0.875 against 0.86 for hybrid. On a 12-document corpus, differences this small are within judge noise.
</details>

<details>
<summary><b>Metric definitions</b></summary>

| Metric | What it measures |
|---|---|
| hit@k, recall@k, MRR | Retrieval quality against the gold `source_docs` |
| Correctness | LLM judge scores 0, ½ or 1 against the reference answer. For unanswerable items, 1 if the agent refused |
| Faithfulness | LLM judge checks whether the answer is supported by the passages used |
| Refusal accuracy | Refused when it should, and answered when it should |
| Latency, rewrites, graph steps | Speed and how hard the agent had to work |

Results are saved to `eval/results/*.json`. The **Eval Dashboard** page and `GET /eval/runs` compare runs.
</details>

> [!WARNING]
> By default the judge is the same 7B model as the agent, and models tend to be lenient with themselves. Treat differences of a few points as noise. Set `JUDGE_PROVIDER=groq` for an independent grader.

## ⚔️ Quiz Duel

You against the Lorekeeper, 10 questions, a 30-second clock.

- The agent starts searching the moment a question appears, while you type.
- A judge scores you both. You get points for correctness, a speed bonus, and streak multipliers up to ×2.
- After each round the agent replies in character, teasing you or conceding.
- Ranks: **Driftling → Tide Apprentice → Guild Scribe → Archivist → Keeper of Veyra**. The leaderboard is saved to `highscores.json`.
- **Stump the Lorekeeper:** write your own question. If the agent gets it wrong or bluffs, you get **+150** and the question joins `eval/user_submitted.jsonl`, so playing the game grows the eval set.
- The quiz bank is kept separate from the eval set, so the agent isn't tuned on its own test.

<details>
<summary><b>🎲 Try these in the chat</b></summary>

| Question | What it tests |
|---|---|
| Who was Regent during the Stormwar? | single-hop answer with a citation |
| Which guild is based on the current Regent's home island? | multi-hop: needs two documents (the 7B model doesn't always get there) |
| Why do the sky-whales migrate every seven years? | false premise: it should correct you |
| What is the population of Duskholm? | unanswerable: it should refuse |
| Hello! Who are you? | small talk: no retrieval at all |

</details>

## 🧪 Tests

```powershell
uv run pytest                                                   # fake LLM: no Ollama, no network
docker compose up -d db
$env:DATABASE_URL="postgresql://lorekeeper:lorekeeper@127.0.0.1:5432/lorekeeper"
uv run pytest tests/test_pgvector.py                            # against real Postgres
```

<details>
<summary><b>What's covered</b></summary>

- **Agents:** chitchat routing, the grounded path, rewriting and then refusing, the regenerate retry, the checking agent using its *own* model (separate fakes prove it), steps tagged per agent, and the retrieval agent running on its own
- **API:** `/ask`, `/ask/stream` event order, error events, 422 validation, eval run listing, path checks
- **Postgres:** schema and index build, full-text and vector search, hybrid retrieval, the `queries` log (using a hashing embedder, so no model is downloaded)
- citation filtering, chunking, reciprocal rank fusion, retrieval metrics, judge parsing, game scoring, data-file validity

CI runs two jobs: `test` (fake LLM) and `pgvector` (a `pgvector/pgvector:pg17` service container).
</details>

## 🔍 Design notes and limits

- **Invented documents** mean the model can't answer from memory, so the eval tests retrieval and answering from passages.
- **The grader can't drop passages.** v2 showed a 7B grader discarding the second document of multi-hop questions.
- **Refusing can be the right answer.** Unanswerable questions score 1 only if the agent refuses.
- **Limits:** the corpus is small (12 documents, 59 chunks), the default judge is the agent's own model, and the remaining misses are mostly date arithmetic and multi-hop chains the 7B model doesn't finish.

<details>
<summary><b>📁 Layout</b></summary>

```
data/lore/            12 lore documents (edit or add your own, then re-run ingest)
data/quiz_bank.json   quiz questions
eval/                 dataset.jsonl, user_submitted.jsonl, results/
src/lorekeeper/
  graph.py            main graph + retrieval agent + checking agent
  retriever.py        Chroma + BM25 hybrid retrieval
  pgstore.py          PostgreSQL + pgvector store, in-database hybrid search
  api.py              FastAPI service
  ingest.py · llm.py · prompts.py · judge.py · metrics.py · eval.py · game.py · config.py
app/                  Streamlit: Home (chat), Quiz Duel, Eval Dashboard
tests/
Dockerfile · docker-compose.yml
```
</details>

<details>
<summary><b>💡 Ideas to extend</b></summary>

- Add a reranker (e.g. a fastembed cross-encoder) and measure the gain on `multi_hop`.
- Write more lore, and check that `unanswerable` items stay unanswerable.
- Track eval scores over time in CI with `--limit` and a small Groq judge.
- Read the `queries` table to find questions people ask that the eval set doesn't cover.
</details>

## License

[MIT](LICENSE). The code and the lore of Veyra are both free to use, modify and share.

The video was made with [/brag](https://github.com/latent-spaces/brag). Music: "Happy Beats / Business Moves Vol. 10" by Sascha Ende, [ende.app](https://ende.app/en) ([CC BY 4.0](https://ende.app/en/standard-license)).
