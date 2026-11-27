# 📜 Lorekeeper

A **RAG agent built with LangGraph** that guards the archives of **Veyra**, an invented island realm of guilds, sky-whales, and one ill-advised war. It comes with a **40-question evaluation set** and a **Quiz Duel** where you compete against the agent.

The knowledge base is fictional on purpose: the LLM can't answer from what it memorized in training, so the eval measures *retrieval and grounding*, not trivia recall.

Everything runs **free**. The default LLM is local (Ollama). Groq's free tier is an optional switch. Embeddings run locally too.

## Quick start

```powershell
# 1. Local LLM (skip if you already have it)
ollama pull qwen2.5:7b

# 2. Install
cd lorekeeper
uv sync
copy .env.example .env

# 3. Build the index (the app also does this automatically on first run)
uv run python -m lorekeeper.ingest

# 4. Play
uv run streamlit run app/Home.py
```

The first run downloads the ~70 MB embedding model into `.cache/`.

### Optional: Groq (free tier, faster and stronger)
1. Get a free key at https://console.groq.com/keys.
2. In `.env`, set `LLM_PROVIDER=groq` and `GROQ_API_KEY=...`.
3. Check https://console.groq.com/docs/models for current model names, and set `GROQ_MODEL` if the default has been retired.

The free tier has rate limits. If a full eval run hits them, use `--limit`.

You can also keep the agent local and use Groq only as the judge, which gives you a stronger grader: set `JUDGE_PROVIDER=groq`.

## The agent

```
route ──chitchat──▶ respond_direct ──▶ END
  │
 lore
  ▼
retrieve ──▶ grade_docs ──relevant──▶ check_premise ──▶ generate ──▶ check_grounded ──grounded──▶ END
  ▲              │                                        ▲              │
  └─ rewrite ◀───┘ nothing relevant                       └── retry once ┘ not grounded
```

| Node | What it does |
|---|---|
| `route` | Decides whether the message is a lore question or small talk |
| `retrieve` | Hybrid search: dense vectors (Chroma + fastembed `bge-small`) fused with BM25 via reciprocal rank fusion |
| `grade_docs` | The LLM keeps only the passages that are actually useful |
| `rewrite_query` | If nothing is useful, rephrases the question and searches again (once) |
| `check_premise` | Detects false assumptions ("Why do sky-whales migrate every *seven* years?") and switches `generate` to a correction prompt |
| `generate` | Answers from the passages, with graded-relevant ones first, using `[doc#section]` citations. Refuses with "The archives hold no record of that." when the passages don't cover the question |
| `check_grounded` | Hallucination check; regenerates once with a stricter prompt if unsupported |

In the app, the *"How the Lorekeeper thought"* panel under each answer shows the graph with the nodes that ran highlighted, plus the step-by-step trace.

## Evaluation

`eval/dataset.jsonl` has 40 hand-written items in four types:

| Type | Count | Tests |
|---|---|---|
| `single_hop` | 15 | The fact is in one place |
| `multi_hop` | 10 | The answer needs two documents (e.g. "Which guild is based on the current Regent's home island?") |
| `unanswerable` | 8 | The lore never says, so the agent must refuse rather than make something up |
| `false_premise` | 7 | The question assumes something false, so the agent must correct it |

```powershell
uv run python -m lorekeeper.eval --limit 5            # smoke test
uv run python -m lorekeeper.eval                      # full run
uv run python -m lorekeeper.eval --no-hybrid          # dense-only, to compare
uv run python -m lorekeeper.eval --top-k 2 --tag k2   # fewer passages
uv run python -m lorekeeper.eval --include-user       # + questions saved from the game
```

Metrics, overall and per type:

| Metric | What it measures |
|---|---|
| hit@k, recall@k, MRR | Retrieval quality against the gold `source_docs` |
| Correctness | LLM judge scores 0, ½, or 1 against the reference answer. For unanswerable items, 1 if the agent refused |
| Faithfulness | LLM judge checks whether the answer is supported by the passages used |
| Refusal accuracy | Refused when it should, and answered when it should |
| Latency, rewrites, graph steps | Speed and how hard the agent had to work |

Results are saved to `eval/results/*.json`. The **Eval Dashboard** page compares two runs side by side.

### What the eval caught while building this
Measured on `qwen2.5:7b` via Ollama, top-k 4, hybrid retrieval, judged by the same model. Retrieval hit@4 was 1.0 throughout, so every gain below came from the agent itself.

| Version | Change | Overall | Multi-hop | False premise |
|---|---|---|---|---|
| v1 | Baseline | 0.78 | 0.45 | 0.50 |
| v2 | Stop the grader dropping passages (it was discarding the second doc in multi-hop questions) | 0.76 | 0.70 | 0.07 |
| v3 | Add a `check_premise` node (the 7B model kept answering false premises with "no record") | 0.84 | 0.55 | 0.86 |
| v4 | Better refusal detection; "not mentioned" no longer counts as a false premise | **0.86** | 0.65 | 0.86 |

On this small corpus, dense-only retrieval scored 0.875 against 0.86 for hybrid, which is within judge noise. The remaining misses are mostly date arithmetic (m02, m06) and the 7B judge misgrading correct answers.

> **Caveat:** by default the judge is the same model as the agent, and models tend to be lenient with themselves. Set `JUDGE_PROVIDER=groq` for a stronger, independent grader.

## The fun part: Quiz Duel ⚔️

- 10 questions from `data/quiz_bank.json`, which is kept separate from the eval set so the agent isn't tuned on its own test.
- The agent starts searching the moment a question appears, while you type against a 30-second clock.
- A judge scores you both. You get points for correctness, a speed bonus, and streak multipliers up to ×2.
- The Lorekeeper delivers an in-character taunt or concession after each round.
- Ranks: Driftling → Tide Apprentice → Guild Scribe → Archivist → Keeper of Veyra. The leaderboard is saved to `highscores.json`.
- **Stump the Lorekeeper:** write your own question. If the agent gets it wrong or bluffs, you get +150 points and the question is added to `eval/user_submitted.jsonl`. Playing the game grows the eval set.

## Tests

```powershell
uv run pytest
```

The tests use a fake LLM, so they need neither Ollama nor network access. They cover:
- graph routing: chitchat, the grounded path, rewriting then refusing, and the regenerate retry
- citation filtering
- chunking
- reciprocal rank fusion
- retrieval metrics
- judge output parsing
- game scoring and persistence
- data-file validity

## Layout

```
data/lore/          12 lore documents (edit or add your own, then re-run ingest)
data/quiz_bank.json quiz questions
eval/               dataset.jsonl, user_submitted.jsonl, results/
src/lorekeeper/     config, llm, ingest, retriever, graph, prompts, judge, metrics, eval, game
app/                Streamlit: Home (chat), Quiz Duel, Eval Dashboard
tests/
```

## License
[MIT](LICENSE). The code and the lore of Veyra are both free to use, modify, and share.

## Ideas to extend
- Add a reranker (e.g. a fastembed cross-encoder) and measure the gain on `multi_hop`.
- Write more lore, and check that `unanswerable` items stay unanswerable.
- Track eval scores over time in CI with `--limit` and a small Groq judge.
