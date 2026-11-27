import json

import pandas as pd
import streamlit as st

from common import sidebar_badge
from lorekeeper.config import EVAL_RESULTS

st.set_page_config(page_title="Eval Dashboard", page_icon="📊", layout="wide")
sidebar_badge()
st.title("📊 Eval Dashboard")

files = sorted(EVAL_RESULTS.glob("*.json"), reverse=True)
if not files:
    st.info("No eval runs yet. Run `uv run python -m lorekeeper.eval --limit 5` (or without `--limit` for the full set).")
    st.stop()

picked = st.multiselect(
    "Pick a run, or two to compare (first = baseline)",
    files, default=files[:1], format_func=lambda p: p.stem, max_selections=2,
)
if not picked:
    st.stop()

runs = [json.loads(p.read_text(encoding="utf-8")) for p in picked]
METRICS = {
    "correctness": "Answer correctness",
    "hit_at_k": "Retrieval hit@k",
    "recall_at_k": "Retrieval recall@k",
    "mrr": "MRR",
    "faithfulness": "Faithfulness",
    "refusal_accuracy": "Refusal accuracy",
    "avg_latency_s": "Avg latency (s)",
}

cols = st.columns(len(runs))
for col, run, path in zip(cols, runs, picked):
    col.caption(f"**{path.stem}**")
    col.json(run["config"], expanded=False)

# metric cards, with deltas vs the baseline when comparing
latest, base = runs[-1]["summary"]["overall"], runs[0]["summary"]["overall"]
cards = st.columns(len(METRICS))
for card, (key, name) in zip(cards, METRICS.items()):
    val = latest.get(key)
    delta = None
    if len(runs) == 2 and val is not None and base.get(key) is not None:
        delta = round(val - base[key], 3)
    card.metric(name, "-" if val is None else val, delta,
                delta_color="inverse" if key == "avg_latency_s" else "normal")

st.subheader("By question type")
metric = st.selectbox("Metric", list(METRICS), format_func=METRICS.get)
frame = pd.DataFrame({
    p.stem[-40:]: {t: s.get(metric) for t, s in r["summary"]["by_type"].items()}
    for p, r in zip(picked, runs)
})
st.bar_chart(frame, stack=False, height=320)
st.dataframe(pd.DataFrame(runs[-1]["summary"]["by_type"]).T, width="stretch")

st.subheader("Where it went wrong")
rows = pd.DataFrame(runs[-1]["rows"])
types = st.multiselect("Types", sorted(rows["type"].unique()), default=sorted(rows["type"].unique()))
only_fail = st.toggle("Only show misses (correctness < 1)", value=True)
view = rows[rows["type"].isin(types)]
if only_fail:
    view = view[view["correctness"] < 1]
st.dataframe(
    view[["id", "type", "question", "reference_answer", "answer", "correctness", "judge_reason",
          "retrieved_sources", "gold_sources", "faithful", "refused", "latency_s"]],
    hide_index=True, width="stretch",
)
with st.expander("Inspect an item's graph trace"):
    if len(view):
        item_id = st.selectbox("Item", view["id"])
        r = rows[rows["id"] == item_id].iloc[0]
        st.code("\n".join(r["trace"]), language=None)
