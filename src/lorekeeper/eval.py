"""Run the evaluation set against the agent.

Examples:
    uv run python -m lorekeeper.eval --limit 5
    uv run python -m lorekeeper.eval --no-hybrid --top-k 3
    uv run python -m lorekeeper.eval --include-user     # also run questions saved from the quiz
"""

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime

from lorekeeper.config import EVAL_DATASET, EVAL_RESULTS, USER_SUBMITTED, get_settings
from lorekeeper.graph import ask, format_passages, load_agent
from lorekeeper.judge import judge_correctness, judge_faithfulness
from lorekeeper.llm import get_judge_llm, provider_label
from lorekeeper.metrics import hit_at_k, mean, mrr, recall_at_k

ANSWERABLE = {"single_hop", "multi_hop", "false_premise", "user_submitted"}


def load_dataset(include_user: bool = False) -> list[dict]:
    items = [json.loads(line) for line in EVAL_DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if include_user and USER_SUBMITTED.exists():
        items += [json.loads(l) for l in USER_SUBMITTED.read_text(encoding="utf-8").splitlines() if l.strip()]
    return items


def evaluate_item(agent, judge, item: dict) -> dict:
    res = ask(agent, item["question"])
    retrieved = [d.metadata["source"] for d in res["docs"]]
    gold = item.get("source_docs", [])
    unanswerable = item["type"] == "unanswerable"

    if unanswerable:
        correctness, reason = (1.0, "correctly refused") if res["refused"] else (0.0, "answered an unanswerable question")
    else:
        correctness, reason = judge_correctness(judge, item["question"], item["reference_answer"], res["answer"])

    faithful = None
    if not res["refused"] and res["relevant_docs"]:
        faithful = judge_faithfulness(judge, res["answer"], format_passages(res["relevant_docs"]))

    return {
        "id": item["id"],
        "type": item["type"],
        "difficulty": item.get("difficulty"),
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "answer": res["answer"],
        "citations": res["citations"],
        "retrieved_sources": retrieved,
        "gold_sources": gold,
        "hit_at_k": hit_at_k(retrieved, gold) if gold else None,
        "recall_at_k": recall_at_k(retrieved, gold) if gold else None,
        "mrr": mrr(retrieved, gold) if gold else None,
        "correctness": correctness,
        "judge_reason": reason,
        "faithful": None if faithful is None else float(faithful),
        "refused": res["refused"],
        "refusal_correct": float(res["refused"] == unanswerable),
        "route": res["route"],
        "attempts": res["attempts"],
        "steps": len(res["trace"]),
        "trace": res["trace"],
        "latency_s": res["latency_s"],
    }


def summarize(rows: list[dict]) -> dict:
    def agg(rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "correctness": mean([r["correctness"] for r in rs]),
            "hit_at_k": mean([r["hit_at_k"] for r in rs]),
            "recall_at_k": mean([r["recall_at_k"] for r in rs]),
            "mrr": mean([r["mrr"] for r in rs]),
            "faithfulness": mean([r["faithful"] for r in rs]),
            "refusal_accuracy": mean([r["refusal_correct"] for r in rs]),
            "avg_rewrites": mean([max(r["attempts"] - 1, 0) for r in rs]),
            "avg_steps": mean([r["steps"] for r in rs]),
            "avg_latency_s": mean([r["latency_s"] for r in rs]),
        }

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"overall": agg(rows), "by_type": {t: agg(rs) for t, rs in sorted(by_type.items())}}


def print_table(summary: dict) -> None:
    cols = ["n", "correctness", "hit_at_k", "recall_at_k", "mrr", "faithfulness", "refusal_accuracy", "avg_latency_s"]
    header = f"{'':16}" + "".join(f"{c[:12]:>13}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, stats in [*summary["by_type"].items(), ("OVERALL", summary["overall"])]:
        cells = "".join(f"{'-' if stats[c] is None else stats[c]:>13}" for c in cols)
        print(f"{name:16}{cells}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the Lorekeeper agent")
    p.add_argument("--limit", type=int, help="only run the first N items")
    p.add_argument("--top-k", type=int, help="override TOP_K")
    p.add_argument("--no-hybrid", action="store_true", help="dense retrieval only (no BM25)")
    p.add_argument("--include-user", action="store_true", help="include questions saved from Stump the Lorekeeper")
    p.add_argument("--tag", default="", help="label added to the results filename")
    args = p.parse_args()

    s = get_settings()
    hybrid = False if args.no_hybrid else s.hybrid
    k = args.top_k or s.top_k
    items = load_dataset(args.include_user)[: args.limit]

    agent, judge = load_agent(k=k, hybrid=hybrid), get_judge_llm()
    config = {
        "provider": provider_label(),
        "judge": provider_label(s.judge_provider),
        "top_k": k,
        "hybrid": hybrid,
        "vector_store": s.vector_store,
        "n_items": len(items),
    }
    print(f"Evaluating {len(items)} items with {config}")

    rows, start = [], time.perf_counter()
    for i, item in enumerate(items, 1):
        row = evaluate_item(agent, judge, item)
        rows.append(row)
        mark = {1.0: "OK  ", 0.5: "HALF"}.get(row["correctness"], "MISS")
        print(f"[{i:>2}/{len(items)}] {mark} {item['id']:<5} {row['latency_s']:>6}s  {item['question'][:70]}")

    summary = summarize(rows)
    print_table(summary)

    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    provider_slug = re.sub(r"[^\w.]+", "-", config["provider"])
    store = "" if s.vector_store == "chroma" else f"_{s.vector_store}"
    name = f"{datetime.now():%Y%m%d-%H%M%S}_{provider_slug}_k{k}_{'hybrid' if hybrid else 'dense'}{store}"
    out = EVAL_RESULTS / f"{name}{'_' + args.tag if args.tag else ''}.json"
    out.write_text(
        json.dumps({"config": config, "summary": summary, "rows": rows,
                    "total_time_s": round(time.perf_counter() - start, 1)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
