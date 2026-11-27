"""LLM-as-judge helpers, used by both the eval runner and the quiz game."""

import json
import re

from langchain_core.language_models import BaseChatModel

from lorekeeper import prompts
from lorekeeper.llm import chat


def parse_score(text: str) -> tuple[float, str]:
    """Pull {"score": x, "reason": "..."} out of a model reply, tolerating messy output."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return snap(float(data.get("score", 0))), str(data.get("reason", ""))
        except (ValueError, TypeError):
            pass
    num = re.search(r"score\D{0,5}(0\.5|1(?:\.0)?|0(?:\.0)?)", text, re.I)
    return (snap(float(num.group(1))), text[:200]) if num else (0.0, f"unparseable judge reply: {text[:120]}")


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\[\]]+#[^\[\]]+\]", "", text).strip()


def snap(x: float) -> float:
    return min((0.0, 0.5, 1.0), key=lambda v: abs(v - x))


def judge_correctness(llm: BaseChatModel, question: str, reference: str, answer: str) -> tuple[float, str]:
    answer = strip_citations(answer)
    if not answer:
        return 0.0, "empty answer"
    user = f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\nANSWER TO GRADE: {answer}"
    return parse_score(chat(llm, prompts.JUDGE_CORRECTNESS, user))


def judge_faithfulness(llm: BaseChatModel, answer: str, context: str) -> bool:
    reply = chat(llm, prompts.JUDGE_FAITHFUL, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}")
    return reply.lower().startswith("y")
