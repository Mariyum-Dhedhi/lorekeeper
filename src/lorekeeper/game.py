"""Quiz Duel game logic. Pure functions plus a small state object, so it's easy to test."""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lorekeeper.config import HIGHSCORES, QUIZ_BANK, USER_SUBMITTED

ROUNDS = 10
TIME_LIMIT_S = 30
STUMP_BONUS = 150

RANKS = [
    (0, "Driftling"),
    (300, "Tide Apprentice"),
    (700, "Guild Scribe"),
    (1200, "Archivist"),
    (1800, "Keeper of Veyra"),
]

TAUNTS = {
    "both_right": [
        "A draw. I suppose even a Driftling can read a map now and then.",
        "We agree. How unsettling for both of us.",
        "Correct, both of us. The archives approve.",
    ],
    "player_only": [
        "...I was testing you. Obviously.",
        "Hmph. Someone has been sneaking into the Great Library after hours.",
        "Mark it down. Quietly. Very quietly.",
    ],
    "agent_only": [
        "The archives never forget. Visitors, apparently, do.",
        "Fair tides, and better luck next round.",
        "Shall I lend you a library card?",
    ],
    "both_wrong": [
        "Let us both agree this question never happened.",
        "Clearly a misfiled scroll. I'll be having words with the Scribes.",
        "Even the Silent Observatory makes more noise than our answers did.",
    ],
}


def load_quiz_bank(path: Path = QUIZ_BANK) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def round_points(score: float, streak: int, seconds: float | None = None) -> int:
    """Points for one round.

    score: judge score 0 / 0.5 / 1
    streak: consecutive correct answers *before* this round (multiplier caps at 2x)
    seconds: time taken to answer; only humans get a speed bonus (None = no bonus)
    """
    base = 100 * score
    speed = 0.0
    if seconds is not None and score >= 0.5:
        speed = 50 * max(0.0, 1 - seconds / TIME_LIMIT_S)
    multiplier = 1 + 0.25 * min(streak, 4)
    return int(round((base + speed) * multiplier))


def rank_for(score: int) -> str:
    title = RANKS[0][1]
    for threshold, name in RANKS:
        if score >= threshold:
            title = name
    return title


def next_rank(score: int) -> tuple[str, int] | None:
    for threshold, name in RANKS:
        if score < threshold:
            return name, threshold - score
    return None


def outcome(player_score: float, agent_score: float) -> str:
    p, a = player_score >= 0.5, agent_score >= 0.5
    return {(True, True): "both_right", (True, False): "player_only",
            (False, True): "agent_only", (False, False): "both_wrong"}[(p, a)]


def taunt(kind: str, rng: random.Random | None = None) -> str:
    return (rng or random).choice(TAUNTS[kind])


@dataclass
class Duel:
    questions: list[dict]
    player_score: int = 0
    agent_score: int = 0
    player_streak: int = 0
    agent_streak: int = 0
    best_streak: int = 0
    round: int = 0
    history: list[dict] = field(default_factory=list)

    @classmethod
    def new(cls, bank: list[dict], rounds: int = ROUNDS, seed: int | None = None) -> "Duel":
        rng = random.Random(seed)
        return cls(questions=rng.sample(bank, min(rounds, len(bank))))

    @property
    def finished(self) -> bool:
        return self.round >= len(self.questions)

    @property
    def current(self) -> dict:
        return self.questions[self.round]

    def record(self, player_judge: float, agent_judge: float, seconds: float, extra: dict | None = None) -> dict:
        p_pts = round_points(player_judge, self.player_streak, seconds)
        a_pts = round_points(agent_judge, self.agent_streak)
        self.player_score += p_pts
        self.agent_score += a_pts
        self.player_streak = self.player_streak + 1 if player_judge >= 0.5 else 0
        self.agent_streak = self.agent_streak + 1 if agent_judge >= 0.5 else 0
        self.best_streak = max(self.best_streak, self.player_streak)
        kind = outcome(player_judge, agent_judge)
        entry = {
            "round": self.round + 1,
            "question": self.current["question"],
            "reference": self.current["answer"],
            "player_judge": player_judge,
            "agent_judge": agent_judge,
            "player_points": p_pts,
            "agent_points": a_pts,
            "seconds": round(seconds, 1),
            "outcome": kind,
            "taunt": taunt(kind),
            **(extra or {}),
        }
        self.history.append(entry)
        self.round += 1
        return entry


# ---- persistence -----------------------------------------------------------

def load_highscores(path: Path = HIGHSCORES) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_highscore(name: str, duel: Duel, path: Path = HIGHSCORES) -> list[dict]:
    scores = load_highscores(path)
    scores.append({
        "name": name or "Anonymous Driftling",
        "score": duel.player_score,
        "agent_score": duel.agent_score,
        "rank": rank_for(duel.player_score),
        "best_streak": duel.best_streak,
        "beat_the_lorekeeper": duel.player_score > duel.agent_score,
        "date": f"{datetime.now():%Y-%m-%d %H:%M}",
    })
    scores.sort(key=lambda s: s["score"], reverse=True)
    path.write_text(json.dumps(scores[:50], indent=2), encoding="utf-8")
    return scores


def save_stumper(question: str, reference_answer: str, agent_answer: str, path: Path = USER_SUBMITTED) -> dict:
    """Save a question that stumped the agent so it becomes part of the eval set."""
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    item = {
        "id": f"user{len([l for l in existing if l.strip()]) + 1:03d}",
        "type": "user_submitted" if reference_answer.strip() else "unanswerable",
        "difficulty": "user",
        "question": question,
        "reference_answer": reference_answer.strip() or "The archives do not record this.",
        "source_docs": [],
        "agent_answer_at_submission": agent_answer,
        "submitted": f"{datetime.now():%Y-%m-%d %H:%M}",
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")
    return item
