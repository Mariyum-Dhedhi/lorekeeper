import json

from lorekeeper.config import EVAL_DATASET, LORE_DIR
from lorekeeper.game import Duel, load_quiz_bank, next_rank, rank_for, round_points, save_highscore, save_stumper


def test_round_points():
    assert round_points(1.0, streak=0) == 100
    assert round_points(0.5, streak=0) == 50
    assert round_points(0.0, streak=3, seconds=1) == 0
    assert round_points(1.0, streak=0, seconds=0) == 150  # full speed bonus
    assert round_points(1.0, streak=0, seconds=60) == 100  # too slow, no bonus
    assert round_points(1.0, streak=4) == 200  # 2x cap
    assert round_points(1.0, streak=10) == 200


def test_ranks():
    assert rank_for(0) == "Driftling"
    assert rank_for(299) == "Driftling"
    assert rank_for(300) == "Tide Apprentice"
    assert rank_for(5000) == "Keeper of Veyra"
    assert next_rank(250) == ("Tide Apprentice", 50)
    assert next_rank(5000) is None


def test_duel_streaks_and_history():
    bank = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]
    duel = Duel.new(bank, rounds=3, seed=1)
    duel.record(1.0, 0.0, seconds=30)   # 100, agent 0
    duel.record(1.0, 1.0, seconds=30)   # 125 (streak 1), agent 100
    duel.record(0.0, 1.0, seconds=30)   # 0, agent 125
    assert duel.finished
    assert (duel.player_score, duel.agent_score) == (225, 225)
    assert duel.player_streak == 0 and duel.best_streak == 2
    assert [h["outcome"] for h in duel.history] == ["player_only", "both_right", "agent_only"]


def test_persistence(tmp_path):
    duel = Duel.new([{"question": "q", "answer": "a"}], rounds=1)
    duel.record(1.0, 0.0, seconds=0)
    scores = save_highscore("Tester", duel, tmp_path / "hs.json")
    assert scores[0]["name"] == "Tester" and scores[0]["beat_the_lorekeeper"]

    p = tmp_path / "user.jsonl"
    a = save_stumper("What is X?", "", "X is Y", p)
    b = save_stumper("What is Z?", "Z", "dunno", p)
    assert a["type"] == "unanswerable" and b["type"] == "user_submitted"
    assert a["id"] == "user001" and b["id"] == "user002"


def test_data_files_are_valid():
    assert len(load_quiz_bank()) >= 30
    items = [json.loads(l) for l in EVAL_DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(items) == 40
    assert len({i["id"] for i in items}) == 40
    types = {i["type"] for i in items}
    assert types == {"single_hop", "multi_hop", "unanswerable", "false_premise"}
    lore = {p.stem for p in LORE_DIR.glob("*.md")}
    for i in items:
        assert set(i["source_docs"]) <= lore, f"{i['id']} cites a missing doc"
        assert bool(i["source_docs"]) == (i["type"] != "unanswerable")
