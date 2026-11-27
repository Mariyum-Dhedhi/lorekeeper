import time

import pandas as pd
import streamlit as st

from common import get_agent, get_executor, get_judge, render_result, sidebar_badge
from lorekeeper.game import (
    STUMP_BONUS,
    TIME_LIMIT_S,
    Duel,
    load_highscores,
    load_quiz_bank,
    next_rank,
    rank_for,
    save_highscore,
    save_stumper,
)
from lorekeeper.graph import ask
from lorekeeper.judge import judge_correctness

st.set_page_config(page_title="Quiz Duel", page_icon="⚔️", layout="wide")
sidebar_badge()
ss = st.session_state
ss.setdefault("phase", "intro")
ss.setdefault("stump_points", 0)


def countdown(start_epoch: float, limit: int = TIME_LIMIT_S) -> None:
    """A purely visual countdown that survives Streamlit reruns (it counts from a fixed start time)."""
    st.iframe(
        f"""
        <div id="t" style="font:600 15px system-ui,sans-serif;padding:6px 2px;color:#334155"></div>
        <div style="height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden">
          <div id="b" style="height:100%;background:#16a34a;width:100%;transition:width .25s linear"></div></div>
        <script>
          const start = {start_epoch * 1000}, limit = {limit};
          function tick() {{
            const left = Math.max(0, limit - (Date.now() - start) / 1000);
            const t = document.getElementById('t'), b = document.getElementById('b');
            b.style.width = (left / limit * 100) + '%';
            b.style.background = left > limit / 2 ? '#16a34a' : left > 5 ? '#d97706' : '#dc2626';
            t.textContent = left > 0 ? '⏳ ' + left.toFixed(0) + 's left for the speed bonus'
                                     : "⌛ Time's up! You can still answer, but the speed bonus is gone.";
            if (left > 0) setTimeout(tick, 250);
          }}
          tick();
        </script>""",
        height=48,
    )


def start_round() -> None:
    duel: Duel = ss.duel
    ss.q_start = time.time()
    ss.agent_future = get_executor().submit(ask, get_agent(), duel.current["question"])
    ss.phase = "question"


def scoreboard(duel: Duel) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("🧑 You", duel.player_score, help=f"Streak: {duel.player_streak}")
    c2.metric("📜 Lorekeeper", duel.agent_score, help=f"Streak: {duel.agent_streak}")
    c3.metric("Round", f"{min(duel.round + (ss.phase == 'question'), len(duel.questions))}/{len(duel.questions)}")
    flames = lambda n: "🔥" * min(n, 5) + (f" ×{1 + 0.25 * min(n, 4):.2f}" if n else "")
    st.caption(f"Your streak: {flames(duel.player_streak) or '-'}  ·  Lorekeeper streak: {flames(duel.agent_streak) or '-'}")


st.title("⚔️ Quiz Duel: You vs the Lorekeeper")
duel_tab, stump_tab, board_tab = st.tabs(["Duel", "Stump the Lorekeeper", "Leaderboard"])

# ---------------------------------------------------------------- Duel
with duel_tab:
    if ss.phase == "intro":
        st.markdown(
            f"""
**How it works.** Ten trivia questions about Veyra. You and the Lorekeeper answer the same question.
The Lorekeeper must *look things up*, while you rely on memory (or on reading the **Ask the Lorekeeper** chat beforehand, no judgement).
A judge scores both answers (full, half, or no credit).

- **100 points** for a correct answer, 50 for half credit
- **Speed bonus** up to +50 if you answer within {TIME_LIMIT_S}s
- **Streaks** multiply points: up to ×2 after four in a row
- Ranks: Driftling → Tide Apprentice → Guild Scribe → Archivist → Keeper of Veyra
"""
        )
        if st.button("Begin the duel", type="primary"):
            ss.duel = Duel.new(load_quiz_bank())
            start_round()
            st.rerun()

    elif ss.phase == "question":
        duel: Duel = ss.duel
        scoreboard(duel)
        st.subheader(f"Q{duel.round + 1}. {duel.current['question']}")
        countdown(ss.q_start)
        st.caption("📜 The Lorekeeper is already rummaging through the stacks...")
        with st.form(f"answer_{duel.round}", clear_on_submit=True):
            answer = st.text_input("Your answer", key=f"ans_{duel.round}")
            submitted = st.form_submit_button("Lock in answer", type="primary")
        if submitted:
            seconds = time.time() - ss.q_start
            q, ref = duel.current["question"], duel.current["answer"]
            with st.spinner("The judge is weighing both answers..."):
                try:
                    agent_res = ss.agent_future.result()
                except Exception as e:
                    st.error(f"The Lorekeeper fainted: {e}. Is Ollama running?")
                    st.stop()
                judge = get_judge()
                p_score, p_reason = judge_correctness(judge, q, ref, answer)
                a_score, a_reason = judge_correctness(judge, q, ref, agent_res["answer"])
            ss.last = duel.record(
                p_score, a_score, seconds,
                extra={"player_answer": answer, "player_reason": p_reason,
                       "agent_answer": agent_res["answer"], "agent_reason": a_reason, "agent_result": agent_res},
            )
            ss.phase = "reveal"
            st.rerun()

    elif ss.phase == "reveal":
        duel: Duel = ss.duel
        e = ss.last
        scoreboard(duel)
        st.subheader(f"Q{e['round']}. {e['question']}")
        st.success(f"**Answer:** {e['reference']}")
        icon = {1.0: "✅", 0.5: "🟡", 0.0: "❌"}
        left, right = st.columns(2)
        with left:
            st.markdown(f"#### 🧑 You {icon[e['player_judge']]} +{e['player_points']}")
            st.markdown(f"> {e['player_answer'] or '_(silence)_'}")
            st.caption(f"{e['seconds']}s · judge: {e['player_reason']}")
        with right:
            st.markdown(f"#### 📜 Lorekeeper {icon[e['agent_judge']]} +{e['agent_points']}")
            st.markdown(f"> {e['agent_answer']}")
            st.caption(f"judge: {e['agent_reason']}")
            render_result(e["agent_result"], show_graph=False)
        st.info(f"📜 *\"{e['taunt']}\"*")
        if st.button("Next question" if not duel.finished else "See final results", type="primary"):
            if duel.finished:
                ss.phase = "done"
            else:
                start_round()
            st.rerun()

    elif ss.phase == "done":
        duel: Duel = ss.duel
        total = duel.player_score + ss.stump_points
        won = total > duel.agent_score
        if won:
            st.balloons()
        st.header("🏆 You out-read the archivist!" if won else "📜 The archives prevail... this time.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Your score", total, help=f"includes {ss.stump_points} stump bonus" if ss.stump_points else None)
        c2.metric("Lorekeeper", duel.agent_score)
        c3.metric("Best streak", duel.best_streak)
        st.markdown(f"### Your rank: **{rank_for(total)}**")
        if nr := next_rank(total):
            st.caption(f"{nr[1]} more points to become a {nr[0]}.")
        st.dataframe(
            pd.DataFrame(duel.history)[["round", "question", "player_answer", "player_points", "agent_answer", "agent_points"]],
            hide_index=True, width="stretch",
        )
        with st.form("save_score"):
            name = st.text_input("Inscribe your name in the Hall of Scribes")
            if st.form_submit_button("Save score"):
                duel.player_score = total
                save_highscore(name, duel)
                ss.stump_points = 0
                st.success("Inscribed. See the Leaderboard tab.")
        if st.button("Play again"):
            ss.phase = "intro"
            st.rerun()

# ---------------------------------------------------------------- Stump
with stump_tab:
    st.markdown(
        f"Write your own question about Veyra. If the Lorekeeper gets it **wrong or makes something up**, "
        f"you earn **+{STUMP_BONUS} bonus points** (added to your next duel score), and your question is saved to "
        f"`eval/user_submitted.jsonl` so it joins the evaluation set. Peek at `data/lore/` for inspiration, "
        f"or ask about something the lore never mentions and see if they bluff."
    )
    st.caption(f"Stump bonus banked: {ss.stump_points}")
    with st.form("stump"):
        sq = st.text_input("Your question")
        go = st.form_submit_button("Challenge the Lorekeeper", type="primary")
    if go and sq.strip():
        with st.spinner("The Lorekeeper squints at your question..."):
            ss.stump = ask(get_agent(), sq.strip())
    if res := ss.get("stump"):
        st.markdown(f"**📜 {res['answer']}**")
        render_result(res)
        st.markdown("**Did they get it right?**")
        ref = st.text_input("The correct answer, if you know it (leave blank if the lore doesn't say)", key="stump_ref")
        c1, c2 = st.columns(2)
        if c1.button("✅ Correct, curses"):
            st.info("📜 *\"As expected. The archives are thorough.\"*")
            ss.stump = None
        if c2.button("❌ Wrong or made up!"):
            item = save_stumper(res["question"], ref, res["answer"])
            ss.stump_points += STUMP_BONUS
            st.success(f"Stumped! +{STUMP_BONUS}. Saved as `{item['id']}`. Run `python -m lorekeeper.eval --include-user` to test it.")
            st.info("📜 *\"I shall be having words with the Scribes about this.\"*")
            ss.stump = None

# ---------------------------------------------------------------- Leaderboard
with board_tab:
    scores = load_highscores()
    if not scores:
        st.caption("No scribes have inscribed their names yet. Be the first.")
    else:
        df = pd.DataFrame(scores)
        df["beat_the_lorekeeper"] = df["beat_the_lorekeeper"].map({True: "🏆", False: ""})
        st.dataframe(
            df[["name", "score", "rank", "agent_score", "beat_the_lorekeeper", "best_streak", "date"]],
            hide_index=True, width="stretch",
        )
