import streamlit as st

from common import get_agent, render_result, sidebar_badge
from lorekeeper.graph import ask

st.set_page_config(page_title="The Lorekeeper", page_icon="📜", layout="centered")
sidebar_badge()

st.title("📜 Ask the Lorekeeper")
st.caption(
    "The archivist of the Great Library of Halcyon Reach answers questions about the island realm of Veyra, "
    "citing the scrolls they consulted. Try the **Quiz Duel** page to challenge them."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Fair tides, visitor. Ask me anything about Veyra: its islands, guilds, sky-whales, or its unfortunate wars."}
    ]

with st.sidebar:
    st.caption("Try asking")
    for q in [
        "Who was Regent during the Stormwar?",
        "What is driftglass used for?",
        "Why did the Hollow Choir ban Stormweaving?",
        "What is the population of Duskholm?",
        "Hello! Who are you?",
    ]:
        if st.button(q, width="stretch"):
            st.session_state.pending = q
    if st.button("Clear conversation", type="secondary"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="📜" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("result"):
            render_result(msg["result"])

question = st.chat_input("Ask about Veyra...") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📜"):
        try:
            with st.spinner("Consulting the archives..."):
                result = ask(get_agent(), question)
        except Exception as e:  # most likely: Ollama isn't running
            st.error(f"The archives are locked: {e}\n\nIs Ollama running (`ollama serve`)? Or check your `.env`.")
            st.stop()
        st.markdown(result["answer"])
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
