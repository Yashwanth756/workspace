"""The allocation chat page."""
import streamlit as st

from config import config
from dashboard.api_client import call_service

CHAT_WAIT_MESSAGE = "Reading the documents and answering."
QUESTION_HINT = ("Ask about a policy, the allocation rules or a request guide, or about an "
                 "allocation by its account code and delivery postcode")


def _show_turn(turn):
    with st.chat_message("user"):
        st.write(turn["question_text"])
    with st.chat_message("assistant", avatar=":material/balance:"):
        colour = config.CHAT_TYPE_BADGE_COLOURS.get(turn["question_type"], "gray")
        st.markdown(f":{colour}-badge[{turn['question_type']}]")
        st.write(turn["answer_text"])
        documents = [name for name in turn["source_documents"].split(", ") if name]
        if documents:
            with st.expander("Sources"):
                for name in documents:
                    st.markdown(f"- {name}")


def render_allocation_chat_page():
    """Ask the allocation assistant a question and show the conversation."""
    st.title(config.PAGE_CHAT)
    turns = call_service("GET", "/chat/turns", query={"turn_limit": config.CHAT_PAGE_TURN_LIMIT})
    for turn in reversed(turns):
        _show_turn(turn)

    question = st.chat_input(QUESTION_HINT)
    if question is None:
        return
    if len(question.strip()) < config.CHAT_QUESTION_MINIMUM_CHARACTERS:
        st.warning(f"Write a question of at least {config.CHAT_QUESTION_MINIMUM_CHARACTERS} "
                   f"characters.")
        return
    call_service("POST", "/chat/ask", payload={"question": question.strip()},
                 waiting_message=CHAT_WAIT_MESSAGE)
    st.rerun()