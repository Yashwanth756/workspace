"""The audit trail page: every allocation review and every chat question."""
import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service


def _released(review):
    confirmed = review["cases_confirmed"]
    return None if confirmed is None else int(sum(confirmed.values()))


def _review_detail(review):
    """What one review recorded: the decision, the texts and the language model calls."""
    st.markdown(f"**{review['review_id']}** · {review['product_code']} {review['product_name']} · "
                f"policy {review['policy_document']} · opened by {review['planner_id']}")
    st.markdown("**Recommendation** (" + review["recommended_decision"] + ")")
    st.write(review["recommendation_text"])
    if review["decision"]:
        st.markdown(f"**Decision**: {review['decision']} by {review['decided_by']}, "
                    f"{review['decided_by_role']}")
        st.write(review["decision_reason"])
    for heading, text in [("Notice", review["notice_text"]), ("Briefing", review["briefing_text"])]:
        if text:
            with st.expander(heading):
                st.write(text)
    budget = call_service("GET", f"/reviews/{review['review_id']}/calls")
    st.markdown(f"**Model calls**: {budget['call_count']} made, {budget['expected_call_count']} "
                f"expected at this stage, {budget['total_latency_milliseconds']:,} ms in all")
    st.dataframe(pd.DataFrame(budget["calls"]), hide_index=True, width="stretch")


def _reviews_tab():
    labels = {status: config.REVIEW_STATUS_LABELS[status] for status in config.REVIEW_STATUS_OPTIONS}
    chosen = st.pills("Status", config.REVIEW_STATUS_OPTIONS, selection_mode="multi",
                      format_func=labels.get, key="audit_status")
    reviews = call_service("GET", "/reviews", query={
        "status_value": list(chosen) if chosen else config.REVIEW_STATUS_OPTIONS})
    if not reviews:
        st.info("No allocation review matches.")
        return
    table = pd.DataFrame({
        "Product": [row["product_code"] for row in reviews],
        "Category": [row["product_category"] for row in reviews],
        "Status": [labels[row["status"]] for row in reviews],
        "Decision": [f"{row['decision']} by {row['decided_by']}" if row["decision"] else ""
                     for row in reviews],
        "Cases released": [_released(row) for row in reviews],
        "Model calls": [row["llm_call_count"] for row in reviews]})
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key=f"audit_table_{sorted(chosen or [])}",
                         column_config={"Cases released": st.column_config.NumberColumn(
                             format="localized")})
    if event.selection.rows:
        _review_detail(reviews[event.selection.rows[0]])


def _chat_log_tab():
    turns = call_service("GET", "/audit/chat-turns", query={"turn_limit": config.CHAT_AUDIT_MAX_ROWS})
    if not turns:
        st.info("No question has been asked yet.")
        return
    st.dataframe(pd.DataFrame({
        "Asked by": [turn["username"] for turn in turns],
        "Role": [turn["user_role"] for turn in turns],
        "Question": [turn["question_text"] for turn in turns],
        "Type": [turn["question_type"] for turn in turns],
        "Account": [turn["account_code"] for turn in turns],
        "Product": [turn["product_code"] for turn in turns],
        "Refused": ["Yes" if turn["was_refused"] else "No" for turn in turns],
        "Sources": [turn["source_documents"] for turn in turns]}),
        hide_index=True, width="stretch")


def render_audit_trail_page():
    """Show the allocation reviews, and the chat turns of everyone, as they were recorded."""
    st.title(config.PAGE_AUDIT)
    reviews_tab, chat_tab = st.tabs(["Allocation reviews", "Chat log"])
    with reviews_tab:
        _reviews_tab()
    with chat_tab:
        _chat_log_tab()