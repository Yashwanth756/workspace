"""The decision queue page: the reviews waiting for a Supply Planning Manager."""
import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import USER_KEY, call_service

DECISION_NOTICE_KEY = "decision_notice"
DECISION_WAIT_MESSAGE = ("Recording the decision and drafting what it needs: up to two more "
                         "language model calls, which can take a minute.")


def _badge(label, colour):
    return f":{colour}-badge[{label}]"


def _outcome_text(recorded):
    """One sentence on what the recorded decision confirmed, held and left."""
    label = config.REVIEW_STATUS_LABELS[recorded["status"]].lower()
    released = sum((recorded["cases_confirmed"] or {}).values())
    if recorded["decision"] == config.DECISION_ESCALATE:
        return (f"{recorded['product_code']} {label}: no case is confirmed and "
                f"{recorded['cases_in_stock']:,} stay in stock.")
    return (f"{recorded['product_code']} {label}: {released:,} cases released, "
            f"{recorded['reserve_cases']:,} held in reserve and "
            f"{recorded['cases_in_stock']:,} left in stock.")


def _customers_table(review):
    lines = review["allocation_lines"]
    release_names = list(review["releases"])
    table = {"Customer": [line["customer_name"] for line in lines],
             "Channel": [line["channel"] for line in lines],
             "Protected": ["Yes" if line["protected"] else "No" for line in lines],
             "Ordered": [line["cases_ordered"] for line in lines],
             "May have": [line["eligible_cases"] for line in lines]}
    for release_name in release_names:
        table[release_name] = [line["cases_by_release"][release_name] for line in lines]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


def _decision_form(review):
    """The form for one review, or the reason the signed-in person cannot decide it."""
    user = st.session_state[USER_KEY]
    if review["planner_id"] == user["username"]:
        st.info("You opened this review, so another Supply Planning Manager must decide it.",
                icon=":material/group:")
        return
    with st.container(border=True):
        st.markdown("**Your decision**")
        with st.form(f"decision_{review['review_id']}"):
            decision = st.radio("Decision", review["decisions_open"], index=None,
                                label_visibility="collapsed")
            reason = st.text_area("Reason", placeholder="For example: every protected customer "
                                  "is served in full, so the cases can go out.")
            recorded = st.form_submit_button("Record the decision", type="primary",
                                             width="stretch", icon=":material/gavel:")
        for decision_name, reasons in review["decisions_not_open"].items():
            st.caption(f"{decision_name} is not open: {reasons}.")
    if not recorded:
        return
    if decision is None:
        st.warning("Choose a decision.")
        return
    if len(reason.strip()) < config.DECISION_REASON_MINIMUM_CHARACTERS:
        st.warning(f"Give a reason of at least {config.DECISION_REASON_MINIMUM_CHARACTERS} "
                   f"characters.")
        return
    answer = call_service("POST", f"/reviews/{review['review_id']}/decision",
                          payload={"decision": decision, "decision_reason": reason.strip()},
                          waiting_message=DECISION_WAIT_MESSAGE)
    st.session_state[DECISION_NOTICE_KEY] = _outcome_text(answer)
    st.rerun()


def render_decision_queue_page():
    """Show a review waiting for a decision and let another manager record the decision."""
    st.title(config.PAGE_DECISION)
    notice = st.session_state.pop(DECISION_NOTICE_KEY, None)
    if notice:
        st.success(notice, icon=":material/check_circle:")
    waiting = call_service("GET", "/reviews", query={"status_value": [config.REVIEW_STATUS_PENDING]})
    if not waiting:
        st.info("No allocation review is waiting for a decision.")
        return
    st.caption(f"{len(waiting)} waiting for a decision")
    labels = {row["review_id"]: f"{row['product_code']} | {row['product_name']}, "
                                f"{row['product_category']}" for row in waiting}
    review_id = st.selectbox("Review", list(labels), format_func=labels.get,
                             label_visibility="collapsed")
    review = next(row for row in waiting if row["review_id"] == review_id)

    left, right = st.columns([3, 1])
    with left:
        recommended = review["recommended_decision"]
        st.markdown(f"{_badge('Recommended: ' + recommended, config.DECISION_BADGE_COLOURS[recommended])} "
                    f"{_badge(review['policy_document'], 'gray')}")
        metrics = st.columns(4)
        metrics[0].metric("Cases available", f"{review['cases_available']:,}", border=True)
        metrics[1].metric("Customers may have", f"{review['cases_eligible']:,}", border=True)
        metrics[2].metric("Cover", ("Not worked out" if review["cover_percent"] is None
                                    else f"{review['cover_percent']:.2f}%"), border=True)
        metrics[3].metric("Shortfall next week", f"{review['forecast_shortfall']:,}", border=True)
        st.write(review["recommendation_text"])
        _customers_table(review)
        with st.expander(f"Request and plan, opened by {review['planner_id']}"):
            st.markdown("**Request**")
            st.write(review["planner_request"])
            st.markdown("**Plan**")
            st.write(review["review_plan_text"])
    with right:
        _decision_form(review)