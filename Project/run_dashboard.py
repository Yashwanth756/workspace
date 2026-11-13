"""Draws the FairShare AI dashboard.

Run from the Project folder:  python3 -m streamlit run run_dashboard.py
The service must already be running:  python3 run_api_service.py
"""
import requests
import streamlit as st

from config import config
from dashboard.allocation_chat import render_allocation_chat_page
from dashboard.allocation_detail import render_allocation_detail_page
from dashboard.api_client import NOTICE_KEY, SESSION_KEY, USER_KEY, call_service, end_session
from dashboard.audit_trail import render_audit_trail_page
from dashboard.decision_queue import render_decision_queue_page
from dashboard.shortage_queue import render_shortage_queue_page

PAGE_RENDERERS = {
    config.PAGE_QUEUE: (render_shortage_queue_page, ":material/format_list_numbered:"),
    config.PAGE_DETAIL: (render_allocation_detail_page, ":material/pie_chart:"),
    config.PAGE_DECISION: (render_decision_queue_page, ":material/gavel:"),
    config.PAGE_AUDIT: (render_audit_trail_page, ":material/fact_check:"),
    config.PAGE_CHAT: (render_allocation_chat_page, ":material/forum:"),
}


def _service_is_listening():
    try:
        requests.get(
            f"{config.API_BASE_URL.rstrip('/')}/docs",
            timeout=config.API_REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException:
        return False
    return True


def _show_sign_in_form():
    st.title(config.DASHBOARD_TITLE)
    st.caption(config.DASHBOARD_SUBTITLE)
    notice = st.session_state.pop(NOTICE_KEY, None)
    if notice:
        st.warning(notice)
    with st.form("sign_in"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if not submitted:
        return
    signed_in = call_service("POST", "/auth/signin",
                             payload={"username": username.strip(), "password": password})
    st.session_state[SESSION_KEY] = signed_in["session_token"]
    st.session_state[USER_KEY] = signed_in
    st.rerun()


def _show_signed_in_pages():
    user = st.session_state[USER_KEY]
    pages = [st.Page(PAGE_RENDERERS[name][0], title=name, icon=PAGE_RENDERERS[name][1],
                     url_path=name.lower().replace(" ", "-"))
             for name in user["pages"]]
    navigation = st.navigation(pages)
    with st.sidebar:
        st.markdown(f"{':material/balance:'} **{config.DASHBOARD_TITLE}**")
        with st.container(border=True):
            st.markdown(f"**{user['display_name']}**")
            st.caption(user["role"])
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            call_service("POST", "/auth/signout")
            end_session()
            st.rerun()
    navigation.run()


def main():
    """Draw the dashboard with only the pages the signed-in role is offered."""
    st.set_page_config(page_title=config.DASHBOARD_TITLE, layout="wide")
    if not _service_is_listening():
        st.error(f"The service is not answering at {config.API_BASE_URL}. Start it from the "
                 f"Project folder with: python3 run_api_service.py")
        st.stop()
    if USER_KEY not in st.session_state:
        _show_sign_in_form()
    else:
        _show_signed_in_pages()


if __name__ == "__main__":
    main()