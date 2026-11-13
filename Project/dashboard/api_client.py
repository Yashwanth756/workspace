"""Every request the dashboard sends passes through this module.

No page holds an address, a timeout or a session token. The one exception is
run_dashboard.py, which checks that something is listening at config.API_BASE_URL
before any page is drawn.
"""
import threading
import time

import requests
import streamlit as st

from config import config

SESSION_KEY = "session_token"
USER_KEY = "signed_in_user"
NOTICE_KEY = "dashboard_notice"


def end_session(notice=None):
    """Forget the signed-in person; the next run shows the sign-in form."""
    st.session_state.pop(SESSION_KEY, None)
    st.session_state.pop(USER_KEY, None)
    if notice:
        st.session_state[NOTICE_KEY] = notice


def _send(method, url, headers, query, payload, outcome):
    try:
        outcome["reply"] = requests.request(
            method, url, headers=headers, params=query, json=payload,
            timeout=config.API_REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as failure:
        outcome["failure"] = failure


def _wait_for(worker, waiting_message):
    """Show the waiting message, with the seconds elapsed, until the call returns."""
    started = time.monotonic()
    with st.spinner(waiting_message):
        elapsed = st.empty()
        while worker.is_alive():
            elapsed.caption(f"{int(time.monotonic() - started)} seconds so far")
            worker.join(timeout=0.5)
        elapsed.empty()


def _detail_of(reply):
    """The service's own detail, or the status and the text where the reply is not JSON."""
    try:
        detail = reply.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail) if detail else f"{reply.status_code} {reply.text}"


def call_service(method, path, query=None, payload=None, waiting_message=None):
    """Make one call to the service and return its reply as read from JSON."""
    token = st.session_state.get(SESSION_KEY)
    headers = {config.SESSION_TOKEN_HEADER: token} if token else {}
    outcome = {}
    arguments = (method, config.API_BASE_URL + path, headers, query, payload, outcome)
    if waiting_message:
        worker = threading.Thread(target=_send, args=arguments, daemon=True)
        worker.start()
        _wait_for(worker, waiting_message)
    else:
        _send(*arguments)

    if "failure" in outcome:
        st.error(f"The service at {config.API_BASE_URL} did not answer {method} {path}: "
                 f"{type(outcome['failure']).__name__}")
        st.stop()
    reply = outcome["reply"]
    if reply.status_code == 401 and token:
        end_session(config.SESSION_ENDED_MESSAGE)
        st.rerun()
    if reply.status_code >= 400:
        st.error(_detail_of(reply))
        st.stop()
    return reply.json()