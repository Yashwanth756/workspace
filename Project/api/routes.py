"""The eleven REST endpoints for FairShare AI."""
import hmac
import re
import secrets
import threading
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors
from langgraph.types import Command
from sqlalchemy import select, true

from access.password_security import hash_password
from access.permissions import authorize_request
from agentic import nodes
from agentic.graph import build_workflow
from api.schemas import (CallBudgetResponse, CategoryFilter, ChatRequest, ChatResponse,
                         ChatTurnResponse, DecisionRequest, QueueRow, ReviewRecordResponse,
                         ReviewRequest, ReviewStatusValue, ShortageDetailResponse,
                         SignInRequest, SignedInUserResponse, SignedOutResponse)
from chat.allocation_assistant import answer_from_documents
from chat.allocation_status_answer import answer_allocation_status
from chat.conversation_memory import recall_conversation
from config import config
from persistence.database import session_factory
from persistence.models import AgentRun, AllocationReview, ChatTurn, UserAccount
from persistence.repository import (fetch_chat_turns, fetch_llm_calls, fetch_review_records,
                                    fetch_user_account, insert_chat_turn, update_agent_run,
                                    update_session_token)
from tools.allocation_tools import rank_shortage_queue
from tools.prediction_tools import (build_request_facts, compute_forecast_shortfall,
                                    forecast_customer_demand)
from tools.shortage_data_tools import (fetch_customer_accounts, fetch_open_emails,
                                       fetch_open_orders, fetch_shortage_profile)

review_workflow = build_workflow(config.CHECKPOINT_DATABASE_FILE)
policy_rules_by_category = {}
shortage_details = {}
queue_entries = []

# Work every live product out once, with the same resources and tools as the workflow.
for product_code in nodes.shortage_products[config.PRODUCT_CODE_COLUMN].tolist():
    profile = fetch_shortage_profile(nodes.shortage_products, product_code)
    product_orders = fetch_open_orders(nodes.open_orders, product_code)
    account_codes = sorted(set(product_orders[config.ACCOUNT_CODE_COLUMN]))
    customer_accounts = fetch_customer_accounts(nodes.customer_master, account_codes)
    allocation_facts = nodes.build_allocation_facts(profile, product_code, product_orders,
                                                    customer_accounts)
    request_facts = build_request_facts(
        nodes.request_model,
        fetch_open_emails(nodes.open_customer_emails, product_code, account_codes))
    customer_forecasts = forecast_customer_demand(nodes.demand_model,
                                                  nodes.demand_feature_names, product_orders)
    forecast_shortfall = compute_forecast_shortfall(
        customer_forecasts, profile[config.CASES_AVAILABLE_NEXT_WEEK_COLUMN])
    policy_rules_by_category[profile[config.PRODUCT_CATEGORY_COLUMN]] = (
        allocation_facts["policy_rules"])
    order_scores = {row["order_line_id"]: row for row in nodes.build_inflated_order_facts(
        nodes.inflated_order_model, nodes.inflated_order_feature_names, product_orders)}
    order_lines = []
    for row in product_orders.to_dict(orient="records"):
        fact = order_scores[row[config.ORDER_LINE_ID_COLUMN]]
        order_lines.append({
            "order_line_id": str(row[config.ORDER_LINE_ID_COLUMN]),
            "order_reference": str(row[config.ORDER_REFERENCE_COLUMN]),
            "order_route": str(row[config.ORDER_ROUTE_COLUMN]),
            "account_code": str(row[config.ACCOUNT_CODE_COLUMN]),
            "customer_name": str(row[config.CUSTOMER_NAME_COLUMN]),
            "ordered_quantity": int(row[config.ORDERED_QUANTITY_COLUMN]),
            "inflated_order_score": fact["inflated_order_score"],
            "inflated_order_band": fact["inflated_order_band"]})
    shortage_details[product_code] = {
        "profile": nodes._plain(profile),
        "policy_rules": allocation_facts["policy_rules"],
        "allocation_lines": allocation_facts["allocation"]["allocation_lines"],
        "summary": allocation_facts["summary"], "forecast_shortfall": forecast_shortfall,
        "order_lines": order_lines, "email_requests": request_facts["email_requests"],
        "customer_requests": request_facts["customer_requests"]}
    summary = allocation_facts["summary"]
    queue_entries.append({
        "product_code": product_code,
        "product_name": profile[config.PRODUCT_NAME_COLUMN],
        "product_category": profile[config.PRODUCT_CATEGORY_COLUMN],
        "cases_available_this_week": profile[config.CASES_AVAILABLE_THIS_WEEK_COLUMN],
        "cases_available_next_week": profile[config.CASES_AVAILABLE_NEXT_WEEK_COLUMN],
        "cases_ordered": summary["cases_ordered"], "cases_eligible": summary["cases_eligible"],
        "cover_percent": summary["cover_percent"],
        "protected_served_in_full": summary["protected_served_in_full"],
        "decisions_open": summary["decisions_open"], "forecast_shortfall": forecast_shortfall})

shortage_queue = [dict(entry, queue_position=position)
                  for position, entry in enumerate(rank_shortage_queue(queue_entries), start=1)]
queue_positions = {row["product_code"]: row["queue_position"] for row in shortage_queue}
review_start_locks = {code: threading.Lock() for code in shortage_details}
decision_lock = threading.Lock()

public_router = APIRouter()
guarded_router = APIRouter(dependencies=[Depends(authorize_request)])
SessionTokenHeader = Annotated[str | None, Header(alias=config.SESSION_TOKEN_HEADER)]
SignedInUser = Annotated[dict, Depends(authorize_request)]
TurnLimit = Annotated[int, Query(ge=1, le=config.CHAT_AUDIT_MAX_ROWS)]


def _http(status_code, detail):
    raise HTTPException(status_code=status_code, detail=detail)


def _review_record(review_id):
    with session_factory() as session:
        rows = fetch_review_records(session, AllocationReview.review_id == review_id)
    if not rows:
        _http(404, config.REVIEW_NOT_FOUND_MESSAGE)
    return rows[0]


def _public_review(record):
    excluded = {"created_at", "status_changed_at"}
    return {key: value for key, value in record.items() if key not in excluded}


def _public_turn(turn):
    return {key: value for key, value in turn.items() if key != "asked_at"}


def _withhold_postcodes(question):
    """The question with every postcode withheld, and the one postcode it gave, or None
    where it gave none or more than one (compared without spaces and whatever the case)."""
    given = re.findall(config.POSTCODE_PATTERN, question)
    distinct = {re.sub(r"\s+", "", postcode).upper() for postcode in given}
    postcode = given[0] if len(distinct) == 1 else None
    return re.sub(config.POSTCODE_PATTERN, config.CHAT_POSTCODE_WITHHELD_TEXT, question), postcode


def _decision_already_sent(review):
    """The decision that already passed the gate for this review, or None.

    The workflow's checkpoint holds it once the gate has taken it: a decision that then
    failed while the notice or the briefing was drafted is still there, waiting to be
    finished.
    """
    snapshot = review_workflow.get_state({"configurable": {"thread_id": review["thread_id"]}})
    return (snapshot.values or {}).get("allocation_decision") or None


@public_router.post("/auth/signin", response_model=SignedInUserResponse)
def sign_in_user(sign_in_request: SignInRequest):
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.username == sign_in_request.username)
        valid = (account is not None and hmac.compare_digest(
            account["password_hash"],
            hash_password(sign_in_request.password, account["password_salt"])))
        if not valid:
            _http(401, config.SIGN_IN_FAILED_MESSAGE)
        token = secrets.token_bytes(config.SESSION_TOKEN_BYTES).hex()
        update_session_token(session, account["username"], token)
        session.commit()
    return {"username": account["username"], "display_name": account["display_name"],
            "role": account["role"], "session_token": token,
            "pages": list(config.ROLE_PAGE_ACCESS[account["role"]]),
            "areas": list(config.ROLE_AREA_ACCESS[account["role"]])}


@public_router.post("/auth/signout", response_model=SignedOutResponse)
def sign_out_user(session_token: SessionTokenHeader = None):
    if not session_token:
        _http(401, config.SIGN_IN_REQUIRED_MESSAGE)
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.session_token == session_token)
        if account is None:
            _http(401, config.SIGN_IN_REQUIRED_MESSAGE)
        update_session_token(session, account["username"], None)
        session.commit()
    return {"username": account["username"], "signed_out": True}


@guarded_router.get("/queue", response_model=list[QueueRow])
def list_shortage_queue(product_category: CategoryFilter):
    if product_category == config.QUEUE_CATEGORY_ALL:
        return shortage_queue
    return [row for row in shortage_queue if row["product_category"] == product_category]


@guarded_router.get("/queue/{product_code}", response_model=ShortageDetailResponse)
def get_shortage_detail(product_code: str):
    if product_code not in shortage_details:
        _http(404, config.PRODUCT_NOT_IN_QUEUE_MESSAGE.format(product_code=product_code))
    detail = shortage_details[product_code]
    return {"queue_position": queue_positions[product_code], "profile": detail["profile"],
            "policy_rules": detail["policy_rules"],
            "allocation_lines": detail["allocation_lines"], "summary": detail["summary"],
            "forecast_shortfall": detail["forecast_shortfall"],
            "order_lines": detail["order_lines"], "email_requests": detail["email_requests"],
            "customer_requests": detail["customer_requests"]}


@guarded_router.post("/reviews", status_code=201, response_model=ReviewRecordResponse)
def start_allocation_review(review_request: ReviewRequest, signed_in_user: SignedInUser,
                            request: Request):
    code = review_request.product_code
    if code not in shortage_details:
        _http(404, config.PRODUCT_NOT_IN_QUEUE_MESSAGE.format(product_code=code))
    with review_start_locks[code]:
        with session_factory() as session:
            existing = fetch_review_records(session, AllocationReview.product_code == code)
        if existing:
            _http(409, config.PRODUCT_ALREADY_REVIEWED_MESSAGE.format(product_code=code))
        thread_id = code + "-" + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH)[
            :config.IDENTIFIER_HEX_LENGTH].upper()
        request.state.thread_id = thread_id
        review_workflow.invoke({"product_code": code,
                                "planner_request": review_request.planner_request,
                                "planner_id": signed_in_user["username"],
                                "thread_id": thread_id, "llm_call_count": 0},
                               {"configurable": {"thread_id": thread_id}})
        with session_factory() as session:
            created = fetch_review_records(session, AllocationReview.product_code == code)
        if not created:
            raise RuntimeError(f"The workflow for {code} reached no stored review")
        return _public_review(created[0])


@guarded_router.get("/reviews", response_model=list[ReviewRecordResponse])
def list_allocation_reviews(status_value: Annotated[list[ReviewStatusValue], Query()]):
    with session_factory() as session:
        rows = fetch_review_records(session, AllocationReview.status.in_(list(status_value)))
    return [_public_review(row) for row in rows]


@guarded_router.post("/reviews/{review_id}/decision", response_model=ReviewRecordResponse)
def submit_allocation_decision(review_id: str, decision_request: DecisionRequest,
                               signed_in_user: SignedInUser, request: Request):
    with decision_lock:
        review = _review_record(review_id)
        if review["planner_id"] == signed_in_user["username"]:
            _http(403, config.OWN_REVIEW_REFUSED_MESSAGE)
        if review["status"] != config.REVIEW_STATUS_PENDING:
            _http(409, config.ALREADY_DECIDED_MESSAGE)
        if decision_request.decision not in review["decisions_open"]:
            reasons = review["decisions_not_open"].get(decision_request.decision, "")
            _http(409, config.DECISION_NOT_OPEN_MESSAGE.format(
                decision=decision_request.decision, reasons=reasons))
        decision = {"decision": decision_request.decision,
                    "decided_by": signed_in_user["display_name"],
                    "decided_by_role": signed_in_user["role"],
                    "decision_reason": decision_request.decision_reason}
        earlier = _decision_already_sent(review)
        if earlier is None:
            resumption = Command(resume=decision)
        else:
            if earlier != decision:
                _http(409, config.DECISION_IN_PROGRESS_MESSAGE.format(
                    decided_by=earlier["decided_by"], decision=earlier["decision"]))
            resumption = None
        request.state.thread_id = review["thread_id"]
        review_workflow.invoke(resumption, {"configurable": {"thread_id": review["thread_id"]}})
        return _public_review(_review_record(review_id))


@guarded_router.get("/reviews/{review_id}/calls", response_model=CallBudgetResponse)
def get_call_budget(review_id: str):
    review = _review_record(review_id)
    with session_factory() as session:
        calls = fetch_llm_calls(session, review["run_id"])
    expected = (config.LLM_CALLS_BEFORE_DECISION if review["decision"] is None
                else config.LLM_CALLS_BY_DECISION[review["decision"]])
    return {"review_id": review_id, "run_id": review["run_id"], "call_count": len(calls),
            "expected_call_count": expected,
            "total_latency_milliseconds": sum(call["latency_milliseconds"] for call in calls),
            "calls": calls}


@guarded_router.post("/chat/ask", response_model=ChatResponse)
def ask_allocation_assistant(chat_request: ChatRequest, signed_in_user: SignedInUser):
    question, postcode = _withhold_postcodes(chat_request.question)
    with session_factory() as session:
        conversation = recall_conversation(session, signed_in_user["username"], question)
        reply = answer_from_documents(nodes.knowledge_collection, question, conversation)
        if reply["question_type"] == config.CHAT_TYPE_ALLOCATION_STATUS:
            reply["answer"] = answer_allocation_status(
                session, signed_in_user, nodes.customer_master, reply["account_code"], postcode,
                reply["product_code"])
        insert_chat_turn(session, {
            "username": signed_in_user["username"], "user_role": signed_in_user["role"],
            "question_text": question, "question_type": reply["question_type"],
            "account_code": reply["account_code"], "product_code": reply["product_code"],
            "answer_text": reply["answer"],
            "source_documents": ", ".join(reply["source_documents"]),
            "closest_distance": reply["closest_distance"], "was_refused": reply["was_refused"]})
        session.commit()
    return {"question_type": reply["question_type"], "account_code": reply["account_code"],
            "product_code": reply["product_code"], "answer": reply["answer"],
            "source_documents": reply["source_documents"], "was_refused": reply["was_refused"]}


@guarded_router.get("/chat/turns", response_model=list[ChatTurnResponse])
def list_own_chat_turns(signed_in_user: SignedInUser, turn_limit: TurnLimit):
    with session_factory() as session:
        return [_public_turn(turn) for turn in fetch_chat_turns(
            session, ChatTurn.username == signed_in_user["username"], turn_limit)]


@guarded_router.get("/audit/chat-turns", response_model=list[ChatTurnResponse])
def list_recent_chat_turns(turn_limit: TurnLimit):
    with session_factory() as session:
        return [_public_turn(turn) for turn in fetch_chat_turns(session, true(), turn_limit)]


def report_failed_request(request: Request, failure: Exception):
    """Turn every failure into one plain reply, and close the run a failed request opened."""
    thread_id = getattr(request.state, "thread_id", None)
    if thread_id:
        with session_factory() as session:
            run = session.execute(select(AgentRun).where(AgentRun.thread_id == thread_id)
                                  ).scalars().first()
            if run is not None:
                update_agent_run(session, run.run_id, config.AGENT_RUN_STATUS_FAILED,
                                 len(fetch_llm_calls(session, run.run_id)))
                session.commit()
    return JSONResponse(status_code=502, content={
        "detail": f"{request.method} {request.url.path} failed: "
                  f"{type(failure).__name__}: {failure}"})


service_application = FastAPI(title=config.API_TITLE)
service_application.include_router(public_router)
service_application.include_router(guarded_router)
for failure_type in (ValueError, genai_errors.APIError, httpx.HTTPError):
    service_application.add_exception_handler(failure_type, report_failed_request)