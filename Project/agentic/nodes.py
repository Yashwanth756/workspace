"""The ten workflow steps and the router out of the gate.

Each node takes the state and returns only the keys it wrote. A node that calls
an agent records the call against the run, in the same transaction as its other
writes, as soon as the agent function returns; a reply the agent function
refuses stops the run before its call is recorded.
"""
import secrets

import numpy as np
import pandas as pd
from langgraph.types import interrupt

from agentic.agents import (advise_on_allocation, analyse_demand, draft_allocation_notice,
                            draft_sales_briefing, plan_allocation_review,
                            review_customer_requests, write_allocation_recommendation)
from config import config
from data_pipeline.dataset_loader import load_raw_dataset
from knowledge.policy_reading import read_allocation_policy
from knowledge.vector_index import open_vector_index
from modeling.model_registry import load_feature_schema, load_model
from persistence.database import session_factory
from persistence.repository import (insert_agent_run, insert_allocation_decision,
                                    insert_allocation_review, insert_llm_call,
                                    record_allocation_outcome, update_agent_run)
from tools.allocation_tools import (allocate_cases, compute_allocation_outcome,
                                    compute_usual_demand, summarise_allocation)
from tools.prediction_tools import (build_inflated_order_facts, build_request_facts,
                                    compute_forecast_shortfall, forecast_customer_demand)
from tools.retrieval_tools import (read_customer_advice, read_planner_checks,
                                   read_sales_briefing_guidance)
from tools.shortage_data_tools import (fetch_customer_accounts, fetch_open_emails,
                                       fetch_open_orders, fetch_shipment_history,
                                       fetch_shortage_profile)

# ---------------------------------------------------------------- module-level resources

inflated_order_model = load_model(config.INFLATED_ORDER_MODEL_FILE)
demand_model = load_model(config.DEMAND_MODEL_FILE)
request_model = load_model(config.REQUEST_MODEL_FILE)
inflated_order_feature_names = load_feature_schema(config.INFLATED_ORDER_SCHEMA_FILE)
demand_feature_names = load_feature_schema(config.DEMAND_SCHEMA_FILE)
shortage_products = load_raw_dataset(config.SHORTAGE_PRODUCTS_FILE)
open_orders = load_raw_dataset(config.OPEN_ORDERS_FILE)
shipment_history = load_raw_dataset(config.SHIPMENT_HISTORY_FILE)
customer_master = load_raw_dataset(config.CUSTOMER_MASTER_FILE)
open_customer_emails = load_raw_dataset(config.OPEN_CUSTOMER_EMAILS_FILE)
knowledge_collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)


# ---------------------------------------------------------------- helpers

def _new_identifier(prefix):
    return prefix + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH)[
        :config.IDENTIFIER_HEX_LENGTH].upper()


def _plain(value):
    """A value the checkpoint and the database can store: no NumPy, pandas or NaN."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, str) else str(value)


def _order_records(product_orders):
    return _plain(product_orders.to_dict(orient="records"))


def _orders_frame(order_records):
    frame = pd.DataFrame(order_records)
    frame[config.ORDERED_QUANTITY_COLUMN] = frame[config.ORDERED_QUANTITY_COLUMN].astype(int)
    return frame


def _labelled_text(fields):
    """Each field as its label, a colon, a space and its text, joined by blank lines."""
    return "\n\n".join(f"{label}: {text}" for label, text in fields.items())


def _record_call(run_id, call_record):
    with session_factory() as session:
        insert_llm_call(session, run_id, call_record)
        session.commit()


# ---------------------------------------------------------------- the nodes

def load_shortage_context_node(state):
    """Gather the product's facts, run the three models and open the run."""
    product_code = state["product_code"]
    shortage_profile = fetch_shortage_profile(shortage_products, product_code)
    product_orders = fetch_open_orders(open_orders, product_code)
    account_codes = sorted(set(product_orders[config.ACCOUNT_CODE_COLUMN]))
    customer_accounts = fetch_customer_accounts(customer_master, account_codes)
    open_emails = fetch_open_emails(open_customer_emails, product_code, account_codes)
    demand_facts = {
        "inflated_order_facts": build_inflated_order_facts(
            inflated_order_model, inflated_order_feature_names, product_orders),
        "customer_forecasts": forecast_customer_demand(demand_model, demand_feature_names,
                                                       product_orders),
    }
    request_facts = build_request_facts(request_model, open_emails)

    run_id = _new_identifier(config.RUN_ID_PREFIX)
    review_id = _new_identifier(config.REVIEW_ID_PREFIX)
    with session_factory() as session:
        insert_agent_run(session, {"run_id": run_id, "thread_id": state["thread_id"],
                                   "product_code": product_code,
                                   "planner_id": state["planner_id"],
                                   "planner_request": state["planner_request"],
                                   "status": config.AGENT_RUN_STATUS_RUNNING})
        session.commit()
    return {"run_id": run_id, "review_id": review_id,
            "shortage_profile": _plain(shortage_profile),
            "open_orders": _order_records(product_orders),
            "customer_accounts": _plain(customer_accounts),
            "demand_facts": _plain(demand_facts), "request_facts": _plain(request_facts)}


def plan_allocation_review_node(state):
    """Ask the planning agent what has to be established."""
    reply = plan_allocation_review(state["shortage_profile"], state["open_orders"],
                                   state["customer_accounts"], state["planner_request"])
    _record_call(state["run_id"], reply["call_record"])
    return {"review_plan": reply["review_plan"], "llm_call_count": 1}


def analyse_demand_node(state):
    """Ask the demand agent to explain the scores and the forecasts."""
    reply = analyse_demand(state["demand_facts"], state["open_orders"], state["customer_accounts"],
                           state["shortage_profile"], state["review_plan"])
    _record_call(state["run_id"], reply["call_record"])
    return {"demand_analysis": reply["demand_analysis"], "llm_call_count": 1}


def review_customer_requests_node(state):
    """Ask the request agent what customers are asking for."""
    request_types = {email[config.EMAIL_TARGET_COLUMN]
                     for email in state["request_facts"]["email_requests"]}
    planner_checks = {request_type: read_planner_checks(knowledge_collection, request_type)
                      for request_type in config.REQUEST_URGENCY_ORDER
                      if request_type in request_types}
    reply = review_customer_requests(state["request_facts"], planner_checks,
                                     state["shortage_profile"], state["customer_accounts"],
                                     state["review_plan"])
    _record_call(state["run_id"], reply["call_record"])
    return {"request_review": reply["request_review"], "llm_call_count": 1}


def build_allocation_facts(shortage_profile, product_code, product_orders, customer_accounts):
    """The policy read, the division and its summary, worked out with the exact cap.

    Shared with the service, so the queue and the review use the same figures.
    """
    policy_rules = read_allocation_policy(knowledge_collection,
                                          shortage_profile[config.PRODUCT_CATEGORY_COLUMN])
    shipments = fetch_shipment_history(shipment_history, product_code)
    usual_demand = compute_usual_demand(shipments, list(customer_accounts),
                                        policy_rules["history_weeks"])
    allocation = allocate_cases(shortage_profile[config.CASES_AVAILABLE_THIS_WEEK_COLUMN],
                                product_orders, customer_accounts, usual_demand, policy_rules)
    summary = summarise_allocation(allocation)
    stored_rules = dict(policy_rules, cap=float(policy_rules["cap"]))
    return _plain({"policy_rules": stored_rules, "allocation": allocation, "summary": summary})


def compute_allocation_node(state):
    """Divide the cases under the category's own policy, then have the advisor explain it."""
    allocation_facts = build_allocation_facts(state["shortage_profile"], state["product_code"],
                                              _orders_frame(state["open_orders"]),
                                              state["customer_accounts"])
    reply = advise_on_allocation(allocation_facts, state["shortage_profile"], state["review_plan"])
    _record_call(state["run_id"], reply["call_record"])
    return {"allocation_facts": allocation_facts, "allocation_advice": reply["allocation_advice"],
            "llm_call_count": 1}


def compose_allocation_recommendation_node(state):
    """Work out the shortfall, have the recommendation written, and record the review."""
    profile = state["shortage_profile"]
    facts = state["allocation_facts"]
    summary = facts["summary"]
    forecast_shortfall = compute_forecast_shortfall(
        state["demand_facts"]["customer_forecasts"],
        profile[config.CASES_AVAILABLE_NEXT_WEEK_COLUMN])
    reply = write_allocation_recommendation(profile, state["planner_request"], facts,
                                            forecast_shortfall, state["review_plan"],
                                            state["demand_analysis"], state["request_review"],
                                            state["allocation_advice"])
    recommendation = reply["allocation_recommendation"]
    with session_factory() as session:
        insert_llm_call(session, state["run_id"], reply["call_record"])
        insert_allocation_review(session, {
            "review_id": state["review_id"], "run_id": state["run_id"],
            "product_code": state["product_code"],
            "product_name": profile[config.PRODUCT_NAME_COLUMN],
            "product_category": profile[config.PRODUCT_CATEGORY_COLUMN],
            "policy_document": facts["policy_rules"]["document_name"],
            "planner_request": state["planner_request"],
            "cases_available": summary["cases_available"],
            "cases_ordered": summary["cases_ordered"],
            "cases_eligible": summary["cases_eligible"],
            "cover_percent": summary["cover_percent"],
            "protected_served_in_full": summary["protected_served_in_full"],
            "allocation_lines": facts["allocation"]["allocation_lines"],
            "releases": summary["releases"],
            "decisions_open": summary["decisions_open"],
            "decisions_not_open": summary["decisions_not_open"],
            "inflated_order_scores": state["demand_facts"]["inflated_order_facts"],
            "email_requests": state["request_facts"]["email_requests"],
            "customer_requests": state["request_facts"]["customer_requests"],
            "forecast_shortfall": int(forecast_shortfall),
            "review_plan_text": _labelled_text(state["review_plan"]),
            "recommendation_text": recommendation["RECOMMENDATION"],
            "recommended_decision": recommendation["RECOMMENDED DECISION"],
            "status": config.REVIEW_STATUS_PENDING,
        })
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_AWAITING_DECISION,
                         int(state["llm_call_count"]) + 1)
        session.commit()
    return {"llm_call_count": 1}


def request_allocation_decision_node(state):
    """Pause until a Supply Planning Manager decides, then work out what it confirms."""
    allocation_decision = interrupt({"review_id": state["review_id"]})
    facts = state["allocation_facts"]
    outcome = compute_allocation_outcome(allocation_decision["decision"], facts["allocation"],
                                         facts["summary"],
                                         state["shortage_profile"][config.PRODUCT_NAME_COLUMN])
    return {"allocation_decision": _plain(allocation_decision),
            "allocation_outcome": _plain(outcome), "allocation_notice": {}}


def route_allocation_decision(state):
    """The decision itself, which names the path the run takes."""
    return state["allocation_decision"]["decision"]


def draft_allocation_notice_node(state):
    """Have the allocation notice written."""
    profile = state["shortage_profile"]
    policy_rules = state["allocation_facts"]["policy_rules"]
    customer_advice = read_customer_advice(knowledge_collection,
                                           profile[config.PRODUCT_CATEGORY_COLUMN])
    reply = draft_allocation_notice(profile, policy_rules, state["allocation_outcome"],
                                    customer_advice, state["customer_accounts"])
    _record_call(state["run_id"], reply["call_record"])
    return {"allocation_notice": reply["allocation_notice"], "llm_call_count": 1}


def draft_sales_briefing_node(state):
    """Have the sales briefing written."""
    profile = state["shortage_profile"]
    guidance = read_sales_briefing_guidance(knowledge_collection,
                                            profile[config.PRODUCT_CATEGORY_COLUMN])
    reply = draft_sales_briefing(profile, state["allocation_facts"]["policy_rules"]["document_name"],
                                 state["allocation_outcome"], state["allocation_notice"],
                                 guidance, state["customer_accounts"])
    _record_call(state["run_id"], reply["call_record"])
    return {"sales_briefing": reply["sales_briefing"], "llm_call_count": 1}


def record_allocation_outcome_node(state):
    """Close the review with its decision and what it confirms; the run is over."""
    decision = state["allocation_decision"]
    outcome = state["allocation_outcome"]
    notice = state["allocation_notice"]
    with session_factory() as session:
        insert_allocation_decision(session, {
            "review_id": state["review_id"], "run_id": state["run_id"],
            "decision": decision["decision"], "decided_by": decision["decided_by"],
            "decided_by_role": decision["decided_by_role"],
            "decision_reason": decision["decision_reason"]})
        record_allocation_outcome(session, state["review_id"], {
            "status": config.REVIEW_STATUS_BY_DECISION[decision["decision"]],
            "cases_confirmed": outcome["cases_confirmed"],
            "reserve_cases": outcome["reserve_cases"],
            "cases_in_stock": outcome["cases_in_stock"],
            "notice_text": _labelled_text(notice) if notice else None,
            "notice_lines": outcome["notice_lines"],
            "briefing_text": _labelled_text(state["sales_briefing"]),
        })
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_COMPLETE,
                         int(state["llm_call_count"]))
        session.commit()
    return {}