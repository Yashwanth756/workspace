"""Every read and write against the six tables. Every function takes an open
session, writes and flushes but never commits (the caller commits), and every
row comes back as a plain dictionary, never a database object."""
from sqlalchemy import func, select

from config import config
from persistence.models import (AgentRun, AllocationDecision, AllocationReview, ChatTurn,
                                LlmCall, UserAccount)

LLM_CALL_FIELDS = ["agent_name", "model_name", "prompt_characters", "response_characters",
                   "latency_milliseconds"]
DECISION_FIELDS = ["decision", "decided_by", "decided_by_role", "decision_reason"]


def _as_record(row):
    """Every column of a mapped row, keyed by column name."""
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def insert_user_account(session, account_record):
    """Create one sign-in account."""
    session.add(UserAccount(**account_record))
    session.flush()


def fetch_user_account(session, condition):
    """One account's row, or None when none matches."""
    row = session.execute(select(UserAccount).where(condition)).scalars().first()
    return None if row is None else _as_record(row)


def update_session_token(session, username, session_token):
    """Start (a token) or end (None) an account's session, replacing any earlier token."""
    row = session.get(UserAccount, username)
    if row is None:
        raise ValueError(f"No account is held for {username}")
    row.session_token = session_token
    session.flush()


def insert_agent_run(session, run_record):
    """Record that a run has started, with a call count of nought."""
    session.add(AgentRun(**run_record, llm_call_count=0))
    session.flush()


def update_agent_run(session, run_id, status_value, llm_call_count):
    """Move a run on; finished_at is set when the status is a finished one."""
    row = session.get(AgentRun, run_id)
    if row is None:
        raise ValueError(f"No agent run has the identifier {run_id}")
    row.status = status_value
    row.llm_call_count = int(llm_call_count)
    if status_value in config.AGENT_RUN_FINISHED_STATUSES:
        row.finished_at = session.execute(select(func.now())).scalar()
    session.flush()


def insert_llm_call(session, run_id, call_record):
    """Log what one language model call cost."""
    session.add(LlmCall(run_id=run_id, **{field_name: call_record[field_name]
                                          for field_name in LLM_CALL_FIELDS}))
    session.flush()


def fetch_llm_calls(session, run_id):
    """Every call of the run, in the order the calls were made."""
    rows = session.execute(select(LlmCall).where(LlmCall.run_id == run_id)
                           .order_by(LlmCall.call_id)).scalars().all()
    return [{field_name: getattr(row, field_name) for field_name in LLM_CALL_FIELDS}
            for row in rows]


def insert_allocation_review(session, review_record):
    """Record the finished review, waiting for its decision."""
    session.add(AllocationReview(**review_record))
    session.flush()


def fetch_review_records(session, condition):
    """Reviews with the run and the decision behind each, newest first."""
    rows = session.execute(
        select(AllocationReview, AgentRun, AllocationDecision)
        .join(AgentRun, AgentRun.run_id == AllocationReview.run_id)
        .outerjoin(AllocationDecision, AllocationDecision.review_id == AllocationReview.review_id)
        .where(condition)
        .order_by(AllocationReview.created_at.desc(), AllocationReview.review_id.desc())).all()
    records = []
    for review, run, decision in rows:
        record = _as_record(review)
        record.update(thread_id=run.thread_id, planner_id=run.planner_id,
                      llm_call_count=run.llm_call_count)
        for field_name in DECISION_FIELDS:
            record[field_name] = None if decision is None else getattr(decision, field_name)
        records.append(record)
    return records


def insert_allocation_decision(session, decision_record):
    """Record a Supply Planning Manager's decision, who took it and why."""
    session.add(AllocationDecision(**decision_record))
    session.flush()


def record_allocation_outcome(session, review_id, outcome_record):
    """Close a review with what its decision confirms, holds and leaves in stock."""
    row = session.get(AllocationReview, review_id)
    if row is None:
        raise ValueError(config.REVIEW_NOT_FOUND_MESSAGE)
    for field_name, value in outcome_record.items():
        setattr(row, field_name, value)
    row.status_changed_at = session.execute(select(func.now())).scalar()
    session.flush()


def insert_chat_turn(session, turn_record):
    """Record one chat question and its answer."""
    session.add(ChatTurn(**turn_record))
    session.flush()


def fetch_chat_turns(session, condition, turn_limit):
    """At most turn_limit matching turns, newest first."""
    rows = session.execute(select(ChatTurn).where(condition)
                           .order_by(ChatTurn.asked_at.desc(), ChatTurn.turn_id.desc())
                           .limit(int(turn_limit))).scalars().all()
    return [_as_record(row) for row in rows]