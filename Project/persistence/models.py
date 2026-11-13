"""The six tables. Table and column names are exactly as the specification gives them."""
import datetime

from sqlalchemy import JSON, DateTime, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """The declarative base every table inherits from."""


class UserAccount(Base):
    __tablename__ = "user_accounts"

    username: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    password_hash: Mapped[str] = mapped_column(String)
    password_salt: Mapped[str] = mapped_column(String)
    session_token: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String)
    product_code: Mapped[str] = mapped_column(String)
    planner_id: Mapped[str] = mapped_column(String)
    planner_request: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)


class AllocationReview(Base):
    __tablename__ = "allocation_reviews"

    review_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String)
    product_code: Mapped[str] = mapped_column(String)
    product_name: Mapped[str] = mapped_column(String)
    product_category: Mapped[str] = mapped_column(String)
    policy_document: Mapped[str] = mapped_column(String)
    planner_request: Mapped[str] = mapped_column(String)
    cases_available: Mapped[int] = mapped_column(Integer)
    cases_ordered: Mapped[int] = mapped_column(Integer)
    cases_eligible: Mapped[int] = mapped_column(Integer)
    cover_percent: Mapped[float | None] = mapped_column(nullable=True)
    protected_served_in_full: Mapped[bool] = mapped_column()
    allocation_lines: Mapped[list] = mapped_column(JSON)
    releases: Mapped[dict] = mapped_column(JSON)
    decisions_open: Mapped[list] = mapped_column(JSON)
    decisions_not_open: Mapped[dict] = mapped_column(JSON)
    inflated_order_scores: Mapped[list] = mapped_column(JSON)
    email_requests: Mapped[list] = mapped_column(JSON)
    customer_requests: Mapped[dict] = mapped_column(JSON)
    forecast_shortfall: Mapped[int] = mapped_column(Integer)
    review_plan_text: Mapped[str] = mapped_column(String)
    recommendation_text: Mapped[str] = mapped_column(String)
    recommended_decision: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    cases_confirmed: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reserve_cases: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cases_in_stock: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notice_text: Mapped[str | None] = mapped_column(String, nullable=True)
    notice_lines: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    briefing_text: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    status_changed_at: Mapped[datetime.datetime] = mapped_column(DateTime,
                                                                 server_default=func.now())


class AllocationDecision(Base):
    __tablename__ = "allocation_decisions"

    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(String)
    run_id: Mapped[str] = mapped_column(String)
    decision: Mapped[str] = mapped_column(String)
    decided_by: Mapped[str] = mapped_column(String)
    decided_by_role: Mapped[str] = mapped_column(String)
    decision_reason: Mapped[str] = mapped_column(String)
    decided_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


class LlmCall(Base):
    __tablename__ = "llm_calls"

    call_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String)
    agent_name: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    prompt_characters: Mapped[int] = mapped_column(Integer)
    response_characters: Mapped[int] = mapped_column(Integer)
    latency_milliseconds: Mapped[int] = mapped_column(Integer)
    called_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    turn_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String)
    user_role: Mapped[str] = mapped_column(String)
    question_text: Mapped[str] = mapped_column(String)
    question_type: Mapped[str] = mapped_column(String)
    account_code: Mapped[str | None] = mapped_column(String, nullable=True)
    product_code: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_text: Mapped[str] = mapped_column(String)
    source_documents: Mapped[str] = mapped_column(String)
    closest_distance: Mapped[float] = mapped_column()
    was_refused: Mapped[bool] = mapped_column()
    asked_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())