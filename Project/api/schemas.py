"""Every request and reply shape used by the FairShare service."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from config import config

CategoryFilter = Literal[tuple(config.QUEUE_CATEGORY_OPTIONS)]
ReviewStatusValue = Literal[tuple(config.REVIEW_STATUS_OPTIONS)]
DecisionValue = Literal[tuple(config.DECISIONS)]
PlannerRequestText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.PLANNER_REQUEST_MINIMUM_CHARACTERS)]
DecisionReasonText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.DECISION_REASON_MINIMUM_CHARACTERS)]
QuestionText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.CHAT_QUESTION_MINIMUM_CHARACTERS)]


class StrictShape(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignInRequest(StrictShape):
    username: str
    password: str


class SignedInUserResponse(StrictShape):
    username: str
    display_name: str
    role: str
    session_token: str
    pages: list[str]
    areas: list[str]


class SignedOutResponse(StrictShape):
    username: str
    signed_out: bool


class QueueRow(StrictShape):
    queue_position: int
    product_code: str
    product_name: str
    product_category: str
    cases_available_this_week: int
    cases_available_next_week: int
    cases_ordered: int
    cases_eligible: int
    cover_percent: float | None
    protected_served_in_full: bool
    decisions_open: list[str]
    forecast_shortfall: int


class ShortageDetailResponse(StrictShape):
    queue_position: int
    profile: dict
    policy_rules: dict
    allocation_lines: list[dict]
    summary: dict
    forecast_shortfall: int
    order_lines: list[dict]
    email_requests: list[dict]
    customer_requests: dict


class ReviewRequest(StrictShape):
    product_code: str
    planner_request: PlannerRequestText


class DecisionRequest(StrictShape):
    decision: DecisionValue
    decision_reason: DecisionReasonText


class ReviewRecordResponse(StrictShape):
    review_id: str
    run_id: str
    product_code: str
    product_name: str
    product_category: str
    policy_document: str
    planner_request: str
    cases_available: int
    cases_ordered: int
    cases_eligible: int
    cover_percent: float | None
    protected_served_in_full: bool
    allocation_lines: list
    releases: dict
    decisions_open: list
    decisions_not_open: dict
    inflated_order_scores: list
    email_requests: list
    customer_requests: dict
    forecast_shortfall: int
    review_plan_text: str
    recommendation_text: str
    recommended_decision: str
    status: str
    cases_confirmed: dict | None
    reserve_cases: int | None
    cases_in_stock: int | None
    notice_text: str | None
    notice_lines: dict | None
    briefing_text: str | None
    thread_id: str
    planner_id: str
    llm_call_count: int
    decision: str | None
    decided_by: str | None
    decided_by_role: str | None
    decision_reason: str | None


class CallBudgetResponse(StrictShape):
    review_id: str
    run_id: str
    call_count: int
    expected_call_count: int
    total_latency_milliseconds: int
    calls: list[dict]


class ChatRequest(StrictShape):
    question: QuestionText


class ChatResponse(StrictShape):
    question_type: str
    account_code: str | None
    product_code: str | None
    answer: str
    source_documents: list[str]
    was_refused: bool


class ChatTurnResponse(StrictShape):
    turn_id: int
    username: str
    user_role: str
    question_text: str
    question_type: str
    account_code: str | None
    product_code: str | None
    answer_text: str
    source_documents: str
    closest_distance: float
    was_refused: bool