"""What the workflow carries from one node to the next."""
import operator
from typing import Annotated, TypedDict


class AllocationReviewState(TypedDict):
    """Every key a node reads or writes, and nothing else.

    llm_call_count is added up across nodes (each agent node returns the one call
    it made); every other key is replaced by whichever node writes it.
    """

    product_code: str
    planner_request: str
    planner_id: str
    thread_id: str
    run_id: str
    review_id: str
    shortage_profile: dict
    open_orders: list
    customer_accounts: dict
    demand_facts: dict
    request_facts: dict
    allocation_facts: dict
    review_plan: dict
    demand_analysis: dict
    request_review: dict
    allocation_advice: dict
    allocation_decision: dict
    allocation_outcome: dict
    allocation_notice: dict
    sales_briefing: dict
    llm_call_count: Annotated[int, operator.add]