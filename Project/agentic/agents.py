"""The language model client, the reply parser and the seven prompts.

Each prompt gives an agent facts that are already established and asks for a
reply in labelled fields. No agent decides a case: every figure it is given was
worked out by the tools, and it is told to use only those.
"""
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import config

load_dotenv(config.ENV_FILE)

generation_settings = types.GenerateContentConfig(
    max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
    temperature=config.GEMINI_TEMPERATURE,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

PLANNER_LABELS = ["PRIORITY", "CHECKS", "RATIONALE"]
DEMAND_LABELS = ["INFLATED ORDERS", "ORDERS AGAINST FORECASTS", "DEMAND SUMMARY"]
REQUEST_LABELS = ["CUSTOMER REQUESTS", "CHECKS TO MAKE", "REQUEST SUMMARY"]
ADVISOR_LABELS = ["PROTECTED CUSTOMERS", "LIMITS REACHED", "CUSTOMERS SHORT",
                  "ALLOCATION SUMMARY"]
WRITER_LABELS = ["RECOMMENDATION", "RECOMMENDED DECISION"]
NOTICE_LABELS = ["HEADLINE", "WHY IT IS SHORT", "HOW ALLOCATIONS WERE WORKED OUT",
                 "WHAT CUSTOMERS CAN DO"]
BRIEFING_LABELS = ["DECISION TAKEN", "WHAT CUSTOMERS WERE TOLD", "WHAT TO SAY"]


# ---------------------------------------------------------------- the client

def call_language_model(prompt_text, agent_name):
    """Make one language model call; return the reply and what the call cost."""
    api_key = os.environ.get(config.GEMINI_API_KEY_VARIABLE, "").strip()
    model_name = os.environ.get(config.GEMINI_MODEL_VARIABLE, "").strip()
    if not api_key or not model_name:
        raise ValueError(f"{agent_name}: no language model call is made without both "
                         f"{config.GEMINI_API_KEY_VARIABLE} and {config.GEMINI_MODEL_VARIABLE} "
                         f"set in {config.ENV_FILE}")
    client = genai.Client(api_key=api_key)
    started = time.perf_counter()
    response = client.models.generate_content(model=model_name, contents=prompt_text,
                                              config=generation_settings)
    latency_milliseconds = int(round((time.perf_counter() - started) * 1000))

    candidates = response.candidates or []
    finish_reason = candidates[0].finish_reason if candidates else None
    response_text = (response.text or "").strip() if candidates else ""
    if not response_text or finish_reason != types.FinishReason.STOP:
        raise ValueError(f"{agent_name}: the model {model_name} gave no complete reply "
                         f"(finish reason {finish_reason})")
    return {"response_text": response_text,
            "call_record": {"agent_name": agent_name, "model_name": model_name,
                            "prompt_characters": len(prompt_text),
                            "response_characters": len(response_text),
                            "latency_milliseconds": latency_milliseconds}}


# ---------------------------------------------------------------- the parser

def _label_key(text):
    """A label as compared: upper case, a hyphen read as a space, single spaces."""
    return re.sub(r"\s+", " ", text.replace("-", " ")).strip().upper()


def _match_label(line, field_labels):
    """(label, text after its colon) when the line starts a field, otherwise None."""
    undecorated = re.sub(r"^[\W_]+", "", line)
    for label in sorted(field_labels, key=len, reverse=True):
        words = _label_key(label).split(" ")
        pattern = r"^" + r"[\s-]+".join(re.escape(word) for word in words) + r"[\W_]*?:"
        found = re.match(pattern, undecorated, flags=re.IGNORECASE)
        if found is not None:
            return label, undecorated[found.end():]
    return None


def parse_labelled_response(response_text, field_labels, agent_name):
    """Read a labelled reply back into its fields, in the order asked for."""
    gathered = {label: [] for label in field_labels}
    current = None
    for raw_line in str(response_text).splitlines():
        line = raw_line.replace("**", "")
        started = _match_label(line, field_labels)
        if started is not None:
            current, text = started
            text = text.strip(config.RESPONSE_DECORATION_CHARACTERS).strip()
            if text:
                gathered[current].append(text)
            continue
        if current is None:
            continue
        if not re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", line.strip()):
            continue
        gathered[current].append(line.strip())
    fields = {label: "\n".join(lines).strip().strip(config.RESPONSE_DECORATION_CHARACTERS)
              .strip() for label, lines in gathered.items()}
    missing = [label for label in field_labels if not fields[label]]
    if missing:
        raise ValueError(f"{agent_name}: the reply gives nothing for {missing} of the labels "
                         f"{list(field_labels)}")
    return {label: fields[label] for label in field_labels}


def _ask(prompt_text, agent_name, field_labels):
    reply = call_language_model(prompt_text, agent_name)
    fields = parse_labelled_response(reply["response_text"], field_labels, agent_name)
    return fields, reply["call_record"]


# ---------------------------------------------------------------- shared prompt pieces

def _is_blank(value):
    return value is None or (isinstance(value, float) and value != value) or value == ""


def _shown(value):
    return config.NOT_RECORDED_TEXT if _is_blank(value) else str(value)


def _reply_instructions(field_labels):
    lines = ["Reply with exactly these labelled fields, each starting on its own line as the "
             "label, a colon and a space, and nothing before the first label:"]
    lines += [f"{label}: ..." for label in field_labels]
    return "\n".join(lines)


def _profile_lines(shortage_profile):
    return "\n".join([
        f"Product: {shortage_profile[config.PRODUCT_CODE_COLUMN]} "
        f"{shortage_profile[config.PRODUCT_NAME_COLUMN]}",
        f"Category: {shortage_profile[config.PRODUCT_CATEGORY_COLUMN]}",
        f"Pack: {shortage_profile[config.PACK_DESCRIPTION_COLUMN]}, "
        f"{shortage_profile[config.UNITS_PER_CASE_COLUMN]} units in a case, "
        f"{shortage_profile[config.CASES_PER_PALLET_COLUMN]} cases on a pallet",
        f"Cases available this week: {shortage_profile[config.CASES_AVAILABLE_THIS_WEEK_COLUMN]:,}",
        f"Cases available next week: {shortage_profile[config.CASES_AVAILABLE_NEXT_WEEK_COLUMN]:,}",
        f"Why it is short: {shortage_profile[config.SHORTAGE_REASON_COLUMN]}",
        f"Weeks on allocation: {shortage_profile[config.WEEKS_ON_ALLOCATION_COLUMN]}",
    ])


def _speaker():
    return (f"You work for {config.ORGANISATION_NAME}'s supply allocation desk. Use only the "
            f"facts given below; invent no figure, rule, customer or contact detail.")


def _rule_sentences(policy_rules):
    sentences = policy_rules["rule_sentences"]
    return "\n".join([f"- {sentences['history']}", f"- {sentences['cap']}",
                      f"- {sentences['protected']}", f"- {sentences['reserve']}",
                      f"(from {policy_rules['document_name']})"])


def _release_lines(allocation, summary):
    lines = []
    protected_any = any(line["protected"] for line in allocation["allocation_lines"])
    for release_name, release in summary["releases"].items():
        protected_cases = sum(line["cases_by_release"][release_name]
                              for line in allocation["allocation_lines"] if line["protected"])
        protected_eligible = sum(line["eligible_cases"]
                                 for line in allocation["allocation_lines"] if line["protected"])
        other_cases = sum(line["cases_by_release"][release_name]
                          for line in allocation["allocation_lines"] if not line["protected"])
        other_eligible = sum(line["eligible_cases"]
                             for line in allocation["allocation_lines"] if not line["protected"])
        split = (config.RELEASE_SPLIT_FORMAT.format(
            protected_cases=protected_cases, protected_eligible=protected_eligible,
            other_cases=other_cases, other_eligible=other_eligible) if protected_any
            else config.RELEASE_WITHOUT_PROTECTION_FORMAT.format(
                other_cases=other_cases, other_eligible=other_eligible))
        closed = ("" if release_name in summary["decisions_open"]
                  else f" (NOT OPEN: {summary['decisions_not_open'].get(release_name, '')})")
        short = ", ".join(release["customers_short"]) or "none"
        lines.append(f"{release_name}{closed}: {release['cases_placed']:,} cases placed, "
                     f"{release['reserve_cases']:,} held in reserve, "
                     f"{release['cases_in_stock']:,} left in stock; {split}; customers short: "
                     f"{short}")
    return "\n".join(lines)


def _cover_line(summary):
    if summary["cover_percent"] is None:
        return f"Cover: {config.NO_COVER_TEXT}"
    return "Cover: " + config.COVER_TEXT_FORMAT.format(cover_percent=summary["cover_percent"])


def _fields_text(fields):
    return "\n".join(f"{label}: {text}" for label, text in fields.items())


def _plain_words(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _refuse_named_customers(fields, customer_accounts, agent_name):
    written = " " + _plain_words(" ".join(fields.values())) + " "
    named = sorted(account[config.CUSTOMER_NAME_COLUMN] for account in customer_accounts.values()
                   if _plain_words(account[config.CUSTOMER_NAME_COLUMN])
                   and " " + _plain_words(account[config.CUSTOMER_NAME_COLUMN]) + " " in written)
    if named:
        raise ValueError(f"{agent_name}: the reply names the customers {named}")


# ---------------------------------------------------------------- the seven agents

def plan_allocation_review(shortage_profile, open_orders, customer_accounts, planner_request):
    """Plan what the review must establish: PRIORITY, CHECKS and RATIONALE."""
    order_lines = []
    for order in open_orders:
        account = customer_accounts[order[config.ACCOUNT_CODE_COLUMN]]
        order_lines.append(
            f"- {order[config.ORDER_LINE_ID_COLUMN]}: {account[config.CUSTOMER_NAME_COLUMN]} "
            f"({order[config.ACCOUNT_CODE_COLUMN]}, {account[config.CHANNEL_COLUMN]}), "
            f"{order[config.ORDERED_QUANTITY_COLUMN]:,} cases, by {order[config.ORDER_ROUTE_COLUMN]}")
    prompt = "\n\n".join([
        _speaker() + " You are the shortage intake planner: plan what this allocation review "
        "must establish.",
        "THE PRODUCT\n" + _profile_lines(shortage_profile),
        "OPEN ORDERS\n" + "\n".join(order_lines),
        f"WHAT THE PLANNER ASKED\n{planner_request}",
        "The category's allocation policy is read later in the review. Name no rule, week, cap, "
        "protected channel, reserve or figure you were not given above.",
        _reply_instructions(PLANNER_LABELS),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_PLANNER, PLANNER_LABELS)
    return {"review_plan": fields, "call_record": call_record}


def analyse_demand(demand_facts, open_orders, customer_accounts, shortage_profile, review_plan):
    """Say which orders look inflated and how far orders stand above the forecasts."""
    scores = {fact["order_line_id"]: fact for fact in demand_facts["inflated_order_facts"]}
    order_lines, ordered = [], {}
    for order in open_orders:
        line_id = order[config.ORDER_LINE_ID_COLUMN]
        account_code = order[config.ACCOUNT_CODE_COLUMN]
        account = customer_accounts[account_code]
        fact = scores[line_id]
        evidence = "; ".join(f"{label}: {_shown(order.get(column_name))}"
                             for label, column_name in config.ORDER_EVIDENCE_FIELDS)
        order_lines.append(
            f"- {line_id}: {account[config.CUSTOMER_NAME_COLUMN]} ({account_code}), "
            f"{order[config.ORDERED_QUANTITY_COLUMN]:,} cases by {order[config.ORDER_ROUTE_COLUMN]}; "
            f"inflated-order score {fact['inflated_order_score']:.4f}, band "
            f"{fact['inflated_order_band']}; {evidence}")
        ordered[account_code] = ordered.get(account_code, 0) + int(order[config.ORDERED_QUANTITY_COLUMN])
    forecast_lines = [
        f"- {customer_accounts[code][config.CUSTOMER_NAME_COLUMN]} ({code}): ordered "
        f"{ordered.get(code, 0):,} cases; forecast for next week {forecast:.1f} cases"
        for code, forecast in sorted(demand_facts["customer_forecasts"].items())]
    prompt = "\n\n".join([
        _speaker() + " You are the demand analyst.",
        "THE PRODUCT\n" + _profile_lines(shortage_profile),
        "ORDERS, EACH WITH ITS OWN INFLATED-ORDER SCORE\n" + "\n".join(order_lines),
        f"BANDS: {config.INFLATED_ORDER_BAND_HIGH} at or above "
        f"{config.INFLATED_ORDER_BAND_HIGH_THRESHOLD}, {config.INFLATED_ORDER_BAND_MEDIUM} at or "
        f"above {config.INFLATED_ORDER_BAND_MEDIUM_THRESHOLD}, otherwise "
        f"{config.INFLATED_ORDER_BAND_LOW}.",
        "CASES ORDERED AGAINST NEXT WEEK'S FORECAST\n" + "\n".join(forecast_lines),
        f"THE PLAN'S CHECKS\n{review_plan['CHECKS']}",
        "The scores and the forecasts are predictions; they decide nothing and move no case. "
        "Do not recalculate or invent any figure.",
        _reply_instructions(DEMAND_LABELS),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_DEMAND, DEMAND_LABELS)
    return {"demand_analysis": fields, "call_record": call_record}


def review_customer_requests(request_facts, planner_checks, shortage_profile, customer_accounts,
                             review_plan):
    """Weigh what customers are asking for against the request guides."""
    emails = request_facts["email_requests"]
    if emails:
        email_lines = "\n".join(
            f"- {email[config.EMAIL_ID_COLUMN]} from "
            f"{customer_accounts[email[config.ACCOUNT_CODE_COLUMN]][config.CUSTOMER_NAME_COLUMN]} "
            f"({email[config.ACCOUNT_CODE_COLUMN]}), {email[config.SENDER_ROLE_COLUMN]}, "
            f"{email[config.EMAIL_RECEIVED_AT_COLUMN]}, subject "
            f"\"{email[config.EMAIL_SUBJECT_COLUMN]}\", read as {email[config.EMAIL_TARGET_COLUMN]}: "
            f"{email[config.EMAIL_TEXT_COLUMN]}" for email in emails)
        urgent_lines = "\n".join(f"- {code}: {request}" for code, request
                                 in request_facts["customer_requests"].items())
    else:
        email_lines = "This product has no open customer email."
        urgent_lines = "none"
    guide_lines = "\n".join(f"- {request} (from {config.REQUEST_GUIDE_DOCUMENTS[request]}): "
                            f"{passage}" for request, passage in planner_checks.items()) or "none"
    prompt = "\n\n".join([
        _speaker() + " You are the customer request analyst.",
        "THE PRODUCT\n" + _profile_lines(shortage_profile),
        "OPEN EMAILS\n" + email_lines,
        "EACH CUSTOMER'S MOST URGENT REQUEST\n" + urgent_lines,
        "ORDER OF URGENCY: " + ", ".join(config.REQUEST_URGENCY_ORDER),
        "WHAT THE PLANNER CHECKS, FROM EACH REQUEST GUIDE\n" + guide_lines,
        f"THE PLAN'S CHECKS\n{review_plan['CHECKS']}",
        "A request is weighed, never granted: the cases are divided by the policy's rules alone. "
        "Use only the emails, their readings and the guides' passages given.",
        _reply_instructions(REQUEST_LABELS),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_REQUESTS, REQUEST_LABELS)
    return {"request_review": fields, "call_record": call_record}


def advise_on_allocation(allocation_facts, shortage_profile, review_plan):
    """Say who gets what and why."""
    allocation, summary = allocation_facts["allocation"], allocation_facts["summary"]
    customer_lines = "\n".join(
        f"- {line['account_code']} {line['customer_name']}, {line['channel']}, "
        f"{'protected' if line['protected'] else 'not protected'}: ordered "
        f"{line['cases_ordered']:,}, usual demand {line['usual_demand']:.4f}, may have "
        f"{line['eligible_cases']:,}; " + ", ".join(
            f"{release_name} {cases:,}" for release_name, cases in line["cases_by_release"].items())
        for line in allocation["allocation_lines"])
    closed = "\n".join(f"- {decision}: {reasons}" for decision, reasons
                       in summary["decisions_not_open"].items()) or "none"
    prompt = "\n\n".join([
        _speaker() + " You are the allocation advisor.",
        "THE PRODUCT\n" + _profile_lines(shortage_profile),
        "THE POLICY'S RULES\n" + _rule_sentences(allocation_facts["policy_rules"]),
        "EVERY CUSTOMER, IN ACCOUNT ORDER\n" + customer_lines,
        "EACH RELEASE\n" + _release_lines(allocation, summary),
        _cover_line(summary),
        "DECISIONS OPEN: " + ", ".join(summary["decisions_open"]),
        "DECISIONS NOT OPEN\n" + closed,
        f"THE PLAN'S CHECKS\n{review_plan['CHECKS']}",
        "Quote each customer's figures exactly as given. Never work out a share and never move "
        "a case from one customer to another.",
        _reply_instructions(ADVISOR_LABELS),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_ADVISOR, ADVISOR_LABELS)
    return {"allocation_advice": fields, "call_record": call_record}


def write_allocation_recommendation(shortage_profile, planner_request, allocation_facts,
                                    forecast_shortfall, review_plan, demand_analysis,
                                    request_review, allocation_advice):
    """Write the recommendation and recommend one of the decisions open."""
    allocation, summary = allocation_facts["allocation"], allocation_facts["summary"]
    meaning = ("customers are forecast to need more than next week's supply" if forecast_shortfall
               > 0 else "next week's supply is forecast to meet what customers need")
    decisions = "\n".join(
        f"- {decision} {config.DECISION_DESCRIPTIONS[decision]}; "
        + ("open for this product" if decision in summary["decisions_open"]
           else f"not open: {summary['decisions_not_open'].get(decision, '')}")
        for decision in config.DECISIONS)
    prompt = "\n\n".join([
        _speaker() + " You are the allocation writer. You recommend; you do not decide.",
        "THE PRODUCT\n" + _profile_lines(shortage_profile),
        f"POLICY: {allocation_facts['policy_rules']['document_name']}",
        f"WHAT THE PLANNER ASKED\n{planner_request}",
        "THE PLAN\n" + _fields_text(review_plan),
        "THE DEMAND ANALYSIS\n" + _fields_text(demand_analysis),
        "THE CUSTOMER REQUESTS\n" + _fields_text(request_review),
        "THE ALLOCATION ADVICE\n" + _fields_text(allocation_advice),
        f"TOTALS: {summary['cases_available']:,} cases available, {summary['cases_ordered']:,} "
        f"ordered, {summary['cases_eligible']:,} that customers may have",
        _cover_line(summary),
        "Every protected customer served in full under RELEASE: "
        + ("yes" if summary["protected_served_in_full"] else "no"),
        "EACH RELEASE\n" + _release_lines(allocation, summary),
        f"FORECAST SHORTFALL: {forecast_shortfall:,} cases ({meaning})",
        "THE THREE DECISIONS\n" + decisions,
        "DECISIONS OPEN FOR THIS PRODUCT: " + ", ".join(summary["decisions_open"]),
        "Never change a customer's allocation, quote every figure exactly, and answer what the "
        "planner asked. RECOMMENDED DECISION is one of the decisions open, on one line.",
        _reply_instructions(WRITER_LABELS),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_WRITER, WRITER_LABELS)
    written = fields["RECOMMENDED DECISION"]
    if "\n" in written:
        raise ValueError(f"{config.AGENT_NAME_WRITER}: RECOMMENDED DECISION runs over more than "
                         f"one line")
    decision = re.sub(r"\s+", " ", written.strip(config.RESPONSE_DECORATION_CHARACTERS + ".")
                      ).upper()
    if decision not in summary["decisions_open"]:
        raise ValueError(f"{config.AGENT_NAME_WRITER}: {written!r} is not one of the decisions "
                         f"open {summary['decisions_open']}")
    fields["RECOMMENDED DECISION"] = decision
    return {"allocation_recommendation": fields, "call_record": call_record}


def draft_allocation_notice(shortage_profile, policy_rules, allocation_outcome, customer_advice,
                            customer_accounts):
    """Write the allocation notice that goes to every customer that ordered."""
    reserve = ("a reserve of " + f"{allocation_outcome['reserve_cases']:,}"
               + " cases is held back for protected customers' urgent orders this week"
               if allocation_outcome["reserve_cases"] > 0 else "no reserve is held this week")
    prompt = "\n\n".join([
        _speaker() + " You are writing the allocation notice that goes to every customer that "
        "ordered this product.",
        f"THE PRODUCT: {shortage_profile[config.PRODUCT_NAME_COLUMN]}, "
        f"{shortage_profile[config.PACK_DESCRIPTION_COLUMN]}",
        f"WHY IT IS SHORT: {shortage_profile[config.SHORTAGE_REASON_COLUMN]}",
        "THE POLICY'S RULES\n" + _rule_sentences(policy_rules),
        f"THIS WEEK: {reserve}",
        f"ADVICE TO CUSTOMERS (from {policy_rules['document_name']})\n{customer_advice}",
        "Write plainly and courteously. Say how allocations were worked out from the rules and "
        "what customers can do from the advice. Never name a customer, give a figure that "
        "belongs to one customer or compare customers, and invent no contact details. The "
        "HEADLINE is one line.",
        _reply_instructions(NOTICE_LABELS),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_NOTICE, NOTICE_LABELS)
    if "\n" in fields["HEADLINE"]:
        raise ValueError(f"{config.AGENT_NAME_NOTICE}: the HEADLINE runs over more than one line")
    _refuse_named_customers(fields, customer_accounts, config.AGENT_NAME_NOTICE)
    return {"allocation_notice": fields, "call_record": call_record}


def draft_sales_briefing(shortage_profile, policy_document, allocation_outcome, allocation_notice,
                         briefing_guidance, customer_accounts):
    """Write the briefing for the sales team."""
    decision = allocation_outcome["decision"]
    channels = ("\n".join(f"- {channel}: {total['cases_released']:,} cases released to "
                          f"{total['customers']} customers"
                          for channel, total in allocation_outcome["channel_totals"].items())
                or "There is no channel total to give.")
    told = (f"The allocation notice's headline: {allocation_notice['HEADLINE']}"
            if allocation_notice else "No allocation notice was written.")
    prompt = "\n\n".join([
        _speaker() + " You are writing the briefing for the sales team.",
        f"THE PRODUCT: {shortage_profile[config.PRODUCT_CODE_COLUMN]} "
        f"{shortage_profile[config.PRODUCT_NAME_COLUMN]}",
        f"THE DECISION: {decision}, which {config.DECISION_DESCRIPTIONS[decision]}",
        f"TOTALS: {allocation_outcome['cases_released']:,} cases released to "
        f"{allocation_outcome['customers_receiving_cases']} customers; "
        f"{allocation_outcome['reserve_cases']:,} held in reserve; "
        f"{allocation_outcome['cases_in_stock']:,} left in stock",
        "CHANNEL TOTALS\n" + channels,
        "WHAT CUSTOMERS WERE TOLD\n" + told,
        f"BRIEFING GUIDANCE (from {policy_document})\n{briefing_guidance}",
        "Quote only the totals given, never a customer's name or any other figure, and invent no "
        "contact details. Where a notice was written, WHAT CUSTOMERS WERE TOLD quotes its "
        "headline word for word.",
        _reply_instructions(BRIEFING_LABELS),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_BRIEFING, BRIEFING_LABELS)
    _refuse_named_customers(fields, customer_accounts, config.AGENT_NAME_BRIEFING)
    if allocation_notice:
        headline = _plain_words(allocation_notice["HEADLINE"])
        if headline not in _plain_words(" ".join(fields.values())):
            raise ValueError(f"{config.AGENT_NAME_BRIEFING}: the briefing does not quote the "
                             f"notice's HEADLINE")
    return {"sales_briefing": fields, "call_record": call_record}