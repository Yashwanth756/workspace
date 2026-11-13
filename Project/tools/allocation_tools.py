"""Works out usual demand, divides the cases under each release, summarises the
division and the decisions open, works out what a decision confirms, and ranks
the queue. Every figure that decides a case is an exact fractions.Fraction."""
import math
from fractions import Fraction

from config import config


def _round_half_up(value, decimals):
    """An exact value rounded half up to the given places, as a float."""
    scale = 10 ** decimals
    return math.floor(Fraction(value) * scale + Fraction(1, 2)) / scale


def compute_usual_demand(product_shipments, account_codes, history_weeks):
    """Cases shipped at most history_weeks weeks ago, over history_weeks, exactly."""
    weeks = int(history_weeks)
    usual_demand = {}
    for account_code in account_codes:
        own = product_shipments.loc[
            (product_shipments[config.ACCOUNT_CODE_COLUMN] == account_code)
            & (product_shipments[config.WEEKS_AGO_COLUMN] <= weeks)]
        shipped = sum(int(cases) for cases in own[config.CASES_SHIPPED_COLUMN])
        usual_demand[account_code] = Fraction(shipped, weeks)
    return usual_demand


def _divide(cases_to_divide, account_codes, eligible, usual_demand):
    """Divide whole cases among customers, none above what it may have.

    In full when they fit; otherwise in proportion to usual demand, a share at
    or above its limit taking that limit and leaving, until none reaches it;
    then each share rounded down and the cases left going to the largest
    remainders, a tie to the larger usual demand, then the lower account code.
    """
    given = {account_code: 0 for account_code in account_codes}
    if sum(eligible[account_code] for account_code in account_codes) <= cases_to_divide:
        return {account_code: eligible[account_code] for account_code in account_codes}

    sharing = [account_code for account_code in account_codes
               if usual_demand[account_code] > 0 and eligible[account_code] > 0]
    cases_left = Fraction(cases_to_divide)
    while sharing:
        total_usual = sum(usual_demand[account_code] for account_code in sharing)
        shares = {account_code: cases_left * usual_demand[account_code] / total_usual
                  for account_code in sharing}
        capped = [account_code for account_code in sharing
                  if shares[account_code] >= eligible[account_code]]
        if not capped:
            break
        for account_code in capped:
            given[account_code] = eligible[account_code]
            cases_left -= eligible[account_code]
            sharing.remove(account_code)
    if not sharing:
        return given

    total_usual = sum(usual_demand[account_code] for account_code in sharing)
    shares = {account_code: cases_left * usual_demand[account_code] / total_usual
              for account_code in sharing}
    for account_code in sharing:
        given[account_code] = math.floor(shares[account_code])
    left_over = int(cases_left) - sum(given[account_code] for account_code in sharing)
    by_remainder = sorted(sharing, key=lambda account_code: (
        -(shares[account_code] - math.floor(shares[account_code])),
        -usual_demand[account_code], account_code))
    for account_code in by_remainder[:left_over]:
        given[account_code] += 1
    return given


def _release(cases_available, account_codes, protected, eligible, usual_demand,
             reserve_percent):
    """One release's cases per customer and the reserve it holds."""
    protected_codes = [account_code for account_code in account_codes if protected[account_code]]
    other_codes = [account_code for account_code in account_codes if not protected[account_code]]
    cases = {account_code: 0 for account_code in account_codes}
    protected_eligible = sum(eligible[account_code] for account_code in protected_codes)
    if protected_eligible > cases_available:
        cases.update(_divide(cases_available, protected_codes, eligible, usual_demand))
        return cases, 0
    for account_code in protected_codes:
        cases[account_code] = eligible[account_code]
    cases_left = cases_available - protected_eligible
    reserve = 0 if reserve_percent is None else math.floor(
        Fraction(cases_left * int(reserve_percent), 100))
    cases.update(_divide(cases_left - reserve, other_codes, eligible, usual_demand))
    return cases, reserve


def allocate_cases(cases_available, product_orders, customer_accounts, usual_demand,
                   policy_rules):
    """Divide the cases under RELEASE and, where the policy holds a reserve, under
    RELEASE WITH RESERVE."""
    cases_available = int(cases_available)
    account_codes = sorted(customer_accounts)
    cap = Fraction(policy_rules["cap"])
    protected_channels = set(policy_rules["protected_channels"])

    cases_ordered, eligible, protected = {}, {}, {}
    for account_code in account_codes:
        own = product_orders.loc[product_orders[config.ACCOUNT_CODE_COLUMN] == account_code]
        cases_ordered[account_code] = int(sum(int(cases) for cases
                                              in own[config.ORDERED_QUANTITY_COLUMN]))
        usual = Fraction(usual_demand[account_code])
        eligible[account_code] = min(cases_ordered[account_code], math.floor(cap * usual))
        protected[account_code] = (customer_accounts[account_code][config.CHANNEL_COLUMN]
                                   in protected_channels)

    releases = {config.DECISION_RELEASE: None}
    if policy_rules["reserve_percent"] is not None:
        releases[config.DECISION_RELEASE_WITH_RESERVE] = policy_rules["reserve_percent"]
    cases_by_release, reserves_by_release = {}, {}
    for release_name, reserve_percent in releases.items():
        cases, reserve = _release(cases_available, account_codes, protected, eligible,
                                  {code: Fraction(usual_demand[code]) for code in account_codes},
                                  reserve_percent)
        cases_by_release[release_name] = cases
        reserves_by_release[release_name] = int(reserve)

    allocation_lines = []
    for account_code in account_codes:
        allocation_lines.append({
            "account_code": account_code,
            "customer_name": customer_accounts[account_code][config.CUSTOMER_NAME_COLUMN],
            "channel": customer_accounts[account_code][config.CHANNEL_COLUMN],
            "protected": bool(protected[account_code]),
            "cases_ordered": cases_ordered[account_code],
            "usual_demand": _round_half_up(usual_demand[account_code],
                                           config.USUAL_DEMAND_DECIMALS),
            "eligible_cases": int(eligible[account_code]),
            "cases_by_release": {release_name: int(cases_by_release[release_name][account_code])
                                 for release_name in releases},
        })
    return {"cases_available": cases_available, "allocation_lines": allocation_lines,
            "reserves_by_release": reserves_by_release}


def summarise_allocation(allocation):
    """The totals, the cover, each release's figures and the decisions open."""
    lines = allocation["allocation_lines"]
    cases_available = int(allocation["cases_available"])
    cases_ordered = sum(line["cases_ordered"] for line in lines)
    cases_eligible = sum(line["eligible_cases"] for line in lines)
    cover = (None if cases_eligible == 0
             else Fraction(cases_available * config.PERCENT_SCALE, cases_eligible))
    protected_served_in_full = all(
        line["cases_by_release"][config.DECISION_RELEASE] == line["eligible_cases"]
        for line in lines if line["protected"])

    releases = {}
    for release_name, reserve_cases in allocation["reserves_by_release"].items():
        placed = sum(line["cases_by_release"][release_name] for line in lines)
        releases[release_name] = {
            "cases_placed": int(placed), "reserve_cases": int(reserve_cases),
            "cases_in_stock": int(cases_available - placed - reserve_cases),
            "customers_short": [line["account_code"] for line in lines
                                if line["cases_by_release"][release_name] < line["eligible_cases"]],
        }

    release_reasons = []
    if cover is None:
        release_reasons.append(config.NOTHING_ELIGIBLE_REASON)
    else:
        if not protected_served_in_full:
            release_reasons.append(config.PROTECTED_CUSTOMER_SHORT_REASON)
        if cover < config.MINIMUM_RELEASE_COVER_PERCENT:
            release_reasons.append(config.COVER_BELOW_MINIMUM_REASON)
    reserve_reasons = list(release_reasons)
    if config.DECISION_RELEASE_WITH_RESERVE not in allocation["reserves_by_release"]:
        reserve_reasons.append(config.NO_RESERVE_HELD_REASON)
    reasons = {config.DECISION_RELEASE: release_reasons,
               config.DECISION_RELEASE_WITH_RESERVE: reserve_reasons,
               config.DECISION_ESCALATE: []}
    return {
        "cases_available": cases_available,
        "cases_ordered": int(cases_ordered),
        "cases_eligible": int(cases_eligible),
        "cover_percent": None if cover is None else _round_half_up(cover, config.COVER_DECIMALS),
        "protected_served_in_full": bool(protected_served_in_full),
        "releases": releases,
        "decisions_open": [decision for decision in config.DECISIONS if not reasons[decision]],
        "decisions_not_open": {decision: config.REASON_SEPARATOR.join(reasons[decision])
                               for decision in config.DECISIONS if reasons[decision]},
    }


def compute_allocation_outcome(decision, allocation, allocation_summary, product_name):
    """What a decision confirms, holds in reserve and leaves in stock."""
    if decision not in config.DECISIONS:
        raise ValueError(f"{decision!r} is not one of the decisions {config.DECISIONS}")
    if decision not in allocation_summary["decisions_open"]:
        raise ValueError(config.DECISION_NOT_OPEN_MESSAGE.format(
            decision=decision, reasons=allocation_summary["decisions_not_open"].get(decision, "")))
    if decision == config.DECISION_ESCALATE:
        return {"decision": decision, "cases_confirmed": {}, "notice_lines": {},
                "cases_released": 0, "customers_receiving_cases": 0, "reserve_cases": 0,
                "cases_in_stock": int(allocation["cases_available"]), "channel_totals": {}}

    lines = allocation["allocation_lines"]
    release = allocation_summary["releases"][decision]
    cases_confirmed = {line["account_code"]: int(line["cases_by_release"][decision])
                       for line in lines}
    notice_lines = {line["account_code"]: config.NOTICE_LINE_FORMAT.format(
        customer_name=line["customer_name"], cases_confirmed=cases_confirmed[line["account_code"]],
        cases_ordered=line["cases_ordered"], product_name=product_name) for line in lines}
    channels = {}
    for line in lines:
        total = channels.setdefault(line["channel"], {"customers": 0, "cases_released": 0})
        total["customers"] += 1
        total["cases_released"] += cases_confirmed[line["account_code"]]
    channel_totals = {channel: channels[channel] for channel in sorted(channels)
                      if channels[channel]["customers"]
                      >= config.SALES_BRIEFING_MINIMUM_CHANNEL_CUSTOMERS}
    return {
        "decision": decision,
        "cases_confirmed": cases_confirmed,
        "notice_lines": notice_lines,
        "cases_released": int(release["cases_placed"]),
        "customers_receiving_cases": sum(1 for cases in cases_confirmed.values() if cases > 0),
        "reserve_cases": int(release["reserve_cases"]),
        "cases_in_stock": int(release["cases_in_stock"]),
        "channel_totals": channel_totals,
    }


def rank_shortage_queue(queue_entries):
    """The entries, unchanged, largest forecast shortfall first, a tie to the lower code."""
    codes = [entry["product_code"] for entry in queue_entries]
    repeated = sorted({code for code in codes if codes.count(code) > 1})
    if repeated:
        raise ValueError(f"product_code: products in the queue more than once: {repeated}")
    return sorted(queue_entries, key=lambda entry: (-entry["forecast_shortfall"],
                                                    entry["product_code"]))