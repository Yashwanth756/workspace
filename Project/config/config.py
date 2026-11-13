RAW_ORDER_HISTORY_TRAIN_FILE = "data/order_history_train.csv"
RAW_ORDER_HISTORY_EVAL_FILE = "data/order_history_eval.csv"
RAW_WEEKLY_DEMAND_TRAIN_FILE = "data/weekly_demand_train.csv"
RAW_WEEKLY_DEMAND_EVAL_FILE = "data/weekly_demand_eval.csv"
RAW_CUSTOMER_EMAILS_FILE = "data/customer_emails.csv"

CLEAN_ORDER_HISTORY_TRAIN_FILE = "processed/order_history_train_clean.csv"
CLEAN_ORDER_HISTORY_EVAL_FILE = "processed/order_history_eval_clean.csv"
CLEAN_WEEKLY_DEMAND_TRAIN_FILE = "processed/weekly_demand_train_clean.csv"
CLEAN_WEEKLY_DEMAND_EVAL_FILE = "processed/weekly_demand_eval_clean.csv"
CLEAN_CUSTOMER_EMAILS_FILE = "processed/customer_emails_clean.csv"

DATA_QUALITY_REPORT_FILE = "output/data_quality_report.json"
TARGET_SHARE_DECIMALS = 4


NULL_TOKENS = ["", "-", "n/a", "null", "?", "not recorded"]


BOOLEAN_YES = "Yes"
BOOLEAN_NO = "No"

BOOLEAN_MAPPING = {
    "yes": BOOLEAN_YES,
    "y": BOOLEAN_YES,
    "true": BOOLEAN_YES,
    "1": BOOLEAN_YES,
    "no": BOOLEAN_NO,
    "n": BOOLEAN_NO,
    "false": BOOLEAN_NO,
    "0": BOOLEAN_NO,
}


WEEKDAY_MAPPING = {
    "monday": "Monday",
    "mon": "Monday",
    "mo.": "Monday",
    "tuesday": "Tuesday",
    "tue": "Tuesday",
    "tu.": "Tuesday",
    "tues": "Tuesday",
    "wednesday": "Wednesday",
    "wed": "Wednesday",
    "we.": "Wednesday",
    "thursday": "Thursday",
    "thu": "Thursday",
    "th.": "Thursday",
    "thurs": "Thursday",
    "friday": "Friday",
    "fri": "Friday",
    "fr.": "Friday",
    "saturday": "Saturday",
    "sat": "Saturday",
    "sa.": "Saturday",
    "sunday": "Sunday",
    "sun": "Sunday",
    "su.": "Sunday",
}


PRODUCT_CATEGORY_MAPPING = {
    "hand hygiene": "Hand hygiene",
    "surface disinfection": "Surface disinfection",
    "laundry": "Laundry",
    "dishwash": "Dishwash",
    "toilet care": "Toilet care",
}

CHANNEL_MAPPING = {
    "supermarket": "Supermarket",
    "supermkt": "Supermarket",
    "grocery multiple": "Supermarket",
    "convenience": "Convenience",
    "conv": "Convenience",
    "convenience store": "Convenience",
    "discounter": "Discounter",
    "discount": "Discounter",
    "discount retailer": "Discounter",
    "wholesale": "Wholesale",
    "wholesaler": "Wholesale",
    "cash and carry": "Wholesale",
    "online": "Online",
    "online retailer": "Online",
    "e-commerce": "Online",
    "pharmacy": "Pharmacy",
    "pharmacy chain": "Pharmacy",
    "chemist": "Pharmacy",
    "healthcare": "Healthcare",
    "health care": "Healthcare",
    "healthcare supplier": "Healthcare",
    "facilities": "Facilities",
    "contract cleaning": "Facilities",
    "facilities management": "Facilities",
}

REGION_MAPPING = {
    "north east": "North East",
    "ne": "North East",
    "n. east": "North East",
    "north west": "North West",
    "nw": "North West",
    "n. west": "North West",
    "yorkshire": "Yorkshire",
    "yorks": "Yorkshire",
    "yorkshire and humber": "Yorkshire",
    "east midlands": "East Midlands",
    "e midlands": "East Midlands",
    "e. midlands": "East Midlands",
    "west midlands": "West Midlands",
    "w midlands": "West Midlands",
    "w. midlands": "West Midlands",
    "east of england": "East of England",
    "east": "East of England",
    "e of england": "East of England",
    "london": "London",
    "greater london": "London",
    "ldn": "London",
    "south east": "South East",
    "se": "South East",
    "s. east": "South East",
    "south west": "South West",
    "sw": "South West",
    "s. west": "South West",
    "wales": "Wales",
    "cymru": "Wales",
    "scotland": "Scotland",
    "scot": "Scotland",
}

SENDER_ROLE_MAPPING = {
    "buyer": "Buyer",
    "category buyer": "Buyer",
    "store manager": "Store manager",
    "branch manager": "Store manager",
    "clinical lead": "Clinical lead",
    "lead nurse": "Clinical lead",
    "procurement officer": "Procurement officer",
    "procurement": "Procurement officer",
    "operations manager": "Operations manager",
    "ops manager": "Operations manager",
}

REQUEST_TYPE_MAPPING = {
    "urgent clinical need": "Urgent clinical need",
    "clinical urgency": "Urgent clinical need",
    "order cancellation": "Order cancellation",
    "cancellation": "Order cancellation",
    "quantity increase": "Quantity increase",
    "qty increase": "Quantity increase",
    "expedite request": "Expedite request",
    "expedite": "Expedite request",
    "substitution accepted": "Substitution accepted",
    "substitute accepted": "Substitution accepted",
    "delivery complaint": "Delivery complaint",
    "delivery issue": "Delivery complaint",
    "forecast update": "Forecast update",
    "forecast change": "Forecast update",
    "allocation query": "Allocation query",
    "allocation question": "Allocation query",
}


CANONICAL_ACCOUNT_CODE_PATTERN = r"^CUS-[0-9]{4}$"
CANONICAL_ACCOUNT_CODE_FORMAT = "CUS-{account_number:04d}"
ACCOUNT_CODE_NOTATIONS = {
    "EDI": r"^C(?P<account_number>[0-9]{4})$",
    "Portal": r"^CUS-(?P<account_number>[0-9]{4})$",
    "Email": r"^Acct (?P<account_number>[1-9][0-9]{0,3})$",
}
ACCOUNT_NUMBER_PART = "account_number"

CONSUMER_UNITS_WORD = "units"
CASES_WORD = "cs"
PALLETS_WORD = "plt"
QUANTITY_NOTATIONS = {
    "EDI": rf"^(?P<amount>[1-9][0-9]*) (?P<unit>{CONSUMER_UNITS_WORD})$",
    "Portal": rf"^(?P<amount>[1-9][0-9]*) (?P<unit>{CASES_WORD})$",
    "Email": rf"^(?P<amount>[1-9][0-9]*) (?P<unit>{CASES_WORD}|{PALLETS_WORD})$",
}
QUANTITY_AMOUNT_PART = "amount"
QUANTITY_UNIT_PART = "unit"

PACK_DESCRIPTION_NOTATIONS = [
    r"^(?P<units_per_case>[1-9][0-9]*)x(?P<unit_millilitres>[1-9][0-9]*)ml$",
    r"^(?P<units_per_case>[1-9][0-9]*) x (?P<unit_millilitres>[1-9][0-9]*) ml$",
    r"^(?P<unit_millilitres>[1-9][0-9]*)ml x(?P<units_per_case>[1-9][0-9]*)$",
    r"^(?P<units_per_case>[1-9][0-9]*)/(?P<unit_millilitres>[1-9][0-9]*)ML$",
]
UNITS_PER_CASE_PART = "units_per_case"
UNIT_MILLILITRES_PART = "unit_millilitres"
CANONICAL_PACK_DESCRIPTION_FORMAT = "{units_per_case}x{unit_millilitres}ml"
CANONICAL_PACK_DESCRIPTION_PATTERN = r"^[0-9]+x[0-9]+ml$"

CUSTOMER_NAME_LEGAL_SUFFIX_PATTERN = r" (?:Ltd|Limited)$"


WHOLE_NUMBER_PATTERN = r"^(-?[0-9]+)$"
COUNT_PATTERN = r"^([0-9]+)$"
POSITIVE_WHOLE_NUMBER_PATTERN = r"^([1-9][0-9]*)$"
DECIMAL_NUMBER_PATTERN = r"^(-?[0-9]+(?:\.[0-9]+)?)$"
NON_NEGATIVE_DECIMAL_PATTERN = r"^([0-9]+(?:\.[0-9]+)?)$"
DECIMAL_PERCENTAGE_PATTERN = r"^(-?[0-9]+(?:\.[0-9]+)?)(?: ?%)?$"
ORDERS_COUNT_PATTERN = r"^([0-9]+)(?: orders?)?$"
SITES_COUNT_PATTERN = r"^([0-9]+)(?: sites?)?$"
DAYS_COUNT_PATTERN = r"^(-?[0-9]+)(?: days?)?$"
WHOLE_WEEKS_PATTERN = r"^([0-9]+)(?: wks?| weeks?)?$"
DECIMAL_WEEKS_PATTERN = r"^(-?[0-9]+(?:\.[0-9]+)?)(?: ?wks| weeks)?$"
CASES_COUNT_PATTERN = r"^(-?[0-9]+)(?: cs)?$"


MARKUP_PATTERN = r"<[^>]+>"
ENCODING_ARTIFACT_PATTERN = r"&[#a-z0-9]+;|\xa0"
WHITESPACE_RUN_PATTERN = r"[\s\xa0]+"
MINIMUM_EMAIL_WORDS = 6
RECEIVED_AT_PATTERN = (r"^(?P<received_weekday>\S+) "
                       r"(?P<received_time>(?:[01][0-9]|2[0-3]):[0-5][0-9])$")
RECEIVED_WEEKDAY_PART = "received_weekday"
RECEIVED_TIME_PART = "received_time"


ORDER_TARGET_COLUMN = "cut_after_shortage"
ORDER_LINE_ID_COLUMN = "order_line_id"
ORDER_REFERENCE_COLUMN = "order_reference"
ORDER_ROUTE_COLUMN = "order_route"
ACCOUNT_CODE_COLUMN = "account_code"
CUSTOMER_NAME_COLUMN = "customer_name"
RECEIVED_SEQUENCE_COLUMN = "received_sequence"
PRODUCT_CODE_COLUMN = "product_code"
PACK_DESCRIPTION_COLUMN = "pack_description"
CASES_PER_PALLET_COLUMN = "cases_per_pallet"
ORDERED_QUANTITY_COLUMN = "ordered_quantity"
ORDER_WEEKDAY_COLUMN = "weekday_entered"
ORDER_KEY_COLUMNS = [ORDER_REFERENCE_COLUMN, PRODUCT_CODE_COLUMN]

ORDER_VALUE_MAPPINGS = {
    "product_category": PRODUCT_CATEGORY_MAPPING,
    "same_order_other_route_flag": BOOLEAN_MAPPING,
    "channel": CHANNEL_MAPPING,
    "region": REGION_MAPPING,
    "key_account_flag": BOOLEAN_MAPPING,
    "shortage_publicised_flag": BOOLEAN_MAPPING,
    "promotion_flag": BOOLEAN_MAPPING,
    "weekday_entered": WEEKDAY_MAPPING,
}

ORDER_TARGET_VALUE_MAPPINGS = {
    "cut_after_shortage": BOOLEAN_MAPPING,
}

ORDER_WHOLE_NUMBER_PATTERNS = {
    "received_sequence": POSITIVE_WHOLE_NUMBER_PATTERN,
    "cases_per_pallet": POSITIVE_WHOLE_NUMBER_PATTERN,
    "orders_this_week": ORDERS_COUNT_PATTERN,
    "lead_time_days": DAYS_COUNT_PATTERN,
    "sites_supplied": SITES_COUNT_PATTERN,
    "account_age_weeks": WHOLE_WEEKS_PATTERN,
    "past_shortage_orders": COUNT_PATTERN,
    "past_orders_cut": COUNT_PATTERN,
    "weeks_on_allocation": COUNT_PATTERN,
}

ORDER_DECIMAL_NUMBER_PATTERNS = {
    "ordered_vs_usual_pct": DECIMAL_PERCENTAGE_PATTERN,
    "customer_stock_cover_weeks": DECIMAL_WEEKS_PATTERN,
}

ORDER_MINIMUM_VALUES = {
    "ordered_vs_usual_pct": 0,
    "lead_time_days": 0,
    "customer_stock_cover_weeks": 0,
}

ORDER_COUNT_TOTALS = {
    "past_orders_cut": "past_shortage_orders",
}


DEMAND_TARGET_COLUMN = "next_week_cases"
DEMAND_WEEKDAY_COLUMN = "extract_weekday"

DEMAND_INPUT_VALUE_MAPPINGS = {
    "promotion_next_week_flag": BOOLEAN_MAPPING,
    "distribution_change_flag": BOOLEAN_MAPPING,
}

DEMAND_INPUT_WHOLE_NUMBER_PATTERNS = {
    "last_week_cases": CASES_COUNT_PATTERN,
    "seasonal_index": WHOLE_NUMBER_PATTERN,
    "weeks_since_launch": WHOLE_WEEKS_PATTERN,
}

DEMAND_INPUT_MINIMUM_VALUES = {
    "last_week_cases": 0,
    "seasonal_index": 0,
}

DEMAND_VALUE_MAPPINGS = {
    "channel": CHANNEL_MAPPING,
    "region": REGION_MAPPING,
    "product_category": PRODUCT_CATEGORY_MAPPING,
    **DEMAND_INPUT_VALUE_MAPPINGS,
    "key_account_flag": BOOLEAN_MAPPING,
    "extract_weekday": WEEKDAY_MAPPING,
}

DEMAND_WHOLE_NUMBER_PATTERNS = {
    "sites_supplied": SITES_COUNT_PATTERN,
    **DEMAND_INPUT_WHOLE_NUMBER_PATTERNS,
    "next_week_cases": WHOLE_NUMBER_PATTERN,
}

DEMAND_DECIMAL_NUMBER_PATTERNS = {
    "avg_4_week_cases": NON_NEGATIVE_DECIMAL_PATTERN,
    "avg_26_week_cases": NON_NEGATIVE_DECIMAL_PATTERN,
}

DEMAND_MINIMUM_VALUES = {
    **DEMAND_INPUT_MINIMUM_VALUES,
    "next_week_cases": 0,
}


EMAIL_TARGET_COLUMN = "request_type"
EMAIL_TEXT_COLUMN = "email_text"
EMAIL_LENGTH_COLUMN = "email_length_chars"
EMAIL_RECEIVED_AT_COLUMN = "received_at"

EMAIL_VALUE_MAPPINGS = {
    "channel": CHANNEL_MAPPING,
    "sender_role": SENDER_ROLE_MAPPING,
    "request_type": REQUEST_TYPE_MAPPING,
}

EMAIL_WHOLE_NUMBER_PATTERNS = {
    "email_sequence": POSITIVE_WHOLE_NUMBER_PATTERN,
}


RANDOM_SEED = 42
EMAIL_HELD_OUT_SHARE = 0.2
METRIC_DECIMALS = 4

INFLATED_ORDER_MODEL_FILE = "output/models/inflated_order_model.pkl"
DEMAND_MODEL_FILE = "output/models/demand_model.pkl"
REQUEST_MODEL_FILE = "output/models/request_model.pkl"
INFLATED_ORDER_SCHEMA_FILE = "output/models/inflated_order_features.json"
DEMAND_SCHEMA_FILE = "output/models/demand_features.json"
MODEL_METRICS_FILE = "output/model_metrics.json"

ORDER_TARGET_ENCODING = {BOOLEAN_YES: 1, BOOLEAN_NO: 0}

INFLATED_ORDER_NUMERIC_FEATURES = [
    "cases_per_pallet", "ordered_quantity", "ordered_vs_usual_pct", "orders_this_week",
    "lead_time_days", "sites_supplied", "customer_stock_cover_weeks", "account_age_weeks",
    "past_shortage_orders", "past_orders_cut", "weeks_on_allocation",
]

INFLATED_ORDER_CATEGORICAL_FEATURES = [
    "order_route", "product_category", "pack_description", "same_order_other_route_flag",
    "channel", "region", "key_account_flag", "shortage_publicised_flag", "promotion_flag",
]

DEMAND_NUMERIC_FEATURES = [
    "sites_supplied", "last_week_cases", "avg_4_week_cases", "avg_26_week_cases",
    "seasonal_index", "weeks_since_launch",
]

DEMAND_CATEGORICAL_FEATURES = [
    "channel", "region", "product_category", "pack_description", "promotion_next_week_flag",
    "distribution_change_flag", "key_account_flag",
]

MINIMUM_INFLATED_ORDER_ROC_AUC = 0.72
MINIMUM_DEMAND_R2 = 0.84
MINIMUM_REQUEST_MACRO_F1 = 0.85


DOCS_DIRECTORY = "data/docs"
DATABASE_DIRECTORY = "database"
VECTOR_STORE_DIRECTORY = f"{DATABASE_DIRECTORY}/chroma_store"
VECTOR_STORE_FILE_NAME = "chroma.sqlite3"
VECTOR_COLLECTION_NAME = "allocation_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
VECTOR_DISTANCE_METRIC = "cosine"

CHUNK_SIZE_CHARACTERS = 700
CHUNK_OVERLAP_CHARACTERS = 120
DOCUMENT_HEADER_PATTERN = (r"Document\s+[A-Za-z0-9 ]+\|\s*Issue\s*[0-9]+"
                           r".*?Review cycle:\s*[A-Za-z ]+")
SECTION_HEADING_PATTERN = r"(?m)^([0-9]{1,2})\. [A-Z]"
SENTENCE_BOUNDARY_PATTERN = r"(?<=[.:;])(?<![0-9]\.)\s+"

CATEGORY_POLICY_DOCUMENTS = {
    "Hand hygiene": "Allocation-Policy-Hand-Hygiene.pdf",
    "Surface disinfection": "Allocation-Policy-Surface-Disinfection.pdf",
    "Laundry": "Allocation-Policy-Laundry.pdf",
    "Dishwash": "Allocation-Policy-Dishwash.pdf",
    "Toilet care": "Allocation-Policy-Toilet-Care.pdf",
}

HISTORY_SENTENCE_PATTERN = (r"Usual demand is the average weekly cases shipped over the last "
                            r"([1-9][0-9]*) weeks\.")
CAP_SENTENCE_PATTERN = (r"An order counts for no more than ([0-9]+\.[0-9]+) times the customer's "
                        r"usual demand\.")
PROTECTED_SENTENCE_PATTERN = (r"Customers in the ([A-Z][a-z]+(?: and [A-Z][a-z]+)*) channels? are "
                              r"protected and are served first\.")
NO_PROTECTION_SENTENCE_PATTERN = r"No customer is protected under this policy\."
RESERVE_SENTENCE_PATTERN = (r"Up to ([1-9][0-9]?) percent of the cases left after protected "
                            r"customers may be held back as a reserve\.")
NO_RESERVE_SENTENCE_PATTERN = r"No reserve is held under this policy\."
PROTECTED_CHANNEL_SEPARATOR = " and "
NO_POLICY_CONFIGURED_MESSAGE = "No allocation policy is configured for {product_category}"
INDEX_HOLDS_NO_CHUNKS_MESSAGE = "The index holds no chunks of {document_name}"


CHANNEL_COLUMN = "channel"
ACCOUNT_STATUS_COLUMN = "account_status"
ACCOUNT_STATUS_ACTIVE = "Active"
PRODUCT_NOT_IN_QUEUE_MESSAGE = "Product {product_code} is not in the shortage queue"

SHORTAGE_PRODUCT_WHOLE_NUMBER_PATTERNS = {
    "units_per_case": POSITIVE_WHOLE_NUMBER_PATTERN,
    "cases_per_pallet": POSITIVE_WHOLE_NUMBER_PATTERN,
    "cases_available_this_week": COUNT_PATTERN,
    "cases_available_next_week": COUNT_PATTERN,
    "weeks_on_allocation": COUNT_PATTERN,
    "seasonal_index": COUNT_PATTERN,
}

OPEN_ORDER_VALUE_MAPPINGS = {
    **ORDER_VALUE_MAPPINGS,
    **DEMAND_INPUT_VALUE_MAPPINGS,
}

OPEN_ORDER_WHOLE_NUMBER_PATTERNS = {
    **ORDER_WHOLE_NUMBER_PATTERNS,
    **DEMAND_INPUT_WHOLE_NUMBER_PATTERNS,
}

OPEN_ORDER_DECIMAL_NUMBER_PATTERNS = {
    **ORDER_DECIMAL_NUMBER_PATTERNS,
    **DEMAND_DECIMAL_NUMBER_PATTERNS,
}

OPEN_ORDER_MINIMUM_VALUES = {
    **ORDER_MINIMUM_VALUES,
    **DEMAND_INPUT_MINIMUM_VALUES,
}

SHIPMENT_REFERENCE_COLUMN = "shipment_reference"
WEEKS_AGO_COLUMN = "weeks_ago"
CASES_SHIPPED_COLUMN = "cases_shipped"
SHIPMENT_WHOLE_NUMBER_PATTERNS = {
    "weeks_ago": POSITIVE_WHOLE_NUMBER_PATTERN,
    "cases_shipped": POSITIVE_WHOLE_NUMBER_PATTERN,
}

EMAIL_ID_COLUMN = "email_id"
EMAIL_SEQUENCE_COLUMN = "email_sequence"
OPEN_EMAIL_VALUE_MAPPINGS = {
    "sender_role": SENDER_ROLE_MAPPING,
}
OPEN_EMAIL_FIELDS = ["email_id", "account_code", "email_sequence", "received_at", "sender_role",
                     "email_subject", "email_text"]

INFLATED_ORDER_SCORE_DECIMALS = 4
INFLATED_ORDER_BAND_HIGH = "HIGH"
INFLATED_ORDER_BAND_MEDIUM = "MEDIUM"
INFLATED_ORDER_BAND_LOW = "LOW"
INFLATED_ORDER_BAND_HIGH_THRESHOLD = 0.60
INFLATED_ORDER_BAND_MEDIUM_THRESHOLD = 0.30

REQUEST_URGENCY_ORDER = [
    "Urgent clinical need", "Order cancellation", "Quantity increase", "Expedite request",
    "Substitution accepted", "Delivery complaint", "Forecast update", "Allocation query",
]

PERCENT_SCALE = 100
USUAL_DEMAND_DECIMALS = 4
COVER_DECIMALS = 2
MINIMUM_RELEASE_COVER_PERCENT = 50

DECISION_RELEASE = "RELEASE"
DECISION_RELEASE_WITH_RESERVE = "RELEASE WITH RESERVE"
DECISION_ESCALATE = "ESCALATE"
DECISIONS = [DECISION_RELEASE, DECISION_RELEASE_WITH_RESERVE, DECISION_ESCALATE]
PROTECTED_CUSTOMER_SHORT_REASON = "a protected customer would go short"
NOTHING_ELIGIBLE_REASON = "no customer may have any of the cases"
COVER_BELOW_MINIMUM_REASON = (f"the cases cover less than {MINIMUM_RELEASE_COVER_PERCENT} percent "
                              f"of what customers may have")
NO_RESERVE_HELD_REASON = "the policy holds no reserve"
DECISION_NOT_OPEN_MESSAGE = "{decision} is not open: {reasons}"
REASON_SEPARATOR = " and "

NOTICE_LINE_FORMAT = ("{customer_name}: {cases_confirmed:,} of the {cases_ordered:,} cases ordered "
                      "of {product_name}")
SALES_BRIEFING_MINIMUM_CHANNEL_CUSTOMERS = 3

REQUEST_GUIDE_DOCUMENTS = {
    "Urgent clinical need": "Guide-Urgent-Clinical-Need.pdf",
    "Order cancellation": "Guide-Order-Cancellation.pdf",
    "Quantity increase": "Guide-Quantity-Increase.pdf",
    "Expedite request": "Guide-Expedite-Request.pdf",
    "Substitution accepted": "Guide-Substitution-Accepted.pdf",
    "Delivery complaint": "Guide-Delivery-Complaint.pdf",
    "Forecast update": "Guide-Forecast-Update.pdf",
    "Allocation query": "Guide-Allocation-Query.pdf",
}
PLANNER_CHECKS_QUESTION = "What does the planner check?"
CUSTOMER_ADVICE_QUESTION = "What is the advice to customers while the product is on allocation?"
SALES_BRIEFING_QUESTION = "What should the sales team be told?"


ORGANISATION_NAME = "Wrenfield Hygiene"

SHORTAGE_PRODUCTS_FILE = "data/input/shortage_products.csv"
OPEN_ORDERS_FILE = "data/input/open_orders.csv"
SHIPMENT_HISTORY_FILE = "data/input/shipment_history.csv"
CUSTOMER_MASTER_FILE = "data/input/customer_master.csv"
OPEN_CUSTOMER_EMAILS_FILE = "data/input/open_customer_emails.csv"

PRODUCT_NAME_COLUMN = "product_name"
PRODUCT_CATEGORY_COLUMN = "product_category"
UNITS_PER_CASE_COLUMN = "units_per_case"
CASES_AVAILABLE_THIS_WEEK_COLUMN = "cases_available_this_week"
CASES_AVAILABLE_NEXT_WEEK_COLUMN = "cases_available_next_week"
SHORTAGE_REASON_COLUMN = "shortage_reason"
WEEKS_ON_ALLOCATION_COLUMN = "weeks_on_allocation"
SENDER_ROLE_COLUMN = "sender_role"
EMAIL_SUBJECT_COLUMN = "email_subject"

ORDER_EVIDENCE_FIELDS = [
    ("order against its average week over the last four weeks, percent", "ordered_vs_usual_pct"),
    ("orders this week", "orders_this_week"),
    ("also sent by another route", "same_order_other_route_flag"),
    ("customer's stock cover in weeks", "customer_stock_cover_weeks"),
    ("shortage in the news", "shortage_publicised_flag"),
    ("on promotion", "promotion_flag"),
    ("past shortage orders", "past_shortage_orders"),
    ("past orders cut", "past_orders_cut"),
]
NOT_RECORDED_TEXT = "not recorded"
COVER_TEXT_FORMAT = ("{cover_percent:.2f} percent: the cases available as a share of what all "
                     "the customers together may have, protected or not")
NO_COVER_TEXT = "not worked out, as no customer may have any of the cases"
RELEASE_SPLIT_FORMAT = ("the protected customers get {protected_cases:,} of the "
                        "{protected_eligible:,} cases they may have, and the other customers "
                        "{other_cases:,} of the {other_eligible:,} cases they may have")
RELEASE_WITHOUT_PROTECTION_FORMAT = ("no customer of this product is protected, and the "
                                     "customers get {other_cases:,} of the {other_eligible:,} "
                                     "cases they may have")

DECISION_DESCRIPTIONS = {
    DECISION_RELEASE: "confirms each customer's cases as divided, with no reserve held",
    DECISION_RELEASE_WITH_RESERVE: ("confirms each customer's cases as divided, with the policy's "
                                    "reserve held back for protected customers' urgent orders"),
    DECISION_ESCALATE: ("confirms no case to any customer and sends the shortage to the supply "
                        "review"),
}

APPLICATION_DATABASE_FILE = f"{DATABASE_DIRECTORY}/fairshare.db"
APPLICATION_DATABASE_URL = f"sqlite:///./{APPLICATION_DATABASE_FILE}"

RUN_ID_PREFIX = "RUN-"
REVIEW_ID_PREFIX = "REV-"
IDENTIFIER_HEX_LENGTH = 12

ENV_FILE = ".env"
GEMINI_API_KEY_VARIABLE = "GEMINI_API_KEY"
GEMINI_MODEL_VARIABLE = "GEMINI_MODEL"
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_TEMPERATURE = 0.2

AGENT_NAME_PLANNER = "shortage_intake_planner"
AGENT_NAME_DEMAND = "demand_analyst"
AGENT_NAME_REQUESTS = "customer_request_analyst"
AGENT_NAME_ADVISOR = "allocation_advisor"
AGENT_NAME_WRITER = "allocation_writer"
AGENT_NAME_NOTICE = "allocation_notice_agent"
AGENT_NAME_BRIEFING = "sales_briefing_agent"

RESPONSE_DECORATION_CHARACTERS = "*#`_~ \t:[]"
RESPONSE_DECORATION_EDGE_PATTERN = r"^[\W_]+|[\W_]+$"

AGENT_RUN_STATUS_RUNNING = "RUNNING"
AGENT_RUN_STATUS_AWAITING_DECISION = "AWAITING_DECISION"
AGENT_RUN_STATUS_COMPLETE = "COMPLETE"
AGENT_RUN_STATUS_FAILED = "FAILED"
AGENT_RUN_FINISHED_STATUSES = [AGENT_RUN_STATUS_COMPLETE, AGENT_RUN_STATUS_FAILED]

REVIEW_STATUS_PENDING = "PENDING_DECISION"
REVIEW_STATUS_RELEASED = "RELEASED"
REVIEW_STATUS_RELEASED_WITH_RESERVE = "RELEASED_WITH_RESERVE"
REVIEW_STATUS_ESCALATED = "ESCALATED"
REVIEW_STATUS_BY_DECISION = {
    DECISION_RELEASE: REVIEW_STATUS_RELEASED,
    DECISION_RELEASE_WITH_RESERVE: REVIEW_STATUS_RELEASED_WITH_RESERVE,
    DECISION_ESCALATE: REVIEW_STATUS_ESCALATED,
}


CHECKPOINT_DATABASE_FILE = f"{DATABASE_DIRECTORY}/checkpoint.db"

LLM_CALLS_BEFORE_DECISION = 5
LLM_CALLS_BY_DECISION = {
    DECISION_RELEASE: 7,
    DECISION_RELEASE_WITH_RESERVE: 7,
    DECISION_ESCALATE: 6,
}

REVIEW_STATUS_OPTIONS = [REVIEW_STATUS_PENDING, REVIEW_STATUS_RELEASED,
                         REVIEW_STATUS_RELEASED_WITH_RESERVE, REVIEW_STATUS_ESCALATED]
DECIDED_REVIEW_STATUSES = [REVIEW_STATUS_RELEASED, REVIEW_STATUS_RELEASED_WITH_RESERVE,
                           REVIEW_STATUS_ESCALATED]
REVIEW_STATUS_LABELS = {
    REVIEW_STATUS_PENDING: "Pending decision",
    REVIEW_STATUS_RELEASED: "Released",
    REVIEW_STATUS_RELEASED_WITH_RESERVE: "Released with reserve",
    REVIEW_STATUS_ESCALATED: "Escalated",
}
POLICY_RULE_LABELS = {"history": "Usual demand", "cap": "Cap on an order",
                      "protected": "Protected customers", "reserve": "Reserve"}
QUEUE_CATEGORY_ALL = "All"
QUEUE_CATEGORY_OPTIONS = [QUEUE_CATEGORY_ALL] + list(CATEGORY_POLICY_DOCUMENTS)

PLANNER_REQUEST_MINIMUM_CHARACTERS = 5
DECISION_REASON_MINIMUM_CHARACTERS = 10

ROLE_PLANNER = "Supply Planner"
ROLE_MANAGER = "Supply Planning Manager"
ROLE_ADVISER = "Customer Service Adviser"

PAGE_QUEUE = "Shortage queue"
PAGE_DETAIL = "Allocation detail"
PAGE_DECISION = "Decision queue"
PAGE_AUDIT = "Audit trail"
PAGE_CHAT = "Allocation chat"

ROLE_PAGE_ACCESS = {
    ROLE_MANAGER: [PAGE_QUEUE, PAGE_DETAIL, PAGE_DECISION, PAGE_AUDIT, PAGE_CHAT],
    ROLE_PLANNER: [PAGE_QUEUE, PAGE_DETAIL, PAGE_AUDIT, PAGE_CHAT],
    ROLE_ADVISER: [PAGE_CHAT],
}

AREA_INTAKE = "intake"
AREA_REVIEW = "review"
AREA_DECISION = "decision"
AREA_AUDIT = "audit"
AREA_CHAT = "chat"

ROLE_AREA_ACCESS = {
    ROLE_MANAGER: [AREA_INTAKE, AREA_REVIEW, AREA_DECISION, AREA_AUDIT, AREA_CHAT],
    ROLE_PLANNER: [AREA_INTAKE, AREA_REVIEW, AREA_AUDIT, AREA_CHAT],
    ROLE_ADVISER: [AREA_CHAT],
}

AREA_BY_ROUTE = {
    ("GET", "/queue"): AREA_INTAKE,
    ("GET", "/queue/{product_code}"): AREA_INTAKE,
    ("POST", "/reviews"): AREA_REVIEW,
    ("GET", "/reviews"): AREA_AUDIT,
    ("POST", "/reviews/{review_id}/decision"): AREA_DECISION,
    ("GET", "/reviews/{review_id}/calls"): AREA_AUDIT,
    ("POST", "/chat/ask"): AREA_CHAT,
    ("GET", "/chat/turns"): AREA_CHAT,
    ("GET", "/audit/chat-turns"): AREA_AUDIT,
}

SEED_USER_ACCOUNTS = [
    {"username": "priya", "display_name": "Priya Raman", "role": ROLE_MANAGER,
     "password": "11111111"},
    {"username": "tom", "display_name": "Tom Hadley", "role": ROLE_MANAGER,
     "password": "11111111"},
    {"username": "grace", "display_name": "Grace Odum", "role": ROLE_MANAGER,
     "password": "11111111"},
    {"username": "ben", "display_name": "Ben Carver", "role": ROLE_PLANNER,
     "password": "22222222"},
    {"username": "aisha", "display_name": "Aisha Khan", "role": ROLE_PLANNER,
     "password": "22222222"},
    {"username": "owen", "display_name": "Owen Price", "role": ROLE_PLANNER,
     "password": "22222222"},
    {"username": "lucy", "display_name": "Lucy Marsh", "role": ROLE_PLANNER,
     "password": "22222222"},
    {"username": "marek", "display_name": "Marek Nowak", "role": ROLE_PLANNER,
     "password": "22222222"},
    {"username": "ruth", "display_name": "Ruth Ellis", "role": ROLE_PLANNER,
     "password": "22222222"},
    {"username": "dev", "display_name": "Dev Patel", "role": ROLE_ADVISER,
     "password": "33333333"},
    {"username": "nina", "display_name": "Nina Brooks", "role": ROLE_ADVISER,
     "password": "33333333"},
    {"username": "omar", "display_name": "Omar Haddad", "role": ROLE_ADVISER,
     "password": "33333333"},
    {"username": "sofia", "display_name": "Sofia Rossi", "role": ROLE_ADVISER,
     "password": "33333333"},
    {"username": "jack", "display_name": "Jack Turner", "role": ROLE_ADVISER,
     "password": "33333333"},
]

PASSWORD_HASH_ALGORITHM = "sha256"
PASSWORD_HASH_ROUNDS = 200000
PASSWORD_SALT_BYTES = 16
SESSION_TOKEN_BYTES = 24

SESSION_TOKEN_HEADER = "X-Session-Token"
SIGN_IN_FAILED_MESSAGE = "Username or password is not correct"
SIGN_IN_REQUIRED_MESSAGE = "Sign in first"
SESSION_ENDED_MESSAGE = ("Your session ended: this account has signed in somewhere else or "
                         "signed out. Sign in again to carry on.")
AREA_REFUSED_MESSAGE = "The {role} role does not include the {area} area"
REVIEW_NOT_FOUND_MESSAGE = "No allocation review has that identifier"
PRODUCT_ALREADY_REVIEWED_MESSAGE = ("Product {product_code} has already been reviewed; a product "
                                    "on allocation is reviewed once")
ALREADY_DECIDED_MESSAGE = "This allocation review has already been decided"
DECISION_IN_PROGRESS_MESSAGE = ("{decided_by} has sent {decision} for this review and it is still "
                                "being recorded; only {decided_by} can finish it, by sending the "
                                "same decision and reason again")
OWN_REVIEW_REFUSED_MESSAGE = ("The person who opened a review may not decide it; another Supply "
                              "Planning Manager must, as the Supply Allocation Policy requires")

AGENT_NAME_ALLOCATION_ASSISTANT = "allocation_assistant"

CHAT_TYPE_POLICY_RULES = "POLICY RULES"
CHAT_TYPE_ALLOCATION_RULES = "ALLOCATION RULES"
CHAT_TYPE_REQUEST_HANDLING = "REQUEST HANDLING"
CHAT_TYPE_ALLOCATION_STATUS = "ALLOCATION STATUS"
CHAT_TYPE_OUT_OF_SCOPE = "OUT OF SCOPE"
CHAT_TYPE_OPTIONS = [CHAT_TYPE_POLICY_RULES, CHAT_TYPE_ALLOCATION_RULES, CHAT_TYPE_REQUEST_HANDLING,
                     CHAT_TYPE_ALLOCATION_STATUS, CHAT_TYPE_OUT_OF_SCOPE]
CHAT_ANSWERED_TYPES = [CHAT_TYPE_POLICY_RULES, CHAT_TYPE_ALLOCATION_RULES,
                       CHAT_TYPE_REQUEST_HANDLING]

CHAT_QUESTION_MINIMUM_CHARACTERS = 3
CHAT_EXCLUDED_DOCUMENTS = ["Past-Shortages-Casebook.pdf"]
CHAT_RETRIEVAL_RESULT_COUNT = 4
CHAT_MAX_RELEVANCE_DISTANCE = 0.65
CHAT_HISTORY_TURN_LIMIT = 3
CHAT_PAGE_TURN_LIMIT = 20
CHAT_AUDIT_MAX_ROWS = 200
CHAT_NO_VALUE_REPLY = "none"
CHAT_STATUS_ANSWER_IN_HISTORY = ("(answered from the allocation records, which are not repeated "
                                 "here)")

DELIVERY_POSTCODE_COLUMN = "delivery_postcode"
POSTCODE_PATTERN = r"\b[A-Za-z]{1,2}[0-9][A-Za-z0-9]? ?[0-9][A-Za-z]{2}\b"
CHAT_POSTCODE_WITHHELD_TEXT = "[postcode withheld]"
PRODUCT_CODE_PATTERN = r"^[A-Z]{2}-[0-9]{4}$"

CHAT_REFUSAL_MESSAGE = ("I can only answer from the Supply Allocation Policy, the category "
                        "allocation policies, the Allocation Rules and the request guides of "
                        f"{ORGANISATION_NAME}, and I could not find anything in them that "
                        "answers that.")
CHAT_NO_ALLOCATION_TEMPLATE = "No allocation can be confirmed for account {account}."
CHAT_ACCOUNT_NOT_NAMED_MESSAGE = ("Give the account's code, written as any of the three order "
                                  "routes writes it, for example CUS-0021, C0021 or Acct 21, and "
                                  "the delivery postcode on the account's record.")
CHAT_SUBJECT_NOT_NAMED_MESSAGE = ("Name the product by its code, for example HS-0500, or the "
                                  "account by its code, written as any of the three order routes "
                                  "writes it, for example CUS-0021, C0021 or Acct 21.")
CHAT_ACCOUNT_HEADING_TEMPLATE = "Account {account_code}, {customer_name}:"
CHAT_ORDER_UNDER_REVIEW_TEMPLATE = ("{product_code} {product_name}: under review, "
                                    "{cases_ordered:,} cases ordered; no case is confirmed until "
                                    "a Supply Planning Manager decides.")
CHAT_ORDER_RELEASED_TEMPLATE = ("{product_code} {product_name}: released, {cases_confirmed:,} of "
                                "the {cases_ordered:,} cases ordered confirmed.")
CHAT_ORDER_ESCALATED_TEMPLATE = ("{product_code} {product_name}: escalated to the supply review; "
                                 "the {cases_ordered:,} cases ordered stay open and no case is "
                                 "confirmed this week.")
CHAT_OTHER_ORDERS_TEXT = ("Any other open order of the account on allocation waits for a "
                          "decision, and no case of it is confirmed yet.")
CHAT_SUBJECT_PRODUCT_TEMPLATE = "product {product_code}"
CHAT_SUBJECT_ACCOUNT_TEMPLATE = "account {account_code}"
CHAT_NO_REVIEW_TEMPLATE = "No review of {subject} has been recorded."
CHAT_REVIEWS_HEADING_TEMPLATE = "Reviews of {subject}, newest first:"
CHAT_REVIEW_TEMPLATE = ("{review_id}, {product_code} {product_name}, divided under "
                        "{policy_document} and opened by {planner_id}: {status}; "
                        "{cases_available:,} cases available and customers may have "
                        "{cases_eligible:,}.")
CHAT_AWAITING_DECISION_TEMPLATE = ("It waits for a Supply Planning Manager's decision; the "
                                   "decisions open are {decisions_open} and the review recommends "
                                   "{recommended_decision}.")
CHAT_DECIDED_TEMPLATE = ('{decided_by}, {decided_by_role}, decided {decision}, giving the reason '
                         '"{decision_reason}"; {reserve_cases:,} cases held in reserve and '
                         '{cases_in_stock:,} left in stock.')
CHAT_REVIEW_LINE_TEMPLATE = ("{customer_name} ({account_code}, {channel}): ordered "
                             "{cases_ordered:,}, may have {eligible_cases:,}; {cases_text}.")
CHAT_CONFIRMED_CASES_TEMPLATE = "confirmed {cases_confirmed:,}"
CHAT_NO_RELEASE_OPEN_TEXT = "no release is open"

API_HOST = "0.0.0.0"
API_PORT = 8081
API_TITLE = "FairShare AI API"  

import os
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    f"http://localhost:{API_PORT}"
)

API_REQUEST_TIMEOUT_SECONDS = 300

DASHBOARD_TITLE = "FairShare AI"
DASHBOARD_SUBTITLE = f"{ORGANISATION_NAME} | supply allocation desk"
INFLATED_ORDER_BAND_BADGE_COLOURS = {INFLATED_ORDER_BAND_HIGH: "red",
                                     INFLATED_ORDER_BAND_MEDIUM: "orange",
                                     INFLATED_ORDER_BAND_LOW: "gray"}
DECISION_BADGE_COLOURS = {DECISION_RELEASE: "green", DECISION_RELEASE_WITH_RESERVE: "blue",
                          DECISION_ESCALATE: "orange"}
STATUS_BADGE_COLOURS = {REVIEW_STATUS_PENDING: "orange", REVIEW_STATUS_RELEASED: "green",
                        REVIEW_STATUS_RELEASED_WITH_RESERVE: "blue",
                        REVIEW_STATUS_ESCALATED: "violet"}
CHAT_TYPE_BADGE_COLOURS = {CHAT_TYPE_POLICY_RULES: "blue", CHAT_TYPE_ALLOCATION_RULES: "violet",
                           CHAT_TYPE_REQUEST_HANDLING: "green",
                           CHAT_TYPE_ALLOCATION_STATUS: "orange", CHAT_TYPE_OUT_OF_SCOPE: "gray"}
DIVISION_FILL_COLOURS = {"available": "#e6ecf7", "protected": "#dcefe2", "reserve": "#f8ecd4",
                         "shared": "#e3e9f6", "stock": "#eceef1"}
DIVISION_LINE_COLOURS = {"available": "#1e3a8a", "protected": "#15803d", "reserve": "#b45309",
                         "shared": "#1d4ed8", "stock": "#64748b"}
DIVISION_TEXT_COLOUR = "#0f172a"
DIVISION_EDGE_COLOUR = "#64748b"
ALLOCATION_CHART_COLOURS = ["#1d4ed8", "#cbd5e1"]
ALLOCATION_CHART_SERIES = ["Gets", "Short of what it may have"]
COVER_NOT_WORKED_OUT_TEXT = "Not worked out"
