import pytest


@pytest.fixture(scope="session")
def processed_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.CLEAN_ORDER_HISTORY_TRAIN_FILE,
                          config.CLEAN_ORDER_HISTORY_EVAL_FILE,
                          config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE,
                          config.CLEAN_WEEKLY_DEMAND_EVAL_FILE,
                          config.CLEAN_CUSTOMER_EMAILS_FILE]:
            loaded[file_path] = pd.read_csv(file_path, keep_default_na=False, na_values=[""],
                                            low_memory=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The cleaned extracts could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on cleaning, features and models reads the files the data "
            "pipeline writes, so the pipeline has to have run and written all five."
        )
    return loaded


@pytest.fixture(scope="session")
def raw_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.RAW_ORDER_HISTORY_TRAIN_FILE, config.RAW_ORDER_HISTORY_EVAL_FILE,
                          config.RAW_WEEKLY_DEMAND_TRAIN_FILE, config.RAW_WEEKLY_DEMAND_EVAL_FILE,
                          config.RAW_CUSTOMER_EMAILS_FILE]:
            loaded[file_path] = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The raw extracts supplied with the project could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The data folder has to stay as it was supplied."
        )
    return loaded


@pytest.fixture(scope="session")
def saved_models():
    from config import config

    loaded = {"problem": None}
    try:
        from modeling.model_registry import load_feature_schema, load_model

        loaded.update({
            "inflated_order": load_model(config.INFLATED_ORDER_MODEL_FILE),
            "demand": load_model(config.DEMAND_MODEL_FILE),
            "request": load_model(config.REQUEST_MODEL_FILE),
            "inflated_order_schema": list(load_feature_schema(config.INFLATED_ORDER_SCHEMA_FILE)),
            "demand_schema": list(load_feature_schema(config.DEMAND_SCHEMA_FILE)),
        })
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The saved models and their feature schemas could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Model training saves them, and every check on the models scores what "
            "training saved."
        )
    return loaded


@pytest.fixture(scope="session")
def live_files():
    from config import config

    loaded = {"problem": None}
    try:
        from data_pipeline.dataset_loader import load_raw_dataset

        loaded.update({
            "products": load_raw_dataset(config.SHORTAGE_PRODUCTS_FILE),
            "orders": load_raw_dataset(config.OPEN_ORDERS_FILE),
            "shipments": load_raw_dataset(config.SHIPMENT_HISTORY_FILE),
            "customers": load_raw_dataset(config.CUSTOMER_MASTER_FILE),
            "emails": load_raw_dataset(config.OPEN_CUSTOMER_EMAILS_FILE),
        })
        for name, column_name in [("products", config.PRODUCT_CODE_COLUMN),
                                  ("orders", config.ORDER_LINE_ID_COLUMN),
                                  ("shipments", config.SHIPMENT_REFERENCE_COLUMN),
                                  ("customers", config.DELIVERY_POSTCODE_COLUMN),
                                  ("emails", config.EMAIL_ID_COLUMN)]:
            assert column_name in getattr(loaded[name], "columns", []), (
                "load_raw_dataset returned "
                + type(loaded[name]).__name__
                + " for the live "
                + name
                + ", where it returns a table holding the column "
                + column_name
                + "."
            )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The live files could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The allocation, the queue and the workflow all start from the products on "
            "allocation, their open orders, the shipment history, the customer master and the "
            "customers' open emails, read as text."
        )
    return loaded


@pytest.fixture(scope="session")
def knowledge_collection():
    from config import config

    opened = {"problem": None}
    try:
        from knowledge.vector_index import open_vector_index

        opened["collection"] = open_vector_index(config.VECTOR_STORE_DIRECTORY)
    except (Exception, SystemExit) as error:
        opened["problem"] = (
            "The vector store could not be opened ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Knowledge indexing builds it, and every check on retrieval, on the "
            "policies and on the worked example reads it."
        )
    return opened


@pytest.fixture(scope="session")
def document_texts():
    from config import config

    read = {"problem": None}
    try:
        from knowledge.pdf_extraction import extract_text_from_directory

        read["documents"] = extract_text_from_directory(config.DOCS_DIRECTORY)
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "The document folder could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the text of the sixteen "
            "documents."
        )
    return read


@pytest.fixture(scope="session")
def document_passages(document_texts):
    cut = {"problem": document_texts["problem"]}
    if cut["problem"] is not None:
        return cut
    try:
        from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

        cut["documents"] = document_texts["documents"]
        cut["sections"] = split_documents_into_sections(cut["documents"])
        cut["chunks"] = build_chunk_records(cut["sections"])
    except (Exception, SystemExit) as error:
        cut["problem"] = (
            "The documents could not be split into sections and cut into passages ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the passages the sixteen "
            "documents are cut into."
        )
    return cut


@pytest.fixture(scope="session")
def planted_policies(tmp_path_factory):
    from config import config

    documents = config.CATEGORY_POLICY_DOCUMENTS
    history = "Usual demand is the average weekly cases shipped over the last {weeks} weeks."
    cap = "An order counts for no more than {cap} times the customer's usual demand."
    protected = "Customers in the {channels} are protected and are served first."
    no_protection = "No customer is protected under this policy."
    reserve = ("Up to {percent} percent of the cases left after protected customers may be held "
               "back as a reserve.")
    no_reserve = "No reserve is held under this policy."
    opening = ("Purpose and scope. This policy sets out how Wrenfield Hygiene divides the cases of "
               "a product in this category among its customers while the product is short. ")
    escalation = ("Who decides. A Supply Planner reviews the shortage and a Supply Planning "
                  "Manager, who did not open the review, decides how the cases are released. ")
    closing = ("Questions about this policy. The supply planning team answers them, and the "
               "Supply Planner applies the policy as written. ")
    rules = {
        "Hand hygiene": {"weeks": "11", "cap": "1.25", "channels": None, "percent": "20"},
        "Surface disinfection": {"weeks": "6", "cap": "1.4", "channels": "Pharmacy channel",
                                 "percent": None},
        "Laundry": {"weeks": "9", "cap": "1.5", "channels": "Healthcare and Online channels",
                    "percent": "5"},
        "Dishwash": {"weeks": "4", "cap": "2.0", "channels": "Wholesale channel", "percent": "30"},
        "Toilet care": {"weeks": "20", "cap": "1.1",
                        "channels": "Facilities and Convenience channels", "percent": "8"},
    }
    sentences = {}
    for category, rule in rules.items():
        sentences[category] = {
            "history": history.format(weeks=rule["weeks"]),
            "cap": cap.format(cap=rule["cap"]),
            "protected": (protected.format(channels=rule["channels"])
                          if rule["channels"] else no_protection),
            "reserve": reserve.format(percent=rule["percent"]) if rule["percent"] else no_reserve,
        }
    passages = {
        "Hand hygiene": [
            opening + sentences["Hand hygiene"]["history"],
            "What a customer may have. " + sentences["Hand hygiene"]["cap"] + " "
            + sentences["Hand hygiene"]["protected"],
            escalation + sentences["Hand hygiene"]["reserve"],
            closing,
        ],
        "Surface disinfection": [
            opening + "Protected customers. " + sentences["Surface disinfection"]["protected"],
            "The reserve. " + sentences["Surface disinfection"]["reserve"] + " " + escalation,
            "Usual demand. " + sentences["Surface disinfection"]["history"] + " "
            + sentences["Surface disinfection"]["cap"],
            closing,
        ],
        "Laundry": [
            opening + sentences["Laundry"]["reserve"],
            "Protected customers. " + sentences["Laundry"]["protected"],
            escalation + sentences["Laundry"]["cap"],
            closing + sentences["Laundry"]["history"],
        ],
        "Dishwash": [
            opening + sentences["Dishwash"]["cap"],
            "Usual demand. " + sentences["Dishwash"]["history"],
            "The reserve and protected customers. " + sentences["Dishwash"]["reserve"] + " "
            + sentences["Dishwash"]["protected"] + " " + escalation,
            closing,
        ],
        "Toilet care": [
            opening + escalation,
            "The four rules. " + sentences["Toilet care"]["protected"] + " "
            + sentences["Toilet care"]["history"],
            sentences["Toilet care"]["reserve"] + " " + sentences["Toilet care"]["cap"],
            closing,
        ],
    }
    decoy = ("Superseded rules. This notice keeps rules that no longer apply to any category. "
             + history.format(weeks="52") + " " + cap.format(cap="3.0") + " "
             + protected.format(channels="Supermarket channel") + " "
             + reserve.format(percent="40"))
    contradictory = {
        "Hand hygiene": [
            opening + history.format(weeks="11") + " " + cap.format(cap="1.25"),
            "Usual demand again. " + history.format(weeks="12"),
            no_protection + " " + reserve.format(percent="20"),
        ],
        "Surface disinfection": [
            opening + history.format(weeks="6") + " " + cap.format(cap="1.4"),
            protected.format(channels="Pharmacy channel") + " " + no_reserve,
            "A second reading. " + cap.format(cap="1.6"),
        ],
        "Laundry": [
            opening + history.format(weeks="9") + " " + cap.format(cap="1.5"),
            protected.format(channels="Healthcare and Online channels") + " "
            + reserve.format(percent="5"),
            closing + no_protection,
        ],
        "Dishwash": [
            opening + history.format(weeks="4") + " " + cap.format(cap="2.0"),
            protected.format(channels="Wholesale channel") + " " + reserve.format(percent="30"),
            "Protected customers again. " + protected.format(channels="Wholesale and Pharmacy "
                                                                      "channels"),
        ],
        "Toilet care": [
            opening + history.format(weeks="20") + " " + cap.format(cap="1.1"),
            protected.format(channels="Facilities and Convenience channels") + " "
            + reserve.format(percent="8"),
            escalation + no_reserve,
        ],
    }
    built = {
        "problem": None,
        "decoy": {"history_weeks": 52, "cap": "3.0", "protected_channels": ["Supermarket"],
                  "reserve_percent": 40},
        "expected": {
            "Hand hygiene": {"history_weeks": 11, "cap": "1.25", "protected_channels": [],
                             "reserve_percent": 20},
            "Surface disinfection": {"history_weeks": 6, "cap": "1.4",
                                     "protected_channels": ["Pharmacy"], "reserve_percent": None},
            "Laundry": {"history_weeks": 9, "cap": "1.5",
                        "protected_channels": ["Healthcare", "Online"], "reserve_percent": 5},
            "Dishwash": {"history_weeks": 4, "cap": "2.0", "protected_channels": ["Wholesale"],
                         "reserve_percent": 30},
            "Toilet care": {"history_weeks": 20, "cap": "1.1",
                            "protected_channels": ["Facilities", "Convenience"],
                            "reserve_percent": 8},
        },
        "sentences": sentences,
        "contradictions": {
            "Hand hygiene": "two different weeks of history, 11 and 12",
            "Surface disinfection": "two different caps on an order, 1.4 and 1.6",
            "Laundry": "protected channels and that no customer is protected",
            "Dishwash": "two different lists of protected channels",
            "Toilet care": "a reserve and that no reserve is held",
        },
    }
    try:
        from knowledge.vector_index import build_vector_index, open_vector_index

        for store_name, written in [("valid", passages), ("contradictory", contradictory)]:
            chunk_records = []
            for category, texts in written.items():
                for position, text in enumerate(texts):
                    chunk_records.append({"chunk_id": documents[category] + "-" + str(position),
                                          "chunk_text": text,
                                          "document_name": documents[category]})
            chunk_records.append({"chunk_id": "Superseded-Rules-Notice.pdf-0",
                                  "chunk_text": decoy,
                                  "document_name": "Superseded-Rules-Notice.pdf"})
            store_path = str(tmp_path_factory.mktemp("planted_" + store_name) / "store")
            build_vector_index(chunk_records, store_path)
            built[store_name] = open_vector_index(store_path)
    except (Exception, SystemExit) as error:
        built["problem"] = (
            "A store of planted policies could not be built with build_vector_index and opened "
            "with open_vector_index ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The checks on reading the policies read rules the suite invented, so a "
            "solution that remembered the real ones is not rewarded for it."
        )
    return built


@pytest.fixture(scope="session")
def planted_shortages(live_files, tmp_path_factory):
    from fractions import Fraction

    import pandas as pd

    from config import config

    built = {"problem": live_files["problem"], "shortages": {}}
    if built["problem"] is not None:
        return built
    shortages = {
        "every notation and an order sent twice": {
            "product_code": "PN-0500", "pack": "12 x 500 ml", "cases": 60,
            "rules": [4, "1.7", [], None],
            "orders": [("PO-900001", "EDI", "C0021", 1, "1200 units"),
                       ("PO-900002", "Portal", "CUS-0034", 2, "60 cs"),
                       ("PO-900003", "Email", "Acct 62", 3, "2 plt"),
                       ("PO-900004", "Email", "Acct 46", 4, "15 cs"),
                       ("PO-900004", "EDI", "C0046", 9, "240 units"),
                       ("PO-900006", "EDI", "C0021", 5, "600 units")],
            "shipments": {"CUS-0021": {1: 25, 2: 25, 3: 25, 4: 25, 5: 90},
                          "CUS-0034": {1: 10, 2: 10, 3: 10, 4: 10},
                          "CUS-0062": {1: 20, 2: 20, 3: 20, 4: 20},
                          "CUS-0046": {1: 20, 2: 20, 3: 20, 4: 20}},
            "usual": {"CUS-0021": Fraction(25), "CUS-0034": Fraction(10),
                      "CUS-0046": Fraction(20), "CUS-0062": Fraction(20)},
        },
        "every other week and 55 cases over seven weeks at 1.4": {
            "product_code": "PW-0500", "pack": "12x500ml", "cases": 193,
            "rules": [7, "1.4", [], None],
            "orders": [("PO-900011", "Portal", "CUS-0046", 1, "50 cs"),
                       ("PO-900012", "Portal", "CUS-0131", 2, "80 cs"),
                       ("PO-900013", "EDI", "C0021", 3, "3600 units")],
            "shipments": {"CUS-0046": {2: 70, 4: 70, 6: 70, 8: 70},
                          "CUS-0131": {1: 7, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8},
                          "CUS-0021": {1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100,
                                       7: 100}},
            "usual": {"CUS-0021": Fraction(100), "CUS-0046": Fraction(30),
                      "CUS-0131": Fraction(55, 7)},
        },
        "a protected group too large for the cases": {
            "product_code": "PP-0500", "pack": "12x500ml", "cases": 120,
            "rules": [4, "1.0", ["Healthcare"], 10],
            "orders": [("PO-900021", "Portal", "CUS-0105", 1, "150 cs"),
                       ("PO-900022", "Portal", "CUS-0118", 2, "80 cs"),
                       ("PO-900023", "EDI", "C0021", 3, "480 units")],
            "shipments": {"CUS-0105": {1: 100, 2: 100, 3: 100, 4: 100},
                          "CUS-0118": {1: 50, 2: 50, 3: 50, 4: 50},
                          "CUS-0021": {1: 40, 2: 40, 3: 40, 4: 40}},
            "usual": {"CUS-0021": Fraction(40), "CUS-0105": Fraction(100),
                      "CUS-0118": Fraction(50)},
        },
        "three customers capped over two passes": {
            "product_code": "PC-0500", "pack": "12x500ml", "cases": 300,
            "rules": [4, "1.0", [], None],
            "orders": [("PO-900031", "Portal", "CUS-0001", 1, "50 cs"),
                       ("PO-900032", "Portal", "CUS-0003", 2, "40 cs"),
                       ("PO-900033", "Portal", "CUS-0021", 3, "65 cs"),
                       ("PO-900034", "Portal", "CUS-0034", 4, "300 cs")],
            "shipments": {"CUS-0001": {1: 100, 2: 100, 3: 100, 4: 100},
                          "CUS-0003": {1: 100, 2: 100, 3: 100, 4: 100},
                          "CUS-0021": {1: 100, 2: 100, 3: 100, 4: 100},
                          "CUS-0034": {1: 200, 2: 200, 3: 200, 4: 200}},
            "usual": {"CUS-0001": Fraction(100), "CUS-0003": Fraction(100),
                      "CUS-0021": Fraction(100), "CUS-0034": Fraction(200)},
        },
        "a tie in the remainders settled by usual demand": {
            "product_code": "PT-0500", "pack": "12x500ml", "cases": 10,
            "rules": [2, "5.0", [], None],
            "orders": [("PO-900041", "Portal", "CUS-0001", 1, "50 cs"),
                       ("PO-900042", "Portal", "CUS-0003", 2, "50 cs")],
            "shipments": {"CUS-0001": {1: 1, 2: 1}, "CUS-0003": {1: 3, 2: 3}},
            "usual": {"CUS-0001": Fraction(1), "CUS-0003": Fraction(3)},
        },
        "a tie in the remainders settled by account code": {
            "product_code": "PU-0500", "pack": "12x500ml", "cases": 11,
            "rules": [2, "5.0", [], None],
            "orders": [("PO-900051", "Portal", "CUS-0003", 1, "50 cs"),
                       ("PO-900052", "Portal", "CUS-0001", 2, "50 cs")],
            "shipments": {"CUS-0001": {1: 2, 2: 2}, "CUS-0003": {1: 2, 2: 2}},
            "usual": {"CUS-0001": Fraction(2), "CUS-0003": Fraction(2)},
        },
        "a reserve of 8.5 cases": {
            "product_code": "PR-0500", "pack": "12x500ml", "cases": 105,
            "rules": [4, "1.0", ["Healthcare"], 10],
            "orders": [("PO-900061", "Portal", "CUS-0105", 1, "30 cs"),
                       ("PO-900062", "Portal", "CUS-0021", 2, "100 cs"),
                       ("PO-900063", "Portal", "CUS-0034", 3, "60 cs")],
            "shipments": {"CUS-0105": {1: 20, 2: 20, 3: 20, 4: 20},
                          "CUS-0021": {1: 50, 2: 50, 3: 50, 4: 50},
                          "CUS-0034": {1: 30, 2: 30, 3: 30, 4: 30}},
            "usual": {"CUS-0021": Fraction(50), "CUS-0034": Fraction(30),
                      "CUS-0105": Fraction(20)},
        },
        "a cover of 49 percent": {
            "product_code": "PH-0500", "pack": "12x500ml", "cases": 49,
            "rules": [4, "1.0", [], None],
            "orders": [("PO-900071", "Portal", "CUS-0021", 1, "60 cs"),
                       ("PO-900072", "Portal", "CUS-0034", 2, "40 cs")],
            "shipments": {"CUS-0021": {1: 60, 2: 60, 3: 60, 4: 60},
                          "CUS-0034": {1: 40, 2: 40, 3: 40, 4: 40}},
            "usual": {"CUS-0021": Fraction(60), "CUS-0034": Fraction(40)},
        },
        "cases left in stock beside a customer with no usual demand": {
            "product_code": "PS-0500", "pack": "12x500ml", "cases": 100,
            "rules": [5, "1.3", [], None],
            "orders": [("PO-900081", "Portal", "CUS-0021", 1, "40 cs"),
                       ("PO-900082", "Portal", "CUS-0034", 2, "30 cs")],
            "shipments": {"CUS-0021": {1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50},
                          "CUS-0034": {7: 30, 8: 30, 9: 30}},
            "usual": {"CUS-0021": Fraction(50), "CUS-0034": Fraction(0)},
        },
        "no customer may have any case": {
            "product_code": "PZ-0500", "pack": "12x500ml", "cases": 50,
            "rules": [5, "1.3", ["Healthcare"], 10],
            "orders": [("PO-900091", "Portal", "CUS-0021", 1, "40 cs"),
                       ("PO-900092", "Portal", "CUS-0105", 2, "20 cs")],
            "shipments": {"CUS-0021": {9: 40}, "CUS-0105": {7: 20, 12: 20}},
            "usual": {"CUS-0021": Fraction(0), "CUS-0105": Fraction(0)},
        },
    }
    template = live_files["orders"].iloc[0].to_dict()
    order_rows = []
    shipment_rows = []
    for position, shortage in enumerate(shortages.values()):
        for number, (reference, route, account, sequence, quantity) in enumerate(
                shortage["orders"], start=1):
            order_row = dict(template)
            order_row.update({
                config.ORDER_LINE_ID_COLUMN: "OL-" + shortage["product_code"] + "-" + str(number),
                config.ORDER_REFERENCE_COLUMN: reference, config.ORDER_ROUTE_COLUMN: route,
                config.ACCOUNT_CODE_COLUMN: account,
                config.RECEIVED_SEQUENCE_COLUMN: str(900000 + 1000 * position + sequence),
                config.PRODUCT_CODE_COLUMN: shortage["product_code"],
                config.PACK_DESCRIPTION_COLUMN: shortage["pack"],
                config.CASES_PER_PALLET_COLUMN: "50", config.ORDERED_QUANTITY_COLUMN: quantity})
            order_rows.append(order_row)
        for account_code, weeks in shortage["shipments"].items():
            for weeks_ago, cases_shipped in weeks.items():
                for product_code, shipped in [(shortage["product_code"], cases_shipped),
                                              ("X" + shortage["product_code"][1:], 999)]:
                    shipment_rows.append({
                        config.SHIPMENT_REFERENCE_COLUMN: "SHP-" + str(
                            9800000 + len(shipment_rows)),
                        config.ACCOUNT_CODE_COLUMN: account_code,
                        config.PRODUCT_CODE_COLUMN: product_code,
                        config.WEEKS_AGO_COLUMN: str(weeks_ago),
                        config.CASES_SHIPPED_COLUMN: str(shipped)})
    shortages["the worked example"] = {
        "product_code": "HS-0500", "cases": 1200, "rules": [8, "1.0", ["Healthcare"], 10],
        "usual": {"CUS-0021": Fraction(400), "CUS-0034": Fraction(300), "CUS-0046": Fraction(30),
                  "CUS-0062": Fraction(120), "CUS-0077": Fraction(100), "CUS-0089": Fraction(80),
                  "CUS-0105": Fraction(250), "CUS-0118": Fraction(150),
                  "CUS-0131": Fraction(90)},
    }
    try:
        from data_pipeline.dataset_loader import load_raw_dataset

        orders_path = str(tmp_path_factory.mktemp("planted_shortages") / "open_orders.csv")
        shipments_path = orders_path.replace("open_orders.csv", "shipment_history.csv")
        pd.DataFrame(order_rows).to_csv(orders_path, index=False)
        pd.DataFrame(shipment_rows).to_csv(shipments_path, index=False)
        planted_orders = load_raw_dataset(orders_path)
        planted_history = load_raw_dataset(shipments_path)
    except (Exception, SystemExit) as error:
        built["problem"] = (
            "The planted open orders and shipments could not be written and read back with "
            "load_raw_dataset ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + ")."
        )
        return built
    for label, shortage in shortages.items():
        weeks, cap, protected_channels, reserve_percent = shortage["rules"]
        rules = {
            "document_name": "Invented-Policy.pdf", "history_weeks": weeks,
            "cap": Fraction(cap), "protected_channels": list(protected_channels),
            "reserve_percent": reserve_percent,
            "rule_sentences": {
                "history": ("Usual demand is the average weekly cases shipped over the last "
                            + str(weeks) + " weeks."),
                "cap": ("An order counts for no more than " + cap
                        + " times the customer's usual demand."),
                "protected": ("Customers in the " + " and ".join(protected_channels)
                              + " channel are protected and are served first."
                              if protected_channels
                              else "No customer is protected under this policy."),
                "reserve": ("Up to " + str(reserve_percent) + " percent of the cases left after "
                            "protected customers may be held back as a reserve."
                            if reserve_percent is not None
                            else "No reserve is held under this policy."),
            },
        }
        result = dict(shortage, rules=rules, reading_problem=None, usual_problem=None,
                      division_problem=None)
        live = label == "the worked example"
        try:
            from tools.shortage_data_tools import (fetch_customer_accounts, fetch_open_orders,
                                                   fetch_shipment_history)

            result["read_orders"] = fetch_open_orders(
                live_files["orders"] if live else planted_orders, shortage["product_code"])
            result["accounts"] = sorted(set(result["read_orders"][config.ACCOUNT_CODE_COLUMN]))
            result["history"] = fetch_shipment_history(
                live_files["shipments"] if live else planted_history, shortage["product_code"])
            result["customer_accounts"] = fetch_customer_accounts(live_files["customers"],
                                                                  result["accounts"])
        except (Exception, SystemExit) as error:
            result["reading_problem"] = (
                "The orders, shipments and customers of the planted shortage '"
                + label
                + "' could not be read with fetch_open_orders, fetch_shipment_history and "
                "fetch_customer_accounts ("
                + type(error).__name__
                + ": "
                + str(error)[:300]
                + ")."
            )
            built["shortages"][label] = result
            continue
        try:
            from tools.allocation_tools import compute_usual_demand

            result["computed_usual"] = compute_usual_demand(result["history"], result["accounts"],
                                                            weeks)
        except (Exception, SystemExit) as error:
            result["usual_problem"] = (
                "compute_usual_demand failed on the planted shortage '"
                + label
                + "' ("
                + type(error).__name__
                + ": "
                + str(error)[:300]
                + ")."
            )
        try:
            from tools.allocation_tools import allocate_cases, summarise_allocation

            result["allocation"] = allocate_cases(shortage["cases"], result["read_orders"],
                                                  result["customer_accounts"], shortage["usual"],
                                                  rules)
            result["lines"] = {allocation_line["account_code"]: allocation_line
                               for allocation_line in result["allocation"]["allocation_lines"]}
            result["summary"] = summarise_allocation(result["allocation"])
        except (Exception, SystemExit) as error:
            result["division_problem"] = (
                "The planted shortage '"
                + label
                + "' could not be divided with allocate_cases, given its orders as "
                "fetch_open_orders reads them, its usual demand worked out exactly and its "
                "rules as read_allocation_policy returns them, and summarised with "
                "summarise_allocation ("
                + type(error).__name__
                + ": "
                + str(error)[:300]
                + ")."
            )
        built["shortages"][label] = result
    return built


@pytest.fixture(scope="session")
def service_client():
    service = {"problem": None}
    try:
        from fastapi.testclient import TestClient

        from api.routes import service_application

        service["client"] = TestClient(service_application, raise_server_exceptions=False)
    except (Exception, SystemExit) as error:
        service["problem"] = (
            "The REST service could not be started ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on sign-in, access, the queue, the workflow and the chat goes "
            "through it."
        )
    return service


@pytest.fixture(scope="session")
def staff(service_client):
    from config import config

    signed_in = {"problem": service_client["problem"]}
    if signed_in["problem"] is not None:
        return signed_in
    try:
        managers = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_MANAGER]
        planners = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_PLANNER]
        advisers = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_ADVISER]
        for staff_key, account in [("manager", managers[0]), ("second_manager", managers[-1]),
                                   ("planner", planners[0]), ("adviser", advisers[0])]:
            reply = service_client["client"].post(
                "/auth/signin",
                json={"username": account["username"], "password": account["password"]})
            assert reply.status_code == 200, (
                "Signing in as "
                + account["username"]
                + " with the password the configuration seeds returned "
                + str(reply.status_code)
                + ", so no check that needs a signed-in "
                + account["role"]
                + " can run."
            )
            signed_in[staff_key] = dict(reply.json(), headers={
                config.SESSION_TOKEN_HEADER: reply.json()["session_token"]})
    except (Exception, SystemExit) as error:
        signed_in["problem"] = (
            "The seeded staff could not be signed in ("
            + type(error).__name__
            + ": "
            + str(error)[:400]
            + "). The accounts are the ones the configuration seeds through the database "
            "setup, and a sign-in answers with the user's display name, role and a session "
            "token."
        )
    return signed_in


@pytest.fixture(scope="session")
def saved_run(service_client, staff):
    from config import config

    run = {"problem": service_client["problem"] or staff["problem"], "decision_problem": None}
    if run["problem"] is not None:
        return run
    client = service_client["client"]
    planner = staff["planner"]
    opener = staff["manager"]
    decider = staff["second_manager"]
    run["planner_request"] = ("Who may have what of this product this week, and can the cases go "
                              "out as divided?")
    run["decision"] = config.DECISION_RELEASE
    run["decision_reason"] = ("Every protected customer is served in full and the cover is above "
                              "half, so the cases go out as divided.")
    run["opener"] = opener
    run["decider"] = decider
    for name in ["detail", "calls_at_gate", "pending_at_gate", "final_calls"]:
        run[name] = {"status": None, "body": "never requested"}
    run["own_decision"] = {"status": None, "body": "never requested", "still_pending": None,
                           "calls_before": None, "calls_after": None}
    run["closed_decision"] = {"status": None, "body": "never requested", "calls_before": None,
                              "calls_after": None}
    run["run_record"] = {"problem": "never read"}
    run["heard_at_gate"] = {"problem": "never asked"}
    run["heard_after"] = {"problem": "never asked"}
    try:
        import pandas as pd

        queue = client.get("/queue", params={"product_category": config.QUEUE_CATEGORY_ALL},
                           headers=planner["headers"])
        assert queue.status_code == 200 and isinstance(queue.json(), list), (
            "The shortage queue returned "
            + str(queue.status_code)
            + " with "
            + queue.text[:300]
            + ", so no product could be chosen to review."
        )
        reviewed = client.get("/reviews", params={"status_value": config.REVIEW_STATUS_OPTIONS},
                              headers=planner["headers"])
        assert reviewed.status_code == 200, (
            "Listing the allocation reviews returned "
            + str(reviewed.status_code)
            + " with "
            + reviewed.text[:300]
            + ", so a product nobody has reviewed yet could not be chosen."
        )
        already_reviewed = {record["product_code"] for record in reviewed.json()}
        candidates = []
        for queue_row in queue.json():
            if (config.DECISION_RELEASE in queue_row["decisions_open"]
                    and config.DECISION_RELEASE_WITH_RESERVE not in queue_row["decisions_open"]
                    and queue_row["product_code"] not in already_reviewed):
                candidates.append(queue_row["product_code"])
        assert len(candidates) > 0, (
            "No product in the queue with RELEASE open under a policy that holds no reserve is "
            "left unreviewed, so the one review run cannot be opened. Start again from an empty "
            "database."
        )
        details = {}
        customer_counts = []
        for product_code in candidates:
            reply = client.get("/queue/" + product_code, headers=planner["headers"])
            assert reply.status_code == 200, (
                "The detail of "
                + product_code
                + " returned "
                + str(reply.status_code)
                + " with "
                + reply.text[:300]
                + ", so the customers of the product cannot be counted."
            )
            details[product_code] = reply.json()
            customer_counts.append((len(reply.json()["allocation_lines"]), product_code))
        run["product_code"] = sorted(customer_counts)[0][1]
        run["detail"] = {"status": 200, "body": details[run["product_code"]]}
        lines = run["detail"]["body"]["allocation_lines"]
        master = pd.read_csv(config.CUSTOMER_MASTER_FILE, dtype=str, keep_default_na=False)
        postcodes = dict(zip(master[config.ACCOUNT_CODE_COLUMN].str.strip(),
                             master[config.DELIVERY_POSTCODE_COLUMN].str.strip()))
        ranked = sorted((-line["cases_by_release"][config.DECISION_RELEASE], line["account_code"])
                        for line in lines)
        run["customer"] = ranked[0][1]
        run["other_customer"] = ranked[-1][1]
        run["postcodes"] = {account_code: postcodes[account_code]
                            for account_code in [run["customer"], run["other_customer"]]}
        assert (run["customer"] != run["other_customer"]
                and run["postcodes"][run["customer"]] != run["postcodes"][run["other_customer"]]), (
            "The product chosen, "
            + run["product_code"]
            + ", has no two customers with different delivery postcodes, so what an adviser hears "
            "with a wrong postcode cannot be told apart from what they hear with the right one."
        )
        opened = client.post("/reviews",
                             json={"product_code": run["product_code"],
                                   "planner_request": run["planner_request"]},
                             headers=opener["headers"])
        assert opened.status_code == 201, (
            "Opening a review of "
            + run["product_code"]
            + " returned "
            + str(opened.status_code)
            + " with "
            + opened.text[:400]
            + ", so the workflow never reached a Supply Planning Manager."
        )
        run["at_gate"] = opened.json()
        review_id = run["at_gate"]["review_id"]
    except (Exception, SystemExit) as error:
        run["problem"] = (
            "The one review run that the agent, workflow and record checks read could not be "
            "opened ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )
        return run

    calls_path = "/reviews/" + review_id + "/calls"
    try:
        reply = client.get(calls_path, headers=planner["headers"])
        run["calls_at_gate"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/reviews", params={"status_value": [config.REVIEW_STATUS_PENDING]},
                           headers=planner["headers"])
        run["pending_at_gate"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["calls_at_gate"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    try:
        from persistence.database import session_factory
        from persistence.models import AgentRun

        with session_factory() as session:
            run["run_record"] = {"problem": None,
                                 "status_at_gate": session.get(AgentRun,
                                                               run["at_gate"]["run_id"]).status}
    except (Exception, SystemExit) as error:
        run["run_record"] = {"problem": ("The paused run could not be read from the database ("
                                         + type(error).__name__ + ": " + str(error)[:300]
                                         + ").")}
    try:
        from chat.allocation_status_answer import answer_allocation_status
        from data_pipeline.dataset_loader import load_raw_dataset
        from persistence.database import session_factory

        customer_master = load_raw_dataset(config.CUSTOMER_MASTER_FILE)
        customer = run["customer"]
        other = run["other_customer"]
        with session_factory() as session:
            run["heard_at_gate"] = {
                "problem": None,
                "adviser": answer_allocation_status(session, staff["adviser"], customer_master,
                                                    customer, run["postcodes"][customer], None),
                "adviser_other": answer_allocation_status(session, staff["adviser"],
                                                          customer_master, other,
                                                          run["postcodes"][other], None),
                "desk": answer_allocation_status(session, planner, customer_master, None, None,
                                                 run["product_code"]),
            }
    except (Exception, SystemExit) as error:
        run["heard_at_gate"] = {"problem": ("An allocation's status could not be answered while "
                                            "the run waited ("
                                            + type(error).__name__ + ": " + str(error)[:300]
                                            + ").")}

    try:
        decision_path = "/reviews/" + review_id + "/decision"
        decision_payload = {"decision": run["decision"], "decision_reason": run["decision_reason"]}
        calls_before = client.get(calls_path, headers=planner["headers"])
        attempted = client.post(decision_path, json=decision_payload, headers=opener["headers"])
        calls_after = client.get(calls_path, headers=planner["headers"])
        pending = client.get("/reviews", params={"status_value": [config.REVIEW_STATUS_PENDING]},
                             headers=planner["headers"])
        run["own_decision"] = {
            "status": attempted.status_code,
            "body": attempted.text[:300],
            "still_pending": pending.status_code == 200 and review_id in [
                str(record.get("review_id")) for record in pending.json()],
            "calls_before": (calls_before.json().get("call_count")
                             if calls_before.status_code == 200 else None),
            "calls_after": (calls_after.json().get("call_count")
                            if calls_after.status_code == 200 else None),
        }
        closed = client.post(decision_path,
                             json={"decision": config.DECISION_RELEASE_WITH_RESERVE,
                                   "decision_reason": "Hold a reserve back for urgent orders."},
                             headers=decider["headers"])
        calls_after_closed = client.get(calls_path, headers=planner["headers"])
        run["closed_decision"] = {
            "status": closed.status_code,
            "body": closed.json() if closed.headers.get("content-type", "").startswith(
                "application/json") else closed.text[:300],
            "calls_before": run["own_decision"]["calls_after"],
            "calls_after": (calls_after_closed.json().get("call_count")
                            if calls_after_closed.status_code == 200 else None),
        }
        decided = client.post(decision_path, json=decision_payload, headers=decider["headers"])
        assert decided.status_code == 200, (
            "Recording the decision on "
            + review_id
            + " returned "
            + str(decided.status_code)
            + " with "
            + decided.text[:400]
            + ", so the paused run was never resumed."
        )
        run["final"] = decided.json()
    except (Exception, SystemExit) as error:
        run["decision_problem"] = (
            "The Supply Planning Manager's decision on the one review run could not be recorded ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )
        return run

    try:
        reply = client.get(calls_path, headers=planner["headers"])
        run["final_calls"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["final_calls"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    try:
        from persistence.database import session_factory
        from persistence.models import AgentRun

        with session_factory() as session:
            finished_run = session.get(AgentRun, run["at_gate"]["run_id"])
            run["run_record"].update({"status_final": finished_run.status,
                                      "thread_id": finished_run.thread_id})
    except (Exception, SystemExit) as error:
        run["run_record"] = {"problem": ("The finished run could not be read from the database ("
                                         + type(error).__name__ + ": " + str(error)[:300]
                                         + ").")}
    try:
        from chat.allocation_status_answer import answer_allocation_status
        from data_pipeline.dataset_loader import load_raw_dataset
        from persistence.database import session_factory

        customer_master = load_raw_dataset(config.CUSTOMER_MASTER_FILE)
        customer = run["customer"]
        other = run["other_customer"]
        written_otherwise = run["postcodes"][customer].replace(" ", "").lower()
        with session_factory() as session:
            run["heard_after"] = {
                "problem": None,
                "adviser": answer_allocation_status(session, staff["adviser"], customer_master,
                                                    customer, run["postcodes"][customer], None),
                "adviser_written_otherwise": answer_allocation_status(
                    session, staff["adviser"], customer_master, customer, written_otherwise,
                    run["product_code"]),
                "adviser_other": answer_allocation_status(session, staff["adviser"],
                                                          customer_master, other,
                                                          run["postcodes"][other], None),
                "adviser_wrong_postcode": answer_allocation_status(
                    session, staff["adviser"], customer_master, customer, run["postcodes"][other],
                    None),
                "adviser_no_postcode": answer_allocation_status(
                    session, staff["adviser"], customer_master, customer, None,
                    run["product_code"]),
                "adviser_no_account": answer_allocation_status(
                    session, staff["adviser"], customer_master, None, run["postcodes"][customer],
                    run["product_code"]),
                "adviser_nowhere": answer_allocation_status(
                    session, staff["adviser"], customer_master, "CUS-9999",
                    run["postcodes"][customer], None),
                "desk": answer_allocation_status(session, planner, customer_master, None, None,
                                                 run["product_code"]),
            }
    except (Exception, SystemExit) as error:
        run["heard_after"] = {"problem": ("An allocation's status could not be answered after the "
                                          "decision ("
                                          + type(error).__name__ + ": " + str(error)[:300]
                                          + ").")}
    return run


@pytest.fixture(scope="session")
def chat_replies(service_client, staff):
    replies = {"problem": service_client["problem"] or staff["problem"],
               "rules_question": ("Which customers are protected and served first under the hand "
                                  "hygiene policy?"),
               "rules": {"status": None, "body": "never asked"},
               "turns": {"status": None, "body": "never read"},
               "latest_turn_before": 0}
    if replies["problem"] is not None:
        return replies
    client = service_client["client"]
    planner = staff["planner"]
    replies["asker"] = planner["username"]
    replies["someone_else"] = staff["adviser"]["username"]
    try:
        earlier = client.get("/chat/turns", params={"turn_limit": 1}, headers=planner["headers"])
        if earlier.status_code == 200:
            replies["latest_turn_before"] = max([turn["turn_id"] for turn in earlier.json()]
                                                + [0])
        reply = client.post("/chat/ask", json={"question": replies["rules_question"]},
                            headers=planner["headers"])
        replies["rules"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/chat/turns", params={"turn_limit": 5}, headers=planner["headers"])
        replies["turns"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        replies["rules"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    return replies


def test_cleaning_keeps_one_row_for_every_order_line_and_an_order_sent_by_two_routes_once(
        raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.common_cleaning import convert_whole_numbers
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.order_cleaning import clean_order_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    rebuilt_path = str(tmp_path / "order_history_eval_cleaned_again.csv")
    clean_order_dataset(config.RAW_ORDER_HISTORY_EVAL_FILE, rebuilt_path)
    rebuilt = pd.read_csv(rebuilt_path, keep_default_na=False, na_values=[""], low_memory=False)
    saved = processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE]
    assert rebuilt.shape == saved.shape and rebuilt.equals(saved), (
        "Cleaning "
        + config.RAW_ORDER_HISTORY_EVAL_FILE
        + " again gives "
        + str(rebuilt.shape)
        + " rows and columns where "
        + config.CLEAN_ORDER_HISTORY_EVAL_FILE
        + " holds "
        + str(saved.shape)
        + ", or the same shape with different values. Every check reads the cleaned files, "
        "so they have to be what the cleaning code produces now: run the data pipeline "
        "again after changing it."
    )
    datasets = [
        (config.RAW_ORDER_HISTORY_TRAIN_FILE, config.CLEAN_ORDER_HISTORY_TRAIN_FILE,
         config.ORDER_LINE_ID_COLUMN, config.ORDER_TARGET_COLUMN),
        (config.RAW_ORDER_HISTORY_EVAL_FILE, config.CLEAN_ORDER_HISTORY_EVAL_FILE,
         config.ORDER_LINE_ID_COLUMN, config.ORDER_TARGET_COLUMN),
        (config.RAW_WEEKLY_DEMAND_TRAIN_FILE, config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE,
         "demand_row_id", config.DEMAND_TARGET_COLUMN),
        (config.RAW_WEEKLY_DEMAND_EVAL_FILE, config.CLEAN_WEEKLY_DEMAND_EVAL_FILE,
         "demand_row_id", config.DEMAND_TARGET_COLUMN),
        (config.RAW_CUSTOMER_EMAILS_FILE, config.CLEAN_CUSTOMER_EMAILS_FILE,
         config.EMAIL_ID_COLUMN, config.EMAIL_TARGET_COLUMN),
    ]
    for raw_path, clean_path, identifier, target in datasets:
        raw_frame = raw_frames[raw_path]
        clean_frame = processed_frames[clean_path]
        raw_identifiers = raw_frame[identifier].str.strip()
        kept_identifiers = set(clean_frame[identifier].astype(str))
        answers_path = str(tmp_path / ("answers_" + raw_path.replace("/", "_")))
        raw_frame[[identifier, target]].to_csv(answers_path, index=False)
        answers = load_raw_dataset(answers_path)
        unanswered = answers[target].isna().to_numpy()
        never_answered = set(raw_identifiers[unanswered]) - set(raw_identifiers[~unanswered])
        kept_unanswered = sorted(never_answered & kept_identifiers)
        assert len(never_answered) > 0 and len(kept_unanswered) == 0, (
            clean_path
            + " keeps "
            + str(len(kept_unanswered))
            + " records whose answer "
            + raw_path
            + " never gives, for example "
            + str(kept_unanswered[:3])
            + ". A record with no answer is removed, never given an invented one: a model "
            "fitted on invented answers learns the invention."
        )
        assert len(clean_frame) < len(raw_frame), (
            clean_path
            + " keeps "
            + str(len(clean_frame))
            + " rows of the "
            + str(len(raw_frame))
            + " in "
            + raw_path
            + ". The extract carries repeated rows and rows with no answer, so nothing "
            "has been removed that should have been."
        )
        repeated = int(clean_frame[identifier].duplicated().sum())
        assert repeated == 0, (
            clean_path
            + " still holds "
            + str(repeated)
            + " rows whose "
            + identifier
            + " appears on another row. A record counted twice is weighted twice by every "
            "model fitted on it."
        )
        empty_answers = int(clean_frame[target].isna().sum())
        assert empty_answers == 0 and clean_frame[target].nunique() > 1, (
            "The answer column "
            + target
            + " in "
            + clean_path
            + " holds "
            + str(empty_answers)
            + " empty cells and "
            + str(clean_frame[target].nunique())
            + " distinct values. Every row kept carries an answer, and a single answer leaves "
            "a model nothing to learn."
        )
        if target == config.ORDER_TARGET_COLUMN:
            found_answers = sorted(clean_frame[target].astype(str).unique())
            assert found_answers == sorted([config.BOOLEAN_NO, config.BOOLEAN_YES]), (
                target
                + " in "
                + clean_path
                + " holds "
                + str(found_answers[:6])
                + " where every spelling of yes and no is folded onto "
                + config.BOOLEAN_YES
                + " and "
                + config.BOOLEAN_NO
                + "."
            )
            arrivals = pd.DataFrame({
                "line": raw_identifiers.to_numpy(),
                "order": (raw_frame[config.ORDER_REFERENCE_COLUMN].str.strip() + " "
                          + raw_frame[config.PRODUCT_CODE_COLUMN].str.strip()).to_numpy(),
                "route": raw_frame[config.ORDER_ROUTE_COLUMN].str.strip().str.lower().to_numpy(),
                "answered": ~unanswered,
            }).drop_duplicates("line")
            routes_per_order = arrivals.groupby("order")["route"].nunique()
            sent_twice = set(routes_per_order.index[routes_per_order > 1])
            kept_orders = (clean_frame[config.ORDER_REFERENCE_COLUMN].astype(str) + " "
                           + clean_frame[config.PRODUCT_CODE_COLUMN].astype(str))
            assert len(sent_twice) > 100 and int(kept_orders.duplicated().sum()) == 0, (
                clean_path
                + " holds "
                + str(int(kept_orders.duplicated().sum()))
                + " order lines whose purchase order and product another line already has, "
                "where "
                + raw_path
                + " sends "
                + str(len(sent_twice))
                + " orders by two routes. An order sent twice is one order: counted twice, it "
                "doubles what the customer asked for."
            )
            lines_per_order = arrivals.groupby("order")["line"].nunique()
            sent_once = set(lines_per_order.index[lines_per_order == 1])
            lost = sorted(set(arrivals.loc[arrivals["order"].isin(sent_once)
                                           & arrivals["answered"], "line"]) - kept_identifiers)
            assert len(lost) == 0, (
                clean_path
                + " has lost "
                + str(len(lost))
                + " order lines that carry an answer and whose purchase order and product "
                "arrived only once, for example "
                + str(lost[:3])
                + ". Only repeated rows, rows with no answer and later arrivals of an order are "
                "removed: a gap in another column is filled when the model is fitted."
            )
        if target == config.DEMAND_TARGET_COLUMN:
            readings = convert_whole_numbers(
                answers, {target: config.DEMAND_WHOLE_NUMBER_PATTERNS[target]})[target]
            possible = readings.notna().to_numpy() & (readings.astype(float) >= 0).to_numpy()
            lost = sorted(set(raw_identifiers[possible]) - kept_identifiers)
            assert len(lost) == 0, (
                clean_path
                + " has lost "
                + str(len(lost))
                + " customer weeks whose next week's cases "
                + raw_path
                + " gives as nought or more, for example "
                + str(lost[:3])
                + ". A gap in another column is not a reason to throw a week away."
            )
            taken = pd.to_numeric(clean_frame[target])
            assert bool((taken >= 0).all()), (
                target
                + " in "
                + clean_path
                + " falls to "
                + str(taken.min())
                + ". A customer cannot take fewer than no cases, and a week whose answer is "
                "impossible goes."
            )
        if target == config.EMAIL_TARGET_COLUMN:
            found_answers = sorted(clean_frame[target].astype(str).unique())
            assert found_answers == sorted(config.REQUEST_GUIDE_DOCUMENTS), (
                target
                + " in "
                + clean_path
                + " holds "
                + str(found_answers)
                + " where every email is coded to one of the eight request types "
                + str(sorted(config.REQUEST_GUIDE_DOCUMENTS))
                + "."
            )
    eval_kept = set(processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE][
        config.ORDER_LINE_ID_COLUMN].astype(str))
    stated_pairs = [("OL-0024391", "OL-0004831"), ("OL-0023247", "OL-0009878"),
                    ("OL-0037273", "OL-0019117"), ("OL-0027813", "OL-0039719"),
                    ("OL-0027489", "OL-0008706"), ("OL-0012153", "OL-0025225")]
    wrong_pairs = [(first, later) for first, later in stated_pairs
                   if first not in eval_kept or later in eval_kept]
    assert len(wrong_pairs) == 0, (
        config.CLEAN_ORDER_HISTORY_EVAL_FILE
        + " keeps the wrong line of the orders "
        + str(wrong_pairs)
        + " (first arrival, second arrival). Each pair is one purchase order and product that "
        + config.RAW_ORDER_HISTORY_EVAL_FILE
        + " sends by two routes, both lines answered, and OL-0024391, OL-0037273 and OL-0027489 "
        "arrived first though the file writes them after their second arrival. Of an order sent "
        "twice the first arrival in the order log is kept, wherever it stands in the file."
    )
    raw_orders = raw_frames[config.RAW_ORDER_HISTORY_EVAL_FILE]
    template = raw_orders[(raw_orders[config.ORDER_ROUTE_COLUMN] == "Portal")
                          & raw_orders[config.ORDER_TARGET_COLUMN].isin(["Yes", "No"])].iloc[0]
    other_product = sorted(set(raw_orders[config.PRODUCT_CODE_COLUMN].str.strip())
                           - {template[config.PRODUCT_CODE_COLUMN].strip()})[0]
    shared = dict(template.to_dict(), **{
        config.ORDER_REFERENCE_COLUMN: "PO-990001", config.PACK_DESCRIPTION_COLUMN: "12x500ml",
        config.CASES_PER_PALLET_COLUMN: "60", "ordered_vs_usual_pct": "180.0",
        "same_order_other_route_flag": "Yes"})
    crafted = [
        dict(shared, **{
            config.ORDER_LINE_ID_COLUMN: "OL-9900001", config.ORDER_ROUTE_COLUMN: "Email",
            config.ACCOUNT_CODE_COLUMN: "Acct 21", config.RECEIVED_SEQUENCE_COLUMN: "99002",
            config.ORDERED_QUANTITY_COLUMN: "1 plt"}),
        dict(shared, **{
            config.ORDER_LINE_ID_COLUMN: "OL-9900002", config.ORDER_ROUTE_COLUMN: "EDI",
            config.ACCOUNT_CODE_COLUMN: "C0021", config.RECEIVED_SEQUENCE_COLUMN: "99001",
            config.ORDERED_QUANTITY_COLUMN: "240 units"}),
        dict(shared, **{
            config.ORDER_LINE_ID_COLUMN: "OL-9900003", config.ORDER_ROUTE_COLUMN: "EDI",
            config.ACCOUNT_CODE_COLUMN: "C0021", config.RECEIVED_SEQUENCE_COLUMN: "99003",
            config.PRODUCT_CODE_COLUMN: other_product,
            config.ORDERED_QUANTITY_COLUMN: "120 units"}),
        dict(shared, **{
            config.ORDER_LINE_ID_COLUMN: "OL-9900004", config.ORDER_ROUTE_COLUMN: "Portal",
            config.ORDER_REFERENCE_COLUMN: "PO-990002", config.ACCOUNT_CODE_COLUMN: "CUS-0021",
            config.RECEIVED_SEQUENCE_COLUMN: "99005", config.ORDERED_QUANTITY_COLUMN: "10 cs"}),
        dict(shared, **{
            config.ORDER_LINE_ID_COLUMN: "OL-9900005", config.ORDER_ROUTE_COLUMN: "EDI",
            config.ORDER_REFERENCE_COLUMN: "PO-990002", config.ACCOUNT_CODE_COLUMN: "C0021",
            config.RECEIVED_SEQUENCE_COLUMN: "99004", config.ORDER_TARGET_COLUMN: "",
            config.ORDERED_QUANTITY_COLUMN: "120 units"}),
    ]
    crafted_path = str(tmp_path / "order_sent_twice.csv")
    cleaned_path = str(tmp_path / "order_sent_twice_clean.csv")
    pd.DataFrame(crafted).to_csv(crafted_path, index=False)
    clean_order_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False)
    found = sorted(zip(cleaned[config.ORDER_LINE_ID_COLUMN], cleaned[config.ACCOUNT_CODE_COLUMN],
                       cleaned[config.ORDERED_QUANTITY_COLUMN]))
    expected = [("OL-9900002", "CUS-0021", "20"), ("OL-9900003", "CUS-0021", "10")]
    assert found == expected, (
        "Five order lines were cleaned: purchase order PO-990001 for "
        + template[config.PRODUCT_CODE_COLUMN].strip()
        + " sent by EDI as 240 units from C0021 and again by email as 1 plt from Acct 21, the "
        "email written first in the file but received later; the same purchase order for "
        + other_product
        + " as 120 units; and purchase order PO-990002 sent first by EDI with no answer and "
        "then by the portal with one. They cleaned to "
        + str(found)
        + " (line, account, cases) where they are "
        + str(expected)
        + ". An order is a purchase order and a product; sent by two routes it is kept once, "
        "as it first arrived, whatever its second arrival says and wherever it stands in the "
        "file, and an order whose first arrival carries no answer is removed."
    )


def test_quantities_in_units_cases_and_pallets_become_whole_cases_with_each_products_own_sizes(
        raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.order_cleaning import clean_order_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    raw_orders = raw_frames[config.RAW_ORDER_HISTORY_EVAL_FILE]
    template = raw_orders[(raw_orders[config.ORDER_ROUTE_COLUMN] == "Portal")
                          & raw_orders[config.ORDER_TARGET_COLUMN].isin(["Yes", "No"])].iloc[0]
    accounts = {"EDI": "C0021", "Portal": "CUS-0021", "Email": "Acct 21"}
    cases = [("EDI", "12x500ml", "60", "120 units", 10),
             ("EDI", "500ml x24", "40", "120 units", 5),
             ("EDI", "6 x 5000 ml", "40", "18 units", 3),
             ("EDI", "12/500ML", "60", "1200 units", 100),
             ("Portal", "12/500ML", "60", "35 cs", 35),
             ("Portal", "6x750ml", "80", "1 cs", 1),
             ("Email", "12x500ml", "60", "2 plt", 120),
             ("Email", "12x500ml", "48", "3 plt", 144),
             ("Email", "2x5000ml", "40", "7 cs", 7)]
    crafted = [dict(template.to_dict(), **{
        config.ORDER_LINE_ID_COLUMN: "OL-9800" + str(position),
        config.ORDER_REFERENCE_COLUMN: "PO-9800" + str(position),
        config.RECEIVED_SEQUENCE_COLUMN: str(98000 + position),
        config.ORDER_ROUTE_COLUMN: route, config.ACCOUNT_CODE_COLUMN: accounts[route],
        config.PACK_DESCRIPTION_COLUMN: pack, config.CASES_PER_PALLET_COLUMN: pallet,
        config.ORDERED_QUANTITY_COLUMN: written,
    }) for position, (route, pack, pallet, written, whole_cases) in enumerate(cases)]
    crafted_path = str(tmp_path / "quantities_written_every_way.csv")
    cleaned_path = str(tmp_path / "quantities_written_every_way_clean.csv")
    pd.DataFrame(crafted).to_csv(crafted_path, index=False)
    clean_order_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index(
        config.ORDER_LINE_ID_COLUMN)
    for position, (route, pack, pallet, written, whole_cases) in enumerate(cases):
        line_id = "OL-9800" + str(position)
        read_as = (cleaned.loc[line_id, config.ORDERED_QUANTITY_COLUMN]
                   if line_id in cleaned.index else "a removed row")
        assert read_as == str(whole_cases), (
            "An order sent by "
            + route
            + " for a product packed "
            + pack
            + " with "
            + pallet
            + " cases to a pallet, written "
            + repr(written)
            + ", cleaned to "
            + repr(read_as)
            + " where it is "
            + str(whole_cases)
            + " cases. EDI writes consumer units, the portal cases and email cases or pallets, "
            "and each is turned into whole cases with the product's own units to a case and "
            "cases to a pallet."
        )
    refusals = [("EDI", "12x500ml", "125 units", "units that are not a whole number of cases"),
                ("EDI", "12x500ml", "10 cs", "cases from EDI, which writes consumer units"),
                ("Portal", "12x500ml", "3 plt", "pallets from the portal, which writes cases"),
                ("Email", "12x500ml", "40 units", "consumer units from email, which writes cases "
                                                  "or pallets"),
                ("Portal", "12x500ml", "35 cases", "a unit word no route writes"),
                ("Email", "12x500ml", "2 pallets", "a unit word no route writes"),
                ("Portal", "12x500ml", "12.5 cs", "a fraction of a case"),
                ("EDI", "12x500ml", "1,200 units", "a thousands comma no route writes"),
                ("Fax", "12x500ml", "35 cs", "an order route the configuration does not know")]
    for position, (route, pack, written, reason) in enumerate(refusals):
        refused_path = str(tmp_path / ("refused_quantity_" + str(position) + ".csv"))
        pd.DataFrame([dict(template.to_dict(), **{
            config.ORDER_LINE_ID_COLUMN: "OL-9810" + str(position),
            config.ORDER_REFERENCE_COLUMN: "PO-9810" + str(position),
            config.ORDER_ROUTE_COLUMN: route,
            config.ACCOUNT_CODE_COLUMN: accounts.get(route, "CUS-0021"),
            config.PACK_DESCRIPTION_COLUMN: pack, config.CASES_PER_PALLET_COLUMN: "60",
            config.ORDERED_QUANTITY_COLUMN: written,
        })]).to_csv(refused_path, index=False)
        refused = False
        try:
            clean_order_dataset(refused_path, refused_path.replace(".csv", "_clean.csv"))
        except ValueError:
            refused = True
        assert refused, (
            "The quantity "
            + repr(written)
            + " sent by "
            + route
            + " was cleaned without a ValueError. That is "
            + reason
            + ", and a quantity read the wrong way moves cases from one customer to another, so "
            "it is refused rather than guessed."
        )
    clean_orders = processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE]
    written_quantities = raw_orders[config.ORDERED_QUANTITY_COLUMN].str.strip()
    units = set(written_quantities.str.split(" ").str[-1])
    stated_lines = [("OL-0020042", "6024 units", "packed 12 x 450 ml", 502),
                    ("OL-0002123", "28 units", "packed 2x5000ml", 14),
                    ("OL-0016005", "2676 units", "packed 12/750ML", 223),
                    ("OL-0040184", "90 units", "packed 6/1000ML", 15),
                    ("OL-0026058", "37 cs", "packed 12/750ML", 37),
                    ("OL-0016804", "647 cs", "packed 12x750ml", 647),
                    ("OL-0011854", "5 plt", "60 cases to a pallet", 300),
                    ("OL-0022217", "1 plt", "55 cases to a pallet", 55),
                    ("OL-0008758", "6 plt", "80 cases to a pallet", 480)]
    wrong = []
    for line_id, written, packed, whole_cases in stated_lines:
        as_written = sorted(set(written_quantities[
            raw_orders[config.ORDER_LINE_ID_COLUMN].str.strip() == line_id]))
        cleaned_cases = clean_orders.loc[clean_orders[config.ORDER_LINE_ID_COLUMN] == line_id,
                                         config.ORDERED_QUANTITY_COLUMN].tolist()
        if as_written != [written] or cleaned_cases != [whole_cases]:
            wrong.append((line_id, written, packed, cleaned_cases, whole_cases))
    assert len(clean_orders) > 8000 and units == {"units", "cs", "plt"} and len(wrong) == 0, (
        str(len(wrong))
        + " of nine order lines of "
        + config.RAW_ORDER_HISTORY_EVAL_FILE
        + " did not clean to the cases they come to in "
        + config.CLEAN_ORDER_HISTORY_EVAL_FILE
        + ", which holds "
        + str(len(clean_orders))
        + " rows: "
        + str(wrong)
        + " (line, as written, pack or pallet, cleaned, cases). Units are divided by the units "
        "to a case of the product's own pack, however the pack is written, and pallets "
        "multiplied by its own cases to a pallet."
    )


def test_account_codes_in_three_notations_are_made_canonical_and_any_other_is_refused(
        raw_frames, processed_frames, tmp_path):
    import re

    import pandas as pd

    from config import config
    from data_pipeline.order_cleaning import clean_order_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    raw_orders = raw_frames[config.RAW_ORDER_HISTORY_EVAL_FILE]
    written = raw_orders[config.ACCOUNT_CODE_COLUMN].str.strip()
    assert int((~written.str.fullmatch(config.CANONICAL_ACCOUNT_CODE_PATTERN)).sum()) > 1000, (
        "The raw order history writes almost every account code canonically, where EDI and "
        "email write their own notations; the data folder has to stay as it was supplied."
    )
    for clean_path in [config.CLEAN_ORDER_HISTORY_TRAIN_FILE, config.CLEAN_ORDER_HISTORY_EVAL_FILE]:
        codes = processed_frames[clean_path][config.ACCOUNT_CODE_COLUMN].astype(str)
        unreadable = sorted(set(codes[~codes.str.fullmatch(config.CANONICAL_ACCOUNT_CODE_PATTERN)]))
        assert len(unreadable) == 0, (
            clean_path
            + " still holds account codes that are not canonical, for example "
            + str(unreadable[:5])
            + ". Every route's notation is read into one code, CUS, a hyphen and four digits, or "
            "one customer is three customers and its orders never meet its shipments."
        )
    clean_orders = processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE].set_index(
        config.ORDER_LINE_ID_COLUMN)
    written_orders = raw_orders.drop_duplicates(config.ORDER_LINE_ID_COLUMN).set_index(
        config.ORDER_LINE_ID_COLUMN)
    written_orders = written_orders.loc[written_orders.index.isin(clean_orders.index)]
    clean_codes = clean_orders.loc[written_orders.index, config.ACCOUNT_CODE_COLUMN].astype(str)
    renumbered = [(line_id, code, clean_code) for line_id, code, clean_code
                  in zip(written_orders.index, written_orders[config.ACCOUNT_CODE_COLUMN],
                         clean_codes)
                  if int(re.sub(r"[^0-9]", "", code)) != int(clean_code[-4:])]
    assert len(renumbered) == 0, (
        str(len(renumbered))
        + " order lines in "
        + config.CLEAN_ORDER_HISTORY_EVAL_FILE
        + " carry an account number other than the one written, for example "
        + str(renumbered[:3])
        + ". A notation changes how the number is written, never the number."
    )
    template = raw_orders[(raw_orders[config.ORDER_ROUTE_COLUMN] == "Portal")
                          & raw_orders[config.ORDER_TARGET_COLUMN].isin(["Yes", "No"])].iloc[0]
    quantities = {"EDI": "120 units", "Portal": "10 cs", "Email": "10 cs"}
    cases = [("EDI", "C0021", "CUS-0021"), ("Portal", "CUS-0021", "CUS-0021"),
             ("Email", "Acct 21", "CUS-0021"), ("Email", "Acct 7", "CUS-0007"),
             ("EDI", "C0180", "CUS-0180"), ("Email", "Acct 180", "CUS-0180"),
             ("Portal", "CUS-0007", "CUS-0007")]
    crafted = [dict(template.to_dict(), **{
        config.ORDER_LINE_ID_COLUMN: "OL-9700" + str(position),
        config.ORDER_REFERENCE_COLUMN: "PO-9700" + str(position),
        config.RECEIVED_SEQUENCE_COLUMN: str(97000 + position),
        config.ORDER_ROUTE_COLUMN: route, config.ACCOUNT_CODE_COLUMN: code,
        config.PACK_DESCRIPTION_COLUMN: "12x500ml",
        config.ORDERED_QUANTITY_COLUMN: quantities[route],
    }) for position, (route, code, canonical) in enumerate(cases)]
    crafted_path = str(tmp_path / "account_codes_written_every_way.csv")
    cleaned_path = str(tmp_path / "account_codes_written_every_way_clean.csv")
    pd.DataFrame(crafted).to_csv(crafted_path, index=False)
    clean_order_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index(
        config.ORDER_LINE_ID_COLUMN)
    for position, (route, code, canonical) in enumerate(cases):
        line_id = "OL-9700" + str(position)
        read_as = (cleaned.loc[line_id, config.ACCOUNT_CODE_COLUMN] if line_id in cleaned.index
                   else "a removed row")
        assert read_as == canonical, (
            "An account code that "
            + route
            + " wrote as "
            + repr(code)
            + " cleaned to "
            + repr(read_as)
            + " where it is "
            + canonical
            + ". The configuration says how each order route writes a code: the portal CUS-0021, "
            "EDI C0021 and email Acct 21."
        )
    refusals = [("Portal", "C0021", "EDI's notation from the portal"),
                ("EDI", "CUS-0021", "the portal's notation from EDI"),
                ("Email", "CUS-0021", "the portal's notation from email"),
                ("Email", "Acct 0021", "a leading nought email never writes"),
                ("Portal", "cus-0021", "a code in small letters"),
                ("EDI", "C021", "three digits where EDI writes four"),
                ("Email", "Account 21", "a notation no route writes")]
    for position, (route, code, reason) in enumerate(refusals):
        refused_path = str(tmp_path / ("refused_code_" + str(position) + ".csv"))
        pd.DataFrame([dict(template.to_dict(), **{
            config.ORDER_LINE_ID_COLUMN: "OL-9710" + str(position),
            config.ORDER_REFERENCE_COLUMN: "PO-9710" + str(position),
            config.ORDER_ROUTE_COLUMN: route, config.ACCOUNT_CODE_COLUMN: code,
            config.PACK_DESCRIPTION_COLUMN: "12x500ml",
            config.ORDERED_QUANTITY_COLUMN: quantities[route],
        })]).to_csv(refused_path, index=False)
        refused = False
        try:
            clean_order_dataset(refused_path, refused_path.replace(".csv", "_clean.csv"))
        except ValueError:
            refused = True
        assert refused, (
            "The account code "
            + repr(code)
            + " sent by "
            + route
            + " was cleaned without a ValueError. That is "
            + reason
            + ", and a code that cannot be read for certain is refused, never guessed: read the "
            "wrong way it gives one customer's order to another."
        )


def test_channels_regions_packs_and_yes_or_no_take_one_spelling_and_units_are_read(
        raw_frames, processed_frames, tmp_path):
    import re

    import pandas as pd

    from config import config
    from data_pipeline.demand_cleaning import clean_demand_dataset
    from data_pipeline.email_cleaning import clean_email_dataset
    from data_pipeline.order_cleaning import clean_order_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    for clean_path, value_mappings in [
            (config.CLEAN_ORDER_HISTORY_TRAIN_FILE, config.ORDER_VALUE_MAPPINGS),
            (config.CLEAN_ORDER_HISTORY_EVAL_FILE, config.ORDER_VALUE_MAPPINGS),
            (config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE, config.DEMAND_VALUE_MAPPINGS),
            (config.CLEAN_WEEKLY_DEMAND_EVAL_FILE, config.DEMAND_VALUE_MAPPINGS),
            (config.CLEAN_CUSTOMER_EMAILS_FILE, config.EMAIL_VALUE_MAPPINGS)]:
        clean_frame = processed_frames[clean_path]
        for column_name, value_mapping in value_mappings.items():
            spellings = set(clean_frame[column_name].dropna().astype(str))
            strays = sorted(spellings - set(value_mapping.values()))
            assert len(strays) == 0 and len(spellings) > 1, (
                column_name
                + " in "
                + clean_path
                + " still holds "
                + str(strays[:6])
                + ", which are not among the names the configuration maps every spelling to."
            )
        if config.PACK_DESCRIPTION_COLUMN in clean_frame.columns:
            packs = clean_frame[config.PACK_DESCRIPTION_COLUMN].dropna().astype(str)
            canonical = packs.str.fullmatch(config.CANONICAL_PACK_DESCRIPTION_PATTERN)
            strays = sorted(set(packs[~canonical]))
            assert len(strays) == 0, (
                clean_path
                + " still holds pack descriptions written otherwise than as units to a case, x, "
                "and millilitres, for example "
                + str(strays[:5])
                + "."
            )
        if config.CUSTOMER_NAME_COLUMN in clean_frame.columns:
            names = clean_frame[config.CUSTOMER_NAME_COLUMN].dropna().astype(str)
            untidy = sorted(set(names[(names != names.str.strip())
                                      | names.str.contains(r"\s{2,}| (?:Ltd|Limited)$")]))
            assert len(untidy) == 0, (
                clean_path
                + " still holds customer names padded, double spaced or ending in Ltd or Limited, "
                "for example "
                + str(untidy[:3])
                + ", so one customer appears under several names."
            )
    raw_orders = raw_frames[config.RAW_ORDER_HISTORY_EVAL_FILE]
    template = raw_orders[(raw_orders[config.ORDER_ROUTE_COLUMN] == "Portal")
                          & raw_orders[config.ORDER_TARGET_COLUMN].isin(["Yes", "No"])].iloc[0]
    order_cases = [
        ({config.CHANNEL_COLUMN: "supermkt", "region": "Yorks", "key_account_flag": "Y",
          config.PACK_DESCRIPTION_COLUMN: "500ml x12", "ordered_vs_usual_pct": "250%",
          "orders_this_week": "3 orders",
          config.CUSTOMER_NAME_COLUMN: "  Kestwell   Cleaning Ltd "},
         {config.CHANNEL_COLUMN: "Supermarket", "region": "Yorkshire", "key_account_flag": "Yes",
          config.PACK_DESCRIPTION_COLUMN: "12x500ml", "ordered_vs_usual_pct": "250",
          "orders_this_week": "3", config.CUSTOMER_NAME_COLUMN: "Kestwell Cleaning"}),
        ({config.CHANNEL_COLUMN: "Grocery multiple", "region": "cymru",
          "shortage_publicised_flag": "TRUE", config.PACK_DESCRIPTION_COLUMN: "12 x 500 ml",
          "ordered_vs_usual_pct": "250.0 %", "customer_stock_cover_weeks": "2.5 wks",
          "lead_time_days": "14 days", config.CUSTOMER_NAME_COLUMN: "Kestwell Cleaning Limited"},
         {config.CHANNEL_COLUMN: "Supermarket", "region": "Wales",
          "shortage_publicised_flag": "Yes", config.PACK_DESCRIPTION_COLUMN: "12x500ml",
          "ordered_vs_usual_pct": "250", "customer_stock_cover_weeks": "2.5",
          "lead_time_days": "14", config.CUSTOMER_NAME_COLUMN: "Kestwell Cleaning"}),
        ({config.CHANNEL_COLUMN: "Chemist", "region": "ne", "promotion_flag": "0",
          config.PACK_DESCRIPTION_COLUMN: "12/500ML", "sites_supplied": "240 sites",
          "account_age_weeks": "52 wks", "orders_this_week": "1 order",
          config.ORDER_WEEKDAY_COLUMN: "THU"},
         {config.CHANNEL_COLUMN: "Pharmacy", "region": "North East", "promotion_flag": "No",
          config.PACK_DESCRIPTION_COLUMN: "12x500ml", "sites_supplied": "240",
          "account_age_weeks": "52", "orders_this_week": "1",
          config.ORDER_WEEKDAY_COLUMN: "Thursday"}),
        ({config.CHANNEL_COLUMN: "contract cleaning", "same_order_other_route_flag": "n",
          "customer_stock_cover_weeks": "1.2wks", "sites_supplied": "1 site",
          config.ORDER_WEEKDAY_COLUMN: "sa.", config.ORDER_TARGET_COLUMN: "true"},
         {config.CHANNEL_COLUMN: "Facilities", "same_order_other_route_flag": "No",
          "customer_stock_cover_weeks": "1.2", "sites_supplied": "1",
          config.ORDER_WEEKDAY_COLUMN: "Saturday", config.ORDER_TARGET_COLUMN: "Yes"}),
    ]
    demand_template = raw_frames[config.RAW_WEEKLY_DEMAND_EVAL_FILE]
    demand_template = demand_template[demand_template[config.DEMAND_TARGET_COLUMN].str.fullmatch(
        r"[0-9]+")].iloc[0]
    demand_cases = [
        ({config.CHANNEL_COLUMN: "online retailer", "product_category": "TOILET CARE",
          "last_week_cases": "450 cs", "weeks_since_launch": "68 wks",
          "promotion_next_week_flag": "yes", config.PACK_DESCRIPTION_COLUMN: "750ml x6"},
         {config.CHANNEL_COLUMN: "Online", "product_category": "Toilet care",
          "last_week_cases": "450", "weeks_since_launch": "68",
          "promotion_next_week_flag": "Yes", config.PACK_DESCRIPTION_COLUMN: "6x750ml"}),
        ({config.CHANNEL_COLUMN: "cash and carry", "sites_supplied": "31 sites",
          "distribution_change_flag": "FALSE", config.DEMAND_WEEKDAY_COLUMN: "tues",
          "avg_4_week_cases": "24.00"},
         {config.CHANNEL_COLUMN: "Wholesale", "sites_supplied": "31",
          "distribution_change_flag": "No", config.DEMAND_WEEKDAY_COLUMN: "Tuesday",
          "avg_4_week_cases": "24.0"}),
    ]
    email_template = raw_frames[config.RAW_CUSTOMER_EMAILS_FILE]
    email_template = email_template[email_template[config.EMAIL_TARGET_COLUMN].isin(
        list(config.REQUEST_GUIDE_DOCUMENTS))].iloc[0]
    email_cases = [
        ({config.SENDER_ROLE_COLUMN: "lead nurse", config.EMAIL_TARGET_COLUMN: "clinical urgency",
          config.EMAIL_RECEIVED_AT_COLUMN: "THU 09:05", config.EMAIL_SEQUENCE_COLUMN: "12"},
         {config.SENDER_ROLE_COLUMN: "Clinical lead",
          config.EMAIL_TARGET_COLUMN: "Urgent clinical need",
          config.EMAIL_RECEIVED_AT_COLUMN: "Thursday 09:05", config.EMAIL_SEQUENCE_COLUMN: "12"}),
        ({config.SENDER_ROLE_COLUMN: "ops manager", config.EMAIL_TARGET_COLUMN: "qty increase",
          config.EMAIL_RECEIVED_AT_COLUMN: "sa. 17:45", config.CHANNEL_COLUMN: "e-commerce"},
         {config.SENDER_ROLE_COLUMN: "Operations manager",
          config.EMAIL_TARGET_COLUMN: "Quantity increase",
          config.EMAIL_RECEIVED_AT_COLUMN: "Saturday 17:45", config.CHANNEL_COLUMN: "Online"}),
    ]
    number = r"-?[0-9]+(?:\.[0-9]+)?"
    for cleaner, row_template, identifier, cases in [
            (clean_order_dataset, template, config.ORDER_LINE_ID_COLUMN, order_cases),
            (clean_demand_dataset, demand_template, "demand_row_id", demand_cases),
            (clean_email_dataset, email_template, config.EMAIL_ID_COLUMN, email_cases)]:
        crafted = []
        crafted_cases = {}
        for position, (written, expected) in enumerate(cases):
            row_id = "SP-9600" + str(position)
            crafted_row = dict(row_template.to_dict(), **written)
            crafted_row[identifier] = row_id
            if identifier == config.ORDER_LINE_ID_COLUMN:
                crafted_row[config.ORDER_REFERENCE_COLUMN] = "PO-9600" + str(position)
                crafted_row[config.RECEIVED_SEQUENCE_COLUMN] = str(96000 + position)
            crafted.append(crafted_row)
            crafted_cases[row_id] = (written, expected)
        crafted_path = str(tmp_path / ("spellings_" + identifier + ".csv"))
        cleaned_path = str(tmp_path / ("spellings_" + identifier + "_clean.csv"))
        pd.DataFrame(crafted).to_csv(crafted_path, index=False)
        cleaner(crafted_path, cleaned_path)
        cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index(identifier)
        for row_id, (written, expected) in crafted_cases.items():
            read_as = ({column_name: cleaned.loc[row_id, column_name] for column_name in expected}
                       if row_id in cleaned.index else "a removed row")
            agrees = read_as != "a removed row" and all(
                float(read_as[column_name]) == float(wanted)
                if re.fullmatch(number, wanted) and re.fullmatch(number, read_as[column_name])
                else read_as[column_name] == wanted
                for column_name, wanted in expected.items())
            assert agrees, (
                "A row written "
                + str(written)
                + " cleaned to "
                + str(read_as)
                + " where it reads "
                + str(expected)
                + ". Every spelling of a channel, a region, a category, a role, a request type, "
                "a weekday and yes or no is put under one name, a pack written four ways is "
                "units to a case, x, and millilitres, and a percentage, a count, weeks, days, "
                "sites or cases written with a unit are read as the number alone."
            )


def test_impossible_readings_become_missing_and_anything_unreadable_is_refused(raw_frames,
                                                                             tmp_path):
    import re

    import pandas as pd

    from config import config
    from data_pipeline.demand_cleaning import clean_demand_dataset
    from data_pipeline.email_cleaning import clean_email_dataset
    from data_pipeline.order_cleaning import clean_order_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_orders = raw_frames[config.RAW_ORDER_HISTORY_EVAL_FILE]
    template = raw_orders[(raw_orders[config.ORDER_ROUTE_COLUMN] == "Portal")
                          & raw_orders[config.ORDER_TARGET_COLUMN].isin(["Yes", "No"])].iloc[0]
    order_cases = [
        ({"ordered_vs_usual_pct": "-131.2"}, "ordered_vs_usual_pct", "",
         "an order percentage below nought"),
        ({"customer_stock_cover_weeks": "-0.5 wks"}, "customer_stock_cover_weeks", "",
         "stock cover below nought"),
        ({"lead_time_days": "-3 days"}, "lead_time_days", "", "a lead time below nought"),
        ({"past_shortage_orders": "5", "past_orders_cut": "9"}, "past_orders_cut", "",
         "more orders cut than shortage orders placed"),
        ({"past_shortage_orders": "5", "past_orders_cut": "5"}, "past_orders_cut", "5",
         "every shortage order cut, which is possible"),
        ({"ordered_vs_usual_pct": "0"}, "ordered_vs_usual_pct", "0",
         "an order of nought percent of the usual week, which is possible"),
        ({"lead_time_days": "n/a"}, "lead_time_days", "", "a lead time written as missing"),
        ({"account_age_weeks": "Not recorded"}, "account_age_weeks", "",
         "an account age written as missing"),
    ]
    demand_template = raw_frames[config.RAW_WEEKLY_DEMAND_EVAL_FILE]
    demand_template = demand_template[demand_template[config.DEMAND_TARGET_COLUMN].str.fullmatch(
        r"[0-9]+")].iloc[0]
    demand_cases = [
        ({"seasonal_index": "-5"}, "seasonal_index", "", "a seasonal index below nought"),
        ({"last_week_cases": "-2 cs"}, "last_week_cases", "", "a week's cases below nought"),
        ({config.DEMAND_TARGET_COLUMN: "0"}, config.DEMAND_TARGET_COLUMN, "0",
         "a next week of nought cases, which is possible"),
        ({config.DEMAND_TARGET_COLUMN: "-1"}, None, None,
         "a next week's cases below nought, an answer no week can have"),
        ({"weeks_since_launch": "?"}, "weeks_since_launch", "", "weeks since launch written as "
                                                                "missing"),
    ]
    for cleaner, row_template, identifier, cases in [
            (clean_order_dataset, template, config.ORDER_LINE_ID_COLUMN, order_cases),
            (clean_demand_dataset, demand_template, "demand_row_id", demand_cases)]:
        crafted = []
        for position, case in enumerate(cases):
            crafted_row = dict(row_template.to_dict(), **case[0])
            crafted_row[identifier] = "IM-9500" + str(position)
            if identifier == config.ORDER_LINE_ID_COLUMN:
                crafted_row[config.ORDER_REFERENCE_COLUMN] = "PO-9500" + str(position)
                crafted_row[config.RECEIVED_SEQUENCE_COLUMN] = str(95000 + position)
            crafted.append(crafted_row)
        crafted_path = str(tmp_path / ("impossible_" + identifier + ".csv"))
        cleaned_path = str(tmp_path / ("impossible_" + identifier + "_clean.csv"))
        pd.DataFrame(crafted).to_csv(crafted_path, index=False)
        cleaner(crafted_path, cleaned_path)
        cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index(identifier)
        for position, (written, column_name, expected, meaning) in enumerate(cases):
            row_id = "IM-9500" + str(position)
            if column_name is None:
                assert row_id not in cleaned.index, (
                    "A row written "
                    + str(written)
                    + " was kept. That is "
                    + meaning
                    + ": an impossible answer is no answer, and the row goes."
                )
                continue
            read_as = (cleaned.loc[row_id, column_name] if row_id in cleaned.index
                       else "a removed row")
            agrees = (read_as == expected if expected == ""
                      else re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", read_as) is not None
                      and float(read_as) == float(expected))
            assert agrees, (
                "A row written "
                + str(written)
                + " cleaned with "
                + column_name
                + " as "
                + repr(read_as)
                + " where it is "
                + (repr(expected) if expected else "missing")
                + ". That is "
                + meaning
                + ": a reading impossible by definition, or written as missing, becomes a gap "
                "the model's own preparation fills, the row stays, and a possible reading stays "
                "as it is."
            )
    email_template = raw_frames[config.RAW_CUSTOMER_EMAILS_FILE]
    email_template = email_template[email_template[config.EMAIL_TARGET_COLUMN].isin(
        list(config.REQUEST_GUIDE_DOCUMENTS))].iloc[0]
    other_script_three = chr(1635)
    refusals = [
        (clean_order_dataset, template, {"ordered_vs_usual_pct": "about 250%"},
         "a percentage written in words"),
        (clean_order_dataset, template, {"orders_this_week": "3 ordrs"},
         "a count with a misspelt unit"),
        (clean_order_dataset, template, {"orders_this_week": other_script_three},
         "a digit from another script"),
        (clean_order_dataset, template, {"customer_stock_cover_weeks": "2,5 wks"},
         "weeks with a decimal comma"),
        (clean_order_dataset, template, {config.CHANNEL_COLUMN: "Hypermarket"},
         "a channel Wrenfield Hygiene does not sell through"),
        (clean_order_dataset, template, {"region": "Northern Ireland"},
         "a region Wrenfield Hygiene does not deliver to"),
        (clean_order_dataset, template, {config.PACK_DESCRIPTION_COLUMN: "12 by 500ml"},
         "a pack written in no known notation"),
        (clean_order_dataset, template, {config.ORDER_TARGET_COLUMN: "Maybe"},
         "an answer that is neither yes nor no"),
        (clean_order_dataset, template,
         {"orders_this_week": "3 ordrs", config.ORDER_TARGET_COLUMN: ""},
         "an unreadable count on a row about to be dropped for having no answer"),
        (clean_demand_dataset, demand_template, {"last_week_cases": "450 cases"},
         "cases written with a unit no extract writes"),
        (clean_demand_dataset, demand_template, {"seasonal_index": "108.5"},
         "a seasonal index with a fraction, which the planning team never writes"),
        (clean_email_dataset, email_template, {config.EMAIL_RECEIVED_AT_COLUMN: "Funday 10:00"},
         "a time whose weekday cannot be read"),
        (clean_email_dataset, email_template, {config.SENDER_ROLE_COLUMN: "Chief executive"},
         "a sender role with no canonical spelling"),
        (clean_email_dataset, email_template, {config.EMAIL_TARGET_COLUMN: "Price complaint"},
         "a request type the configuration does not know"),
    ]
    for position, (cleaner, row_template, written, reason) in enumerate(refusals):
        refused_path = str(tmp_path / ("refused_reading_" + str(position) + ".csv"))
        refused_row = dict(row_template.to_dict(), **written)
        pd.DataFrame([refused_row]).to_csv(refused_path, index=False)
        refused = False
        try:
            cleaner(refused_path, refused_path.replace(".csv", "_clean.csv"))
        except ValueError:
            refused = True
        assert refused, (
            "A row written "
            + repr(written)
            + " was cleaned without a ValueError. That is "
            + reason
            + ", and a value no notation fits is refused, never guessed, even on a row the "
            "cleaning would go on to drop."
        )


def test_saved_feature_schemas_leave_out_the_answer_every_identifier_and_the_noise(
        processed_frames, saved_models):
    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    for schema_name, schema, target, identifiers, noise, declared, train_path in [
            ("inflated_order", saved_models["inflated_order_schema"], config.ORDER_TARGET_COLUMN,
             [config.ORDER_LINE_ID_COLUMN, config.ORDER_REFERENCE_COLUMN,
              config.ACCOUNT_CODE_COLUMN, config.CUSTOMER_NAME_COLUMN,
              config.RECEIVED_SEQUENCE_COLUMN, config.PRODUCT_CODE_COLUMN],
             ["entered_by", config.ORDER_WEEKDAY_COLUMN, "form_version"],
             config.INFLATED_ORDER_NUMERIC_FEATURES + config.INFLATED_ORDER_CATEGORICAL_FEATURES,
             config.CLEAN_ORDER_HISTORY_TRAIN_FILE),
            ("demand", saved_models["demand_schema"], config.DEMAND_TARGET_COLUMN,
             ["demand_row_id", config.ACCOUNT_CODE_COLUMN, config.PRODUCT_CODE_COLUMN],
             ["planner_code", config.DEMAND_WEEKDAY_COLUMN, "extract_batch"],
             config.DEMAND_NUMERIC_FEATURES + config.DEMAND_CATEGORICAL_FEATURES,
             config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE)]:
        assert len(schema) > 0, (
            "The saved " + schema_name + " feature schema is empty, so nothing records which "
            "columns the model was fitted on."
        )
        assert target not in schema, (
            "The saved "
            + schema_name
            + " schema lists "
            + target
            + ", the answer the model predicts. A model shown its own answer scores almost "
            "perfectly and has learned nothing."
        )
        leaked = [column_name for column_name in identifiers if column_name in schema]
        assert len(leaked) == 0, (
            "The saved "
            + schema_name
            + " schema lists "
            + str(leaked)
            + ". An identifier says which record a row is, not what it means, and a model "
            "fitted on it memorises rows instead of learning from them."
        )
        noisy = [column_name for column_name in noise if column_name in schema]
        assert len(noisy) == 0, (
            "The saved "
            + schema_name
            + " schema lists "
            + str(noisy)
            + ", columns the order desk and the planning team record for their own housekeeping "
            "that say nothing about the answer. A model fitted on noise finds patterns that are "
            "not there."
        )
        assert len(schema) == len(set(schema)), (
            "The saved "
            + schema_name
            + " schema names a column twice: "
            + str(schema)
            + ". A row built to that schema no longer lines up with what the model was fitted on."
        )
        assert sorted(schema) == sorted(declared), (
            "The saved "
            + schema_name
            + " schema lists "
            + str(sorted(set(schema) - set(declared)))
            + " beyond the columns the configuration declares and leaves out "
            + str(sorted(set(declared) - set(schema)))
            + ". The inputs are declared, not guessed: every declared number and category "
            "column, and no other."
        )
        unknown = [column_name for column_name in schema
                   if column_name not in processed_frames[train_path].columns]
        assert len(unknown) == 0, (
            "The saved "
            + schema_name
            + " schema lists "
            + str(unknown)
            + ", which "
            + train_path
            + " does not have, so no cleaned row can be scored."
        )


def test_model_inputs_are_built_in_the_saved_schema_order(processed_frames, saved_models):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    for schema_name, frame, numeric_columns, category_columns in [
            ("inflated_order", processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE].copy(),
             config.INFLATED_ORDER_NUMERIC_FEATURES, config.INFLATED_ORDER_CATEGORICAL_FEATURES),
            ("demand", processed_frames[config.CLEAN_WEEKLY_DEMAND_EVAL_FILE].copy(),
             config.DEMAND_NUMERIC_FEATURES, config.DEMAND_CATEGORICAL_FEATURES)]:
        shuffled = frame.sample(frac=1.0, random_state=config.RANDOM_SEED)
        features = build_feature_frame(shuffled, numeric_columns, category_columns)
        built = list(features.columns)
        saved = list(saved_models[schema_name + "_schema"])
        assert built == saved, (
            "The "
            + schema_name
            + " features built from the cleaned held-out rows are "
            + str(built[:5])
            + " ... ("
            + str(len(built))
            + " columns) but the schema saved beside the model is "
            + str(saved[:5])
            + " ... ("
            + str(len(saved))
            + " columns). Same names in a different order score silently and wrongly."
        )
        assert len(features) == len(shuffled) and list(features.index) == list(shuffled.index), (
            "Building the "
            + schema_name
            + " inputs turned "
            + str(len(shuffled))
            + " rows into "
            + str(len(features))
            + ", or changed their order. Choosing columns must not drop, add or move rows."
        )
        not_numbers = [column_name for column_name in numeric_columns
                       if features[column_name].dtype.kind != "f"]
        assert len(not_numbers) == 0, (
            "The "
            + schema_name
            + " inputs carry "
            + str(not_numbers)
            + " as something other than floating-point numbers, so a gap cannot be told from a "
            "value."
        )
        written_blanks = frame.head(4).copy().reset_index(drop=True)
        written_blanks[numeric_columns[-1]] = pd.Series([2.5, None, pd.NA, np.nan], dtype=object)
        written_blanks[category_columns[0]] = pd.Series(["Planted", None, pd.NA, np.nan],
                                                        dtype=object)
        building_problem = None
        try:
            blank_features = build_feature_frame(written_blanks, numeric_columns, category_columns)
        except (TypeError, ValueError) as error:
            building_problem = type(error).__name__ + ": " + str(error)[:200]
        assert building_problem is None, (
            "Written as [2.5, None, pd.NA, NaN], a number column, and written as ['Planted', "
            "None, pd.NA, NaN], a category column, made build_feature_frame raise "
            + str(building_problem)
            + ". A gap is written None, pd.NA or NaN depending on where the row came from, and "
            "each has to reach the model as a gap, never stop the scoring."
        )
        numbers = list(blank_features[numeric_columns[-1]])
        categories = list(blank_features[category_columns[0]])
        assert (numbers[0] == 2.5
                and all(isinstance(value, float) and np.isnan(value) for value in numbers[1:])
                and categories[0] == "Planted"
                and all(pd.isna(value) for value in categories[1:])), (
            "Written as [2.5, None, pd.NA, NaN], a number column came out of "
            "build_feature_frame as "
            + repr(numbers)
            + ", and written as ['Planted', None, pd.NA, NaN], a category column came out as "
            + repr(categories)
            + ". A value has to come out as itself and a gap as a gap: a live order reaches "
            "the model from a file, a database or the service, each writing a gap its own way, "
            "and the pipeline saved with the model handles a gap only if it still reads as one."
        )


def test_saved_models_score_rows_with_gaps_and_values_never_seen(processed_frames,
                                                                 saved_models):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    for model_name, frame, numeric_columns, category_columns in [
            ("inflated_order",
             processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE].head(60).copy(),
             config.INFLATED_ORDER_NUMERIC_FEATURES, config.INFLATED_ORDER_CATEGORICAL_FEATURES),
            ("demand", processed_frames[config.CLEAN_WEEKLY_DEMAND_EVAL_FILE].head(60).copy(),
             config.DEMAND_NUMERIC_FEATURES, config.DEMAND_CATEGORICAL_FEATURES)]:
        schema = saved_models[model_name + "_schema"]
        rows = build_feature_frame(frame, numeric_columns, category_columns)[schema].copy()
        rows.loc[rows.index[:20], numeric_columns[0]] = np.nan
        rows.loc[rows.index[20:40], category_columns[0]] = np.nan
        if isinstance(rows[category_columns[-1]].dtype, pd.CategoricalDtype):
            rows[category_columns[-1]] = rows[category_columns[-1]].cat.add_categories(
                ["A value no training row ever held"])
        rows.loc[rows.index[40:], category_columns[-1]] = "A value no training row ever held"
        model = saved_models[model_name]
        refusal = ""
        try:
            scores = (model.predict_proba(rows)[:, 1] if model_name == "inflated_order"
                      else model.predict(rows))
        except ValueError as error:
            scores, refusal = [], str(error)
        assert refusal == "", (
            "The saved "
            + model_name
            + " model refused rows with a gap or a value it never saw: "
            + refusal[:200]
            + ". A live order or customer week arrives with gaps and new spellings, so the "
            "pipeline saved with the model has to fill every gap and ignore what it does not "
            "know."
        )
        assert len(scores) == len(rows), (
            "The saved "
            + model_name
            + " model returned "
            + str(len(scores))
            + " scores for "
            + str(len(rows))
            + " rows. Every row given has to get a score, or the scores are matched to the "
            "wrong rows."
        )
        assert np.isfinite(np.asarray(scores, dtype=float)).all(), (
            "The saved "
            + model_name
            + " model returned an empty or infinite score for a row with a gap or a value it "
            "never saw."
        )


def test_each_saved_model_was_fitted_on_the_columns_its_schema_names(saved_models):
    assert saved_models["problem"] is None, saved_models["problem"]
    for model_name in ["inflated_order", "demand"]:
        model = saved_models[model_name]
        schema = [str(column_name) for column_name in saved_models[model_name + "_schema"]]
        fitted_on = getattr(model, "feature_names_in_", None)
        if fitted_on is None:
            assert getattr(model, "n_features_in_", None) == len(schema), (
                "The saved "
                + model_name
                + " model records neither the names nor the number of the columns it was "
                "fitted on, or the number is not the "
                + str(len(schema))
                + " its saved schema lists, so nothing shows that the schema describes it."
            )
        else:
            fitted_names = [str(column_name) for column_name in fitted_on]
            assert fitted_names == schema, (
                "The saved "
                + model_name
                + " model was fitted on "
                + str(fitted_names[:5])
                + " ... ("
                + str(len(fitted_names))
                + " columns) but the schema saved beside it lists "
                + str(schema[:5])
                + " ... ("
                + str(len(schema))
                + " columns). Live rows are built to the schema, so a model fitted on other "
                "columns, or on the same ones in another order, scores them wrongly or not at "
                "all."
            )
    classes = getattr(saved_models["inflated_order"], "classes_", None)
    assert classes is not None and list(classes) == [0, 1], (
        "The saved inflated-order model's classes are "
        + str(None if classes is None else list(classes))
        + ". Every tool reads the second column of its probabilities as the chance that the "
        "customer cuts the order once supply recovers, which holds only when the classes are 0 "
        "and 1 in that order."
    )


def test_the_quality_report_accounts_for_every_row_it_received(raw_frames, processed_frames,
                                                               tmp_path):
    import json
    import os
    import pathlib

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.order_cleaning import clean_order_dataset
    from data_pipeline.quality_profiler import (profile_dataset, profile_demand_details,
                                                profile_email_details, profile_order_details)

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    planted_raw = pd.DataFrame({"record_id": ["A", "A", "B", "B", "C"],
                                "reading": ["1", "1", None, None, "3"],
                                "answer": ["Yes", "Yes", None, None, "No"]})
    planted_clean = pd.DataFrame({"record_id": ["A", "C"], "reading": ["1", None],
                                  "answer": ["Yes", "No"]})
    planted_expected = {"rows_received": 5, "rows_kept": 2, "rows_removed": 3,
                        "duplicate_rows": 2, "rows_missing_target": 1, "columns": 3,
                        "missing_cells_as_received": {"reading": 1, "answer": 1},
                        "missing_cells_after_cleaning": {"reading": 1}}
    planted_profile = profile_dataset(planted_raw, planted_clean, "answer")
    planted_found = {field_name: planted_profile.get(field_name)
                     for field_name in planted_expected}
    assert planted_found == planted_expected, (
        "Profiling a planted extract of five rows, two of them repeats of the row before and "
        "one first copy with no answer, against a cleaned table of two rows with one gap, gave "
        + str(planted_found)
        + " where the rows give "
        + str(planted_expected)
        + ". A repeat is counted for every copy after the first; a missing answer, and every "
        "figure as received, only once for each record, over its first copy; and each gap in "
        "the column that holds it."
    )
    assert os.path.isfile(config.DATA_QUALITY_REPORT_FILE), (
        "There is no quality report at "
        + config.DATA_QUALITY_REPORT_FILE
        + ". The data pipeline writes one beside the cleaned files to say what cleaning did."
    )
    with open(config.DATA_QUALITY_REPORT_FILE, encoding="utf-8") as report_file:
        report = json.load(report_file)
    for raw_path, clean_path, target, removal_field, profile_details, profiled_again in [
            (config.RAW_ORDER_HISTORY_TRAIN_FILE, config.CLEAN_ORDER_HISTORY_TRAIN_FILE,
             config.ORDER_TARGET_COLUMN, "answered_second_arrivals_removed",
             profile_order_details, False),
            (config.RAW_ORDER_HISTORY_EVAL_FILE, config.CLEAN_ORDER_HISTORY_EVAL_FILE,
             config.ORDER_TARGET_COLUMN, "answered_second_arrivals_removed",
             profile_order_details, True),
            (config.RAW_WEEKLY_DEMAND_TRAIN_FILE, config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE,
             config.DEMAND_TARGET_COLUMN, "rows_with_impossible_target",
             profile_demand_details, False),
            (config.RAW_WEEKLY_DEMAND_EVAL_FILE, config.CLEAN_WEEKLY_DEMAND_EVAL_FILE,
             config.DEMAND_TARGET_COLUMN, "rows_with_impossible_target",
             profile_demand_details, True),
            (config.RAW_CUSTOMER_EMAILS_FILE, config.CLEAN_CUSTOMER_EMAILS_FILE,
             config.EMAIL_TARGET_COLUMN, "rows_with_too_few_words", profile_email_details,
             True)]:
        name = pathlib.Path(raw_path).stem
        assert name in report, (
            "The quality report has no entry for "
            + name
            + ", so nothing says what cleaning did to "
            + raw_path
            + "."
        )
        entry = report[name]
        clean_frame = processed_frames[clean_path]
        file_facts = {"rows_received": len(raw_frames[raw_path]),
                      "rows_kept": len(clean_frame),
                      "rows_removed": len(raw_frames[raw_path]) - len(clean_frame),
                      "columns": int(clean_frame.shape[1])}
        if profiled_again:
            raw_frame = load_raw_dataset(raw_path)
            profiled = dict(profile_dataset(raw_frame, clean_frame, target),
                            **profile_details(raw_frame, clean_frame))
        else:
            profiled = dict(entry, **file_facts)
        missing = sorted((set(planted_expected) | {removal_field}) - set(entry)
                         | (set(planted_expected) | {removal_field}) - set(profiled))
        assert len(missing) == 0, (
            "The quality report entry for "
            + name
            + ", or the profile worked out for it, has no "
            + str(missing)
            + ", so what cleaning did to that extract cannot be read."
        )
        for field_name in sorted(set(profiled) if profiled_again else set(planted_expected)):
            wanted = file_facts.get(field_name, profiled[field_name])
            assert entry.get(field_name) == profiled[field_name] == wanted, (
                "The quality report says "
                + field_name
                + " is "
                + str(entry.get(field_name))[:200]
                + " for "
                + name
                + (" and profiling the extract again gives " + str(profiled[field_name])[:200]
                   if profiled_again else "")
                + (", but the files say " + str(wanted) if field_name in file_facts else "")
                + ". A report that does not match the data it describes cannot be used to "
                "defend what the pipeline did."
            )
        assert entry["duplicate_rows"] > 0 and entry["rows_missing_target"] > 0, (
            "The quality report gives "
            + name
            + " "
            + str(entry["duplicate_rows"])
            + " repeated rows and "
            + str(entry["rows_missing_target"])
            + " rows with no answer, where the extract arrives with both."
        )
        assert entry[removal_field] > 0 and entry["rows_removed"] == (
                entry["duplicate_rows"] + entry["rows_missing_target"] + entry[removal_field]), (
            "The quality report says "
            + str(entry["rows_removed"])
            + " rows were removed from "
            + name
            + " but gives reasons for "
            + str(entry["duplicate_rows"] + entry["rows_missing_target"] + entry[removal_field])
            + ", "
            + str(entry[removal_field])
            + " of them under "
            + removal_field
            + ". Every row removed is accounted for once: a repeat, then no answer, then a "
            "later arrival of an order, an impossible answer or an email too short to learn "
            "from."
        )
    raw_orders = load_raw_dataset(config.RAW_ORDER_HISTORY_EVAL_FILE)
    planted_orders = raw_orders[raw_orders[config.ORDER_TARGET_COLUMN].notna()].head(3).copy()
    planted_orders[config.ORDER_LINE_ID_COLUMN] = ["OL-9400001", "OL-9400002", "OL-9400003"]
    planted_orders[config.ORDER_REFERENCE_COLUMN] = "PO-940001"
    planted_orders[config.PRODUCT_CODE_COLUMN] = planted_orders[config.PRODUCT_CODE_COLUMN].iloc[0]
    planted_orders[config.PACK_DESCRIPTION_COLUMN] = "12x500ml"
    planted_orders[config.CASES_PER_PALLET_COLUMN] = "60"
    planted_orders[config.RECEIVED_SEQUENCE_COLUMN] = ["94003", "94001", "94002"]
    planted_orders[config.ORDER_ROUTE_COLUMN] = ["Portal", "EDI", "Email"]
    planted_orders[config.ACCOUNT_CODE_COLUMN] = ["CUS-0021", "C0021", "Acct 21"]
    planted_orders[config.ORDERED_QUANTITY_COLUMN] = ["10 cs", "120 units", "1 plt"]
    planted_path = str(tmp_path / "planted_order_history.csv")
    planted_clean_path = str(tmp_path / "planted_order_history_clean.csv")
    pd.concat([planted_orders, planted_orders.iloc[[1]]], ignore_index=True).to_csv(
        planted_path, index=False)
    clean_order_dataset(planted_path, planted_clean_path)
    planted_details = profile_order_details(
        load_raw_dataset(planted_path),
        pd.read_csv(planted_clean_path, keep_default_na=False, na_values=[""], low_memory=False))
    found = {field_name: planted_details.get(field_name) for field_name in [
        "answered_second_arrivals_removed", "account_codes_not_written_canonically",
        "quantities_by_unit_as_written"]}
    expected = {"answered_second_arrivals_removed": 2, "account_codes_not_written_canonically": 2,
                "quantities_by_unit_as_written": {"cs": 1, "units": 1, "plt": 1}}
    assert found == expected, (
        "Profiling four planted order lines, one purchase order sent by the portal as 10 cs "
        "from CUS-0021, by EDI as 120 units from C0021 and by email as 1 plt from Acct 21, EDI's "
        "first in the order log and the fourth line an exact repeat of it, against the table "
        "clean_order_dataset made of them, gave "
        + str(found)
        + " where the lines give "
        + str(expected)
        + ". Each record is counted once, over its first copy: two answered arrivals come after "
        "the first, two codes are written otherwise than canonically, and one quantity is "
        "written in each unit."
    )


def test_inflated_order_model_ranks_held_out_orders_above_the_configured_minimum(
        processed_frames, saved_models):
    import copy
    import json
    import os
    import re

    import numpy as np
    from sklearn.metrics import roc_auc_score

    from config import config
    from data_pipeline.feature_builder import build_feature_frame
    from modeling.model_evaluation import evaluate_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    train_frame = processed_frames[config.CLEAN_ORDER_HISTORY_TRAIN_FILE].head(6000)
    held_out_frame = processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE]
    schema = saved_models["inflated_order_schema"]
    train_features = build_feature_frame(train_frame.copy(),
                                         config.INFLATED_ORDER_NUMERIC_FEATURES,
                                         config.INFLATED_ORDER_CATEGORICAL_FEATURES)[schema]
    held_out_features = build_feature_frame(held_out_frame.copy(),
                                            config.INFLATED_ORDER_NUMERIC_FEATURES,
                                            config.INFLATED_ORDER_CATEGORICAL_FEATURES)[schema]
    train_answers = train_frame[config.ORDER_TARGET_COLUMN].map(
        config.ORDER_TARGET_ENCODING).astype(int)
    held_out_answers = held_out_frame[config.ORDER_TARGET_COLUMN].map(
        config.ORDER_TARGET_ENCODING).astype(int)
    model = saved_models["inflated_order"]
    scores = evaluate_classifier(copy.deepcopy(model), train_features, train_answers,
                                 held_out_features, held_out_answers)
    missing = sorted({"held_out_roc_auc", "train_roc_auc", "generalisation_gap"} - set(scores))
    assert len(missing) == 0, (
        "evaluate_classifier returned no "
        + str(missing)
        + ", so the inflated-order model's scores cannot be read."
    )
    assert scores["held_out_roc_auc"] >= config.MINIMUM_INFLATED_ORDER_ROC_AUC, (
        "The inflated-order model scored "
        + str(scores["held_out_roc_auc"])
        + " ROC AUC on the held-out orders, below the "
        + str(config.MINIMUM_INFLATED_ORDER_ROC_AUC)
        + " the configuration sets. The bar sits below what a sound model reaches."
    )
    probabilities = model.predict_proba(held_out_features)[:, 1]
    measured = roc_auc_score(held_out_answers, probabilities)
    assert abs(scores["held_out_roc_auc"] - measured) < 0.006, (
        "The evaluation reports a ROC AUC of "
        + str(scores["held_out_roc_auc"])
        + " for the held-out orders, but the model's own probabilities give "
        + str(round(measured, 4))
        + ". evaluate_classifier scores the fitted model it is given, without fitting it "
        "again, on the held-out rows and from probabilities."
    )
    trained_on = roc_auc_score(train_answers, model.predict_proba(train_features)[:, 1])
    assert abs(scores["train_roc_auc"] - trained_on) < 0.006, (
        "The evaluation reports a training ROC AUC of "
        + str(scores["train_roc_auc"])
        + ", but the model's own probabilities on the training rows it was given give "
        + str(round(trained_on, 4))
        + ". The training score is measured on the training rows, or the gap cannot show how "
        "much the model memorised."
    )
    assert abs(scores["generalisation_gap"]
               - (scores["train_roc_auc"] - scores["held_out_roc_auc"])) < 0.011, (
        "The reported gap "
        + str(scores["generalisation_gap"])
        + " is not the training score "
        + str(scores["train_roc_auc"])
        + " less the held-out score "
        + str(scores["held_out_roc_auc"])
        + "."
    )
    answers = np.asarray(held_out_answers)
    cut = float(probabilities[answers == 1].mean())
    kept = float(probabilities[answers == 0].mean())
    assert cut > kept, (
        "Orders the customer went on to cut average "
        + str(round(cut, 4))
        + " and orders it kept "
        + str(round(kept, 4))
        + ". The model scores the wrong way round."
    )
    assert abs(float(probabilities.mean()) - float(answers.mean())) < 0.05, (
        "The held-out orders average a probability of "
        + str(round(float(probabilities.mean()), 4))
        + " that the customer cuts the order, where "
        + str(round(float(answers.mean()), 4))
        + " of them were cut. The score is read as a probability and banded at fixed "
        "thresholds, so a model that leans one way puts orders in the wrong band."
    )
    assert os.path.isfile(config.MODEL_METRICS_FILE), (
        "There is no metrics file at "
        + config.MODEL_METRICS_FILE
        + ". Training writes the scores it measured beside the models it saved."
    )
    with open(config.MODEL_METRICS_FILE, encoding="utf-8") as metrics_file:
        recorded = json.dumps(json.load(metrics_file))
    figures = [float(figure) for figure in re.findall(r"-?\d+\.\d+(?:[eE]-?\d+)?", recorded)]
    assert any(abs(figure - measured) < 0.0006 for figure in figures), (
        config.MODEL_METRICS_FILE
        + " does not record the held-out ROC AUC of "
        + str(round(measured, 4))
        + " that the saved inflated-order model scores, so the metrics describe some other "
        "model."
    )


def test_demand_model_forecasts_held_out_weeks_above_the_minimum_and_never_below_nought(
        processed_frames, saved_models, live_files):
    import copy

    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score

    from config import config
    from data_pipeline.feature_builder import build_feature_frame
    from modeling.model_evaluation import evaluate_regressor
    from tools.prediction_tools import forecast_customer_demand
    from tools.shortage_data_tools import fetch_open_orders

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    train_frame = processed_frames[config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE].head(3000)
    held_out_frame = processed_frames[config.CLEAN_WEEKLY_DEMAND_EVAL_FILE]
    schema = saved_models["demand_schema"]
    train_features = build_feature_frame(train_frame.copy(), config.DEMAND_NUMERIC_FEATURES,
                                         config.DEMAND_CATEGORICAL_FEATURES)[schema]
    held_out_features = build_feature_frame(held_out_frame.copy(), config.DEMAND_NUMERIC_FEATURES,
                                            config.DEMAND_CATEGORICAL_FEATURES)[schema]
    train_cases = train_frame[config.DEMAND_TARGET_COLUMN]
    held_out_cases = held_out_frame[config.DEMAND_TARGET_COLUMN]
    model = saved_models["demand"]
    scores = evaluate_regressor(copy.deepcopy(model), train_features, train_cases,
                                held_out_features, held_out_cases)
    missing = sorted({"held_out_r2", "train_r2", "generalisation_gap", "mean_absolute_error",
                      "average_estimate_mean_absolute_error", "lowest_prediction",
                      "highest_prediction"} - set(scores))
    assert len(missing) == 0, (
        "evaluate_regressor returned no "
        + str(missing)
        + ", so the demand model's scores cannot be read."
    )
    assert scores["held_out_r2"] >= config.MINIMUM_DEMAND_R2, (
        "The demand model scored "
        + str(scores["held_out_r2"])
        + " R squared on the held-out customer weeks, below the "
        + str(config.MINIMUM_DEMAND_R2)
        + " the configuration sets."
    )
    predictions = np.asarray(model.predict(held_out_features), dtype=float)
    measured = r2_score(held_out_cases, predictions)
    assert abs(scores["held_out_r2"] - measured) < 0.006, (
        "The evaluation reports R squared "
        + str(scores["held_out_r2"])
        + ", but the model's own forecasts for the held-out weeks give "
        + str(round(measured, 4))
        + ". evaluate_regressor scores the fitted model it is given, without fitting it again."
    )
    trained_on = r2_score(train_cases, np.asarray(model.predict(train_features), dtype=float))
    assert (abs(scores["train_r2"] - trained_on) < 0.006
            and abs(scores["generalisation_gap"]
                    - (scores["train_r2"] - scores["held_out_r2"])) < 0.011), (
        "The evaluation reports a training R squared of "
        + str(scores["train_r2"])
        + " and a gap of "
        + str(scores["generalisation_gap"])
        + ", but the model's own forecasts for the training rows it was given give "
        + str(round(trained_on, 4))
        + ". The training score is measured on the training rows and the gap is that less the "
        "held-out score."
    )
    average_error = float(np.mean(np.abs(np.asarray(held_out_cases, dtype=float)
                                         - float(np.mean(np.asarray(train_cases, dtype=float))))))
    assert abs(scores["average_estimate_mean_absolute_error"] - average_error) < 0.051, (
        "The evaluation reports that always answering the average misses by "
        + str(scores["average_estimate_mean_absolute_error"])
        + " cases, but always answering the average of the training weeks it was given misses "
        "the held-out weeks by "
        + str(round(average_error, 2))
        + " cases on average."
    )
    assert scores["mean_absolute_error"] < scores["average_estimate_mean_absolute_error"], (
        "The model misses by "
        + str(scores["mean_absolute_error"])
        + " cases on average, no better than the "
        + str(scores["average_estimate_mean_absolute_error"])
        + " cases missed by always answering the training average."
    )
    assert (abs(scores["lowest_prediction"] - float(predictions.min())) < 0.051
            and abs(scores["highest_prediction"] - float(predictions.max())) < 0.051), (
        "The evaluation reports forecasts from "
        + str(scores["lowest_prediction"])
        + " to "
        + str(scores["highest_prediction"])
        + ", but the model's forecasts for the held-out weeks run from "
        + str(round(float(predictions.min()), 2))
        + " to "
        + str(round(float(predictions.max()), 2))
        + "."
    )
    below_nought = [round(float(value), 2) for value in predictions if value < 0]
    assert len(below_nought) == 0, (
        "The saved demand model forecasts "
        + str(len(below_nought))
        + " of the "
        + str(len(predictions))
        + " held-out customer weeks below nought cases, for example "
        + str(below_nought[:3])
        + ". A customer cannot take fewer than no cases."
    )
    products = live_files["products"][config.PRODUCT_CODE_COLUMN].tolist()
    for product_code in [products[0], products[len(products) // 2], products[-1]]:
        product_orders = fetch_open_orders(live_files["orders"], product_code)
        forecasts = forecast_customer_demand(model, schema, product_orders.copy())
        customers = sorted(set(product_orders[config.ACCOUNT_CODE_COLUMN]))
        below_nought = [round(float(value), 2) for value in forecasts.values() if value < 0]
        assert list(forecasts) == customers and len(below_nought) == 0, (
            "forecast_customer_demand gave "
            + product_code
            + " forecasts for "
            + str(list(forecasts))
            + " where its open orders come from "
            + str(customers)
            + ", in code order, and "
            + str(len(below_nought))
            + " are below nought. Every customer with an open order gets one forecast of next "
            "week's cases, and none can be less than none."
        )
    base_rows = fetch_open_orders(live_files["orders"], products[0])
    base_rows = base_rows[base_rows["last_week_cases"].notna()
                          & base_rows["avg_4_week_cases"].notna()].head(2)
    planted = base_rows.iloc[[0, 0, 1, 1]].reset_index(drop=True)
    planted[config.ORDER_LINE_ID_COLUMN] = ["OL-9700001", "OL-9700002", "OL-9700003",
                                            "OL-9700004"]
    planted[config.ACCOUNT_CODE_COLUMN] = ["CUS-0901", "CUS-0901", "CUS-0902", "CUS-0902"]
    planted[config.RECEIVED_SEQUENCE_COLUMN] = [970004, 970001, 970002, 970003]
    for column_name in ["last_week_cases", "avg_4_week_cases"]:
        planted[column_name] = pd.to_numeric(planted[column_name]) + pd.Series([500, 0, 0, 500])
    expected_planted = dict(zip(["CUS-0901", "CUS-0902"], model.predict(build_feature_frame(
        planted.iloc[[1, 2]].copy(), config.DEMAND_NUMERIC_FEATURES,
        config.DEMAND_CATEGORICAL_FEATURES)[schema]).tolist()))
    planted_forecasts = forecast_customer_demand(model, schema, planted.copy())
    differing = {account_code: (planted_forecasts.get(account_code), round(expected, 4))
                 for account_code, expected in expected_planted.items()
                 if account_code not in planted_forecasts
                 or abs(float(planted_forecasts[account_code]) - expected) > 0.001}
    assert sorted(planted_forecasts) == sorted(expected_planted) and len(differing) == 0, (
        "Two planted customers each sent an order twice: CUS-0901 first as OL-9700002 and "
        "later as OL-9700001, written above it, and CUS-0902 first as OL-9700003 and later as "
        "OL-9700004, each later line with 500 cases more last week and over four weeks. "
        "forecast_customer_demand gave "
        + str(planted_forecasts)
        + ", and "
        + str(differing)
        + " (given, forecast from the first order) differ. Every customer's forecast is made "
        "from its first order line of the product in the order log, wherever it stands in the "
        "table."
    )


def test_inflated_order_score_rises_with_an_order_further_above_usual_and_one_sent_twice(
        processed_frames, saved_models):
    import numpy as np

    from config import config
    from data_pipeline.feature_builder import build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    schema = saved_models["inflated_order_schema"]
    model = saved_models["inflated_order"]
    held_out = processed_frames[config.CLEAN_ORDER_HISTORY_EVAL_FILE]
    as_placed = model.predict_proba(build_feature_frame(
        held_out.head(3000).copy(), config.INFLATED_ORDER_NUMERIC_FEATURES,
        config.INFLATED_ORDER_CATEGORICAL_FEATURES)[schema])[:, 1]
    assert float(np.std(as_placed)) > 0.05, (
        "The inflated-order model scores about the same for every order, with a spread of "
        + str(round(float(np.std(as_placed)), 4))
        + ". A score that ignores the order cannot say which orders stand above what the "
        "customer will really take."
    )
    measured = held_out[held_out["ordered_vs_usual_pct"].notna()].head(1500)
    assert len(measured) > 1000, (
        "Only "
        + str(len(measured))
        + " held-out orders carry the order as a percentage of the customer's usual week, too "
        "few to compare."
    )
    for change, stronger, weaker in [
            ("the order standing 100 percentage points further above the customer's usual week",
             measured.assign(ordered_vs_usual_pct=measured["ordered_vs_usual_pct"] + 100),
             measured),
            ("the same order also sent by another route rather than not",
             measured.assign(same_order_other_route_flag=config.BOOLEAN_YES),
             measured.assign(same_order_other_route_flag=config.BOOLEAN_NO))]:
        averages = [float(model.predict_proba(build_feature_frame(
            frame.copy(), config.INFLATED_ORDER_NUMERIC_FEATURES,
            config.INFLATED_ORDER_CATEGORICAL_FEATURES)[schema])[:, 1].mean())
            for frame in [stronger, weaker]]
        assert averages[0] > averages[1], (
            "With "
            + change
            + " the same orders average an inflated-order score of "
            + str(round(averages[0], 4))
            + ", against "
            + str(round(averages[1], 4))
            + " otherwise. An order far above the customer's usual week, or one sent twice, is "
            "the order a customer cuts once supply recovers; the model has not learned what "
            "marks an order as inflated."
        )


def test_demand_forecast_rises_with_the_four_week_average_and_with_a_promotion_next_week(
        processed_frames, saved_models):
    import numpy as np

    from config import config
    from data_pipeline.feature_builder import build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    schema = saved_models["demand_schema"]
    model = saved_models["demand"]
    weeks = processed_frames[config.CLEAN_WEEKLY_DEMAND_EVAL_FILE]
    as_recorded = model.predict(build_feature_frame(
        weeks.head(1500).copy(), config.DEMAND_NUMERIC_FEATURES,
        config.DEMAND_CATEGORICAL_FEATURES)[schema])
    assert float(np.std(as_recorded)) > 20, (
        "The demand model forecasts about the same cases for every customer week, with a "
        "spread of "
        + str(round(float(np.std(as_recorded)), 2))
        + " cases. A forecast that ignores the customer cannot rank shortages by what "
        "customers will need."
    )
    averaged = weeks[weeks["avg_4_week_cases"].notna()].head(1500)
    for change, larger, smaller in [
            ("a four-week average 20 cases higher",
             averaged.assign(avg_4_week_cases=averaged["avg_4_week_cases"] + 20), averaged),
            ("a promotion next week rather than none",
             averaged.assign(promotion_next_week_flag=config.BOOLEAN_YES),
             averaged.assign(promotion_next_week_flag=config.BOOLEAN_NO))]:
        averages = [float(np.mean(model.predict(build_feature_frame(
            frame.copy(), config.DEMAND_NUMERIC_FEATURES,
            config.DEMAND_CATEGORICAL_FEATURES)[schema]))) for frame in [larger, smaller]]
        assert averages[0] > averages[1], (
            "Shown "
            + change
            + ", the same customer weeks average a forecast of "
            + str(round(averages[0], 2))
            + " cases, against "
            + str(round(averages[1], 2))
            + " otherwise. A customer that has been taking more takes more next week, and a "
            "promotion sells more, so the model has not learned what decides next week's "
            "cases."
        )


def test_inflated_order_facts_score_and_band_every_order_by_the_configured_thresholds(
        live_files, saved_models):
    import numpy as np

    from config import config
    from data_pipeline.feature_builder import build_feature_frame
    from data_pipeline.order_cleaning import clean_order_records
    from tools.prediction_tools import build_inflated_order_facts

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    orders = clean_order_records(live_files["orders"].copy(), config.OPEN_ORDER_VALUE_MAPPINGS,
                                 config.OPEN_ORDER_WHOLE_NUMBER_PATTERNS,
                                 config.OPEN_ORDER_DECIMAL_NUMBER_PATTERNS,
                                 config.OPEN_ORDER_MINIMUM_VALUES)
    orders = orders.sample(frac=1.0, random_state=config.RANDOM_SEED)
    orders.index = [str(position) + "-row" for position in range(len(orders))]
    facts = build_inflated_order_facts(saved_models["inflated_order"],
                                       saved_models["inflated_order_schema"], orders.copy())
    assert len(facts) == len(orders), (
        "Inflated-order facts were built for "
        + str(len(facts))
        + " of the "
        + str(len(orders))
        + " open orders given, so the facts no longer line up with the orders they describe."
    )
    missing = sorted({"order_line_id", "account_code", "inflated_order_score",
                      "inflated_order_band"} - set(facts[0]))
    assert len(missing) == 0, (
        "The inflated-order facts carry no "
        + str(missing)
        + ", so the agents and the allocation page cannot read them."
    )
    features = build_feature_frame(orders.copy(), config.INFLATED_ORDER_NUMERIC_FEATURES,
                                   config.INFLATED_ORDER_CATEGORICAL_FEATURES)[
        saved_models["inflated_order_schema"]]
    probabilities = saved_models["inflated_order"].predict_proba(features)[:, 1]
    bands_seen = set()
    for position, (fact, probability) in enumerate(zip(facts, probabilities)):
        line_id = str(orders[config.ORDER_LINE_ID_COLUMN].iloc[position])
        account_code = str(orders[config.ACCOUNT_CODE_COLUMN].iloc[position])
        assert fact["order_line_id"] == line_id and fact["account_code"] == account_code, (
            "The facts in position "
            + str(position)
            + " describe "
            + str(fact["order_line_id"])
            + " where the order in that position is "
            + line_id
            + ". Facts are matched to orders by row, whatever labels the table's index carries."
        )
        assert abs(fact["inflated_order_score"] - float(probability)) < 0.00051, (
            line_id
            + " was given an inflated-order score of "
            + str(fact["inflated_order_score"])
            + " where the saved model scores it "
            + str(round(float(probability), 4))
            + "."
        )
        if fact["inflated_order_score"] >= config.INFLATED_ORDER_BAND_HIGH_THRESHOLD:
            expected_band = config.INFLATED_ORDER_BAND_HIGH
        elif fact["inflated_order_score"] >= config.INFLATED_ORDER_BAND_MEDIUM_THRESHOLD:
            expected_band = config.INFLATED_ORDER_BAND_MEDIUM
        else:
            expected_band = config.INFLATED_ORDER_BAND_LOW
        assert fact["inflated_order_band"] == expected_band, (
            line_id
            + " scored "
            + str(fact["inflated_order_score"])
            + " and was banded "
            + str(fact["inflated_order_band"])
            + " where the thresholds "
            + str(config.INFLATED_ORDER_BAND_MEDIUM_THRESHOLD)
            + " and "
            + str(config.INFLATED_ORDER_BAND_HIGH_THRESHOLD)
            + " put it in "
            + expected_band
            + ". The band is cut on the score as shown, and a score sitting on a threshold "
            "belongs to the band above it."
        )
        bands_seen.add(fact["inflated_order_band"])
    assert bands_seen == {config.INFLATED_ORDER_BAND_HIGH, config.INFLATED_ORDER_BAND_MEDIUM,
                          config.INFLATED_ORDER_BAND_LOW}, (
        "Across the open orders only "
        + str(sorted(bands_seen))
        + " occur. The queue holds orders of every band, so the scores are not separating them."
    )
    assert float(np.std(probabilities)) > 0.05, (
        "Every open order scores about the same, so the scores cannot tell an inflated order "
        "from an ordinary one."
    )


def test_request_model_clears_the_configured_macro_f1(processed_frames, saved_models):
    import copy

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from config import config
    from modeling.model_evaluation import evaluate_text_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    emails = processed_frames[config.CLEAN_CUSTOMER_EMAILS_FILE]
    train_text, held_out_text, train_answers, held_out_answers = train_test_split(
        emails[config.EMAIL_TEXT_COLUMN], emails[config.EMAIL_TARGET_COLUMN],
        test_size=config.EMAIL_HELD_OUT_SHARE, stratify=emails[config.EMAIL_TARGET_COLUMN],
        random_state=config.RANDOM_SEED)
    model = saved_models["request"]
    scores = evaluate_text_classifier(copy.deepcopy(model), train_text.head(3000),
                                      train_answers.head(3000), held_out_text, held_out_answers)
    missing = sorted({"held_out_macro_f1", "train_macro_f1", "generalisation_gap",
                      "per_class_f1"} - set(scores))
    assert len(missing) == 0, (
        "evaluate_text_classifier returned no "
        + str(missing)
        + ", so the request model's scores cannot be read."
    )
    assert scores["held_out_macro_f1"] >= config.MINIMUM_REQUEST_MACRO_F1, (
        "The request model scored "
        + str(scores["held_out_macro_f1"])
        + " macro F1 on the held-out emails, below the "
        + str(config.MINIMUM_REQUEST_MACRO_F1)
        + " the configuration sets. Macro F1 weighs the rarest request as much as the "
        "commonest, which is the point: an urgent clinical need misread as an expedite request "
        "sends the planner to the wrong guide."
    )
    measured = f1_score(held_out_answers, model.predict(held_out_text), average="macro")
    assert abs(scores["held_out_macro_f1"] - measured) < 0.006, (
        "The evaluation reports macro F1 "
        + str(scores["held_out_macro_f1"])
        + ", but the model's own predictions on the held-out emails give "
        + str(round(measured, 4))
        + ". evaluate_text_classifier scores the fitted model it is given, without fitting it "
        "again."
    )
    trained_on = f1_score(train_answers.head(3000), model.predict(train_text.head(3000)),
                          average="macro")
    assert (abs(scores["train_macro_f1"] - trained_on) < 0.006
            and abs(scores["generalisation_gap"]
                    - (scores["train_macro_f1"] - scores["held_out_macro_f1"])) < 0.011), (
        "The evaluation reports a training macro F1 of "
        + str(scores["train_macro_f1"])
        + " and a gap of "
        + str(scores["generalisation_gap"])
        + ", but the model's own predictions on the training emails it was given give "
        + str(round(trained_on, 4))
        + ". The training score is measured on the training emails and the gap is that less "
        "the held-out score."
    )
    assert sorted(scores["per_class_f1"]) == sorted(config.REQUEST_GUIDE_DOCUMENTS), (
        "The per-request scores cover "
        + str(sorted(scores["per_class_f1"]))
        + " where the emails are coded to "
        + str(sorted(config.REQUEST_GUIDE_DOCUMENTS))
        + ". A request left out is a request whose score nobody sees."
    )


def test_request_predictions_stay_inside_the_vocabulary_and_reach_every_class(processed_frames,
                                                                             saved_models):
    from sklearn.model_selection import train_test_split

    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    emails = processed_frames[config.CLEAN_CUSTOMER_EMAILS_FILE]
    split = train_test_split(
        emails[config.EMAIL_TEXT_COLUMN], emails[config.EMAIL_TARGET_COLUMN],
        test_size=config.EMAIL_HELD_OUT_SHARE, stratify=emails[config.EMAIL_TARGET_COLUMN],
        random_state=config.RANDOM_SEED)
    held_out_text, held_out_answers = split[1], split[3]
    model = saved_models["request"]
    requests = sorted(config.REQUEST_GUIDE_DOCUMENTS)
    predicted = [str(label) for label in model.predict(held_out_text)]
    strays = sorted(set(predicted) - set(requests))
    assert len(strays) == 0, (
        "The request model predicted "
        + str(strays)
        + ", which the configuration does not list. Every request downstream is looked up by "
        "name, its guide and its urgency included."
    )
    never = sorted(set(requests) - set(predicted))
    assert len(never) == 0, (
        "Across "
        + str(len(predicted))
        + " held-out emails the model never once predicted "
        + str(never)
        + ", so those customers are always given another request's guide."
    )
    for request in requests:
        given = [label for label, truth in zip(predicted, held_out_answers) if truth == request]
        commonest = max(set(given), key=given.count)
        assert commonest == request, (
            "Emails asking for "
            + request
            + " are most often read as "
            + commonest
            + ". A model that confuses a request with another more often than it names it has "
            "not learned that request."
        )
    unseen = ["Zyxwv qrstu plmok wvbnm ghjkl asdfg.", "The the the the the the the."]
    read_unseen = [str(label) for label in model.predict(unseen)]
    assert len(read_unseen) == 2 and set(read_unseen) <= set(requests), (
        "Emails made of words the model never saw were read as "
        + str(read_unseen)
        + ". An email written in new words still comes back as one of the eight requests."
    )


def test_email_cleaning_leaves_plain_words_whatever_markup_or_capitals_the_email_came_in(
        raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.email_cleaning import clean_email_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    raw_emails = raw_frames[config.RAW_CUSTOMER_EMAILS_FILE].drop_duplicates(
        subset=[config.EMAIL_ID_COLUMN])
    written = raw_emails[config.EMAIL_TEXT_COLUMN].str.strip()
    answered = ~raw_emails[config.EMAIL_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    plain_emails = raw_emails[answered & ~written.str.contains("[<>&\xa0]", regex=True)
                              & (written != written.str.upper())
                              & (written.str.split().str.len() >= 8)].head(150)
    variants = []
    for email_row in plain_emails.to_dict("records"):
        words = email_row[config.EMAIL_TEXT_COLUMN].split()
        marked_up = ("<p>" + "&nbsp;".join(words[:2]) + " <b>" + " ".join(words[2:4])
                     + "</b> <br/> " + " ".join(words[4:]).replace("'", "&#39;") + "</p>")
        email_id = str(email_row[config.EMAIL_ID_COLUMN]).strip()
        variants.append(dict(email_row, **{config.EMAIL_ID_COLUMN: "MARKED-" + email_id,
                                           config.EMAIL_TEXT_COLUMN: marked_up}))
        variants.append(dict(email_row, **{config.EMAIL_ID_COLUMN: "CAPITALS-" + email_id,
                                           config.EMAIL_TEXT_COLUMN:
                                               email_row[config.EMAIL_TEXT_COLUMN].upper()}))
    first_row = plain_emails.iloc[0].to_dict()
    first_id = str(first_row[config.EMAIL_ID_COLUMN]).strip()
    variants.append(dict(first_row, **{config.EMAIL_ID_COLUMN: "SHORT-" + first_id,
                                       config.EMAIL_TEXT_COLUMN: "Any news on our order?"}))
    variants.append(dict(first_row, **{config.EMAIL_ID_COLUMN: "SHORTMARKED-" + first_id,
                                       config.EMAIL_TEXT_COLUMN:
                                           "<p>Any news</p> <br/> <br/> <b>on our order?</b>"}))
    sample_path = str(tmp_path / "emails_raw_sample.csv")
    cleaned_path = str(tmp_path / "emails_cleaned_sample.csv")
    pd.concat([plain_emails, pd.DataFrame(variants)]).to_csv(sample_path, index=False)
    clean_email_dataset(sample_path, cleaned_path)
    cleaned_sample = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False)
    by_id = cleaned_sample.set_index(config.EMAIL_ID_COLUMN)
    changed = []
    for email_id in plain_emails[config.EMAIL_ID_COLUMN].astype(str).str.strip():
        as_written = (by_id.loc[email_id, config.EMAIL_TEXT_COLUMN] if email_id in by_id.index
                      else None)
        for prefix, wanted in [("MARKED-", as_written),
                               ("CAPITALS-", None if as_written is None else as_written.lower())]:
            found = (by_id.loc[prefix + email_id, config.EMAIL_TEXT_COLUMN]
                     if prefix + email_id in by_id.index else None)
            if found != wanted or wanted is None:
                changed.append((prefix + email_id, found, wanted))
    assert len(plain_emails) >= 100 and len(changed) == 0, (
        str(len(changed))
        + " of "
        + str(2 * len(plain_emails))
        + " emails rewritten with markup and escaped characters, or in capitals throughout, did "
        "not clean to what the same email written plainly cleans to, for example "
        + str(changed[:2])
        + ". Markup comes out, escaped characters are put back and an email written in "
        "capitals is kept in lower case, so the words the model reads do not depend on how the "
        "email was written."
    )
    for prefix in ["SHORT-", "SHORTMARKED-"]:
        assert prefix + first_id not in by_id.index, (
            "The email 'Any news on our order?', written "
            + ("plainly" if prefix == "SHORT-" else "with markup that makes it seven pieces")
            + ", was kept. An email of fewer than "
            + str(config.MINIMUM_EMAIL_WORDS)
            + " words once its markup is out is too short to learn from and is removed."
        )
    for source, emails in [("the sample just cleaned", cleaned_sample),
                           (config.CLEAN_CUSTOMER_EMAILS_FILE,
                            processed_frames[config.CLEAN_CUSTOMER_EMAILS_FILE])]:
        texts = emails[config.EMAIL_TEXT_COLUMN].astype(str)
        for description, flagged in [
                ("markup", texts.str.contains(config.MARKUP_PATTERN, regex=True)),
                ("escaped characters", texts.str.contains(config.ENCODING_ARTIFACT_PATTERN,
                                                          regex=True)),
                ("runs of whitespace or leading and trailing spaces",
                 (texts != texts.str.strip())
                 | texts.str.contains(r"\s{2,}|[\t\n\r]", regex=True)),
                ("capitals throughout", (texts == texts.str.upper())
                 & (texts != texts.str.lower())),
                ("fewer than " + str(config.MINIMUM_EMAIL_WORDS) + " words",
                 texts.str.split().str.len() < config.MINIMUM_EMAIL_WORDS),
                ("a length that disagrees with the cleaned text",
                 emails[config.EMAIL_LENGTH_COLUMN].astype(int) != texts.str.len())]:
            assert int(flagged.sum()) == 0, (
                str(int(flagged.sum()))
                + " emails in "
                + source
                + " still carry "
                + description
                + ", for example "
                + str(texts[flagged].head(1).tolist())
                + ". The model reads words, and a tag, an escape or a fragment is noise to it."
            )


def test_each_customers_most_urgent_request_is_taken_in_the_configured_order(
        live_files, processed_frames, saved_models):
    from config import config
    from tools.prediction_tools import build_request_facts
    from tools.shortage_data_tools import fetch_open_emails, fetch_open_orders

    assert live_files["problem"] is None, live_files["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    model = saved_models["request"]
    emails = live_files["emails"]
    product_code = str(emails.groupby(config.PRODUCT_CODE_COLUMN).size().idxmax())
    account_codes = sorted(set(fetch_open_orders(live_files["orders"], product_code)[
        config.ACCOUNT_CODE_COLUMN]))
    open_emails = fetch_open_emails(emails, product_code, account_codes)
    facts = build_request_facts(model, open_emails)
    assert isinstance(facts, dict) and {"email_requests", "customer_requests"} <= set(facts), (
        "build_request_facts returned "
        + str(type(facts).__name__)
        + " without email_requests and customer_requests, so the request analyst has nothing "
        "to read."
    )
    email_texts = [open_email[config.EMAIL_TEXT_COLUMN] for open_email in open_emails]
    read_types = [str(label) for label in model.predict(email_texts)]
    returned = facts["email_requests"]
    altered = [position for position, open_email in enumerate(open_emails)
               if position >= len(returned)
               or {field_name: returned[position].get(field_name)
                   for field_name in open_email} != open_email]
    found_types = [str(email_request.get("request_type")) for email_request in returned]
    assert (len(open_emails) > 10 and len(returned) == len(open_emails) and len(altered) == 0
            and found_types == read_types), (
        "Of the "
        + str(len(open_emails))
        + " open emails about "
        + product_code
        + ", build_request_facts returned "
        + str(len(returned))
        + "; those at positions "
        + str(altered[:8])
        + " lost a field they carried or moved, and the requests begin "
        + str(found_types[:6])
        + " where the saved request model reads "
        + str(read_types[:6])
        + ". Every email keeps what it carried, in the order given, and gains the request the "
        "saved model reads in it."
    )
    written_requests = {}
    for open_email, request_type in zip(open_emails, read_types):
        written_requests.setdefault(open_email[config.ACCOUNT_CODE_COLUMN], set()).add(
            request_type)
    misread = sorted(account_code for account_code, request_type
                     in facts["customer_requests"].items()
                     if request_type not in written_requests.get(account_code, set()))
    assert sorted(facts["customer_requests"]) == sorted(written_requests) and len(misread) == 0, (
        "build_request_facts gave the customers who wrote about "
        + product_code
        + " the requests "
        + str(facts["customer_requests"])
        + ", where "
        + str(sorted(written_requests))
        + " wrote, and "
        + str(misread)
        + " were given a request none of their own emails makes. Each customer that wrote is "
        "given one request, the most urgent of those its emails make."
    )
    clean_emails = processed_frames[config.CLEAN_CUSTOMER_EMAILS_FILE].head(3000)
    text_for_request = {}
    for email_text, request_type in zip(clean_emails[config.EMAIL_TEXT_COLUMN],
                                        model.predict(clean_emails[config.EMAIL_TEXT_COLUMN])):
        text_for_request.setdefault(str(request_type), email_text)
    planted = [("CUS-0901", "Forecast update"), ("CUS-0901", "Urgent clinical need"),
               ("CUS-0901", "Allocation query"), ("CUS-0902", "Delivery complaint"),
               ("CUS-0902", "Expedite request"), ("CUS-0903", "Substitution accepted"),
               ("CUS-0904", "Order cancellation"), ("CUS-0904", "Quantity increase")]
    planted_emails = []
    for position, (account_code, request_type) in enumerate(planted):
        planted_emails.append({config.EMAIL_ID_COLUMN: "EM-99000" + str(position),
                               config.ACCOUNT_CODE_COLUMN: account_code,
                               config.EMAIL_SEQUENCE_COLUMN: position + 1,
                               config.EMAIL_RECEIVED_AT_COLUMN: "Monday 09:00",
                               config.SENDER_ROLE_COLUMN: "Buyer",
                               config.EMAIL_SUBJECT_COLUMN: "Our order",
                               config.EMAIL_TEXT_COLUMN: text_for_request[request_type]})
    planted_facts = build_request_facts(model, planted_emails)
    expected_planted = {"CUS-0901": "Urgent clinical need", "CUS-0902": "Expedite request",
                        "CUS-0903": "Substitution accepted", "CUS-0904": "Order cancellation"}
    planted_types = [str(email_request.get("request_type"))
                     for email_request in planted_facts["email_requests"]]
    assert (planted_types == [request_type for account_code, request_type in planted]
            and planted_facts["customer_requests"] == expected_planted), (
        "Planted emails the saved model reads as a forecast update, an urgent clinical need and "
        "an allocation query from CUS-0901, a delivery complaint and an expedite request from "
        "CUS-0902, a substitution accepted from CUS-0903 and an order cancellation and a "
        "quantity increase from CUS-0904 were read as "
        + str(planted_types)
        + " and gave the most urgent requests "
        + str(planted_facts["customer_requests"])
        + " where the configured order gives "
        + str(expected_planted)
        + ". The most urgent request is the one earliest in the configuration's order, never "
        "the first or the last email."
    )
    nothing = build_request_facts(model, [])
    assert nothing == {"email_requests": [], "customer_requests": {}}, (
        "With no open emails build_request_facts returned "
        + str(nothing)
        + " where there is nothing to read: no email requests and no customer's request."
    )


def test_emails_are_read_for_one_product_in_the_order_they_were_received(live_files):
    import numbers

    import pandas as pd

    from config import config
    from tools.shortage_data_tools import fetch_open_emails

    assert live_files["problem"] is None, live_files["problem"]
    emails = live_files["emails"]
    orders = live_files["orders"]
    counts = emails.groupby(config.PRODUCT_CODE_COLUMN).size().sort_values(ascending=False)
    named_products = dict(zip(emails[config.EMAIL_ID_COLUMN], emails[config.PRODUCT_CODE_COLUMN]))
    for product_code in list(counts.index[:6]):
        account_codes = sorted(set(orders.loc[orders[config.PRODUCT_CODE_COLUMN] == product_code,
                                              config.ACCOUNT_CODE_COLUMN].str.strip()))
        account_codes = [account_code for account_code in account_codes
                         if account_code.startswith("CUS-")][:6]
        read = fetch_open_emails(emails.sample(frac=1.0, random_state=config.RANDOM_SEED),
                                 product_code, account_codes)
        unreadable = [open_email for open_email in read
                      if sorted(open_email) != sorted(config.OPEN_EMAIL_FIELDS)
                      or not isinstance(open_email[config.EMAIL_SEQUENCE_COLUMN],
                                        numbers.Integral)]
        assert len(unreadable) == 0, (
            "The open emails of "
            + product_code
            + " came back carrying "
            + str(unreadable[0] if unreadable else [])[:300]
            + " where each carries exactly "
            + str(config.OPEN_EMAIL_FIELDS)
            + ", its place among its customer's emails as a whole number."
        )
        strays = [open_email[config.EMAIL_ID_COLUMN] for open_email in read
                  if open_email[config.ACCOUNT_CODE_COLUMN] not in account_codes
                  or not pd.isna(named_products.get(open_email[config.EMAIL_ID_COLUMN], ""))
                  and named_products.get(open_email[config.EMAIL_ID_COLUMN]) != product_code]
        places = [(str(open_email[config.ACCOUNT_CODE_COLUMN]),
                   int(open_email[config.EMAIL_SEQUENCE_COLUMN])) for open_email in read]
        assert len(strays) == 0 and places == sorted(places) and len(set(places)) == len(places), (
            "The open emails of "
            + product_code
            + " from "
            + str(account_codes)
            + " came back as "
            + str([open_email[config.EMAIL_ID_COLUMN] for open_email in read])[:600]
            + ", "
            + str(strays[:5])
            + " of them from another customer or about another product, in the places "
            + str(places)[:300]
            + ". Only the emails about the product, or naming no product, from the customers "
            "given are read, and a later email answers an earlier one, so each customer's are "
            "put back in the order that customer sent them, customers in code order."
        )
    product_code = str(counts.index[0])
    account_code = str(emails.loc[emails[config.PRODUCT_CODE_COLUMN] == product_code,
                                  config.ACCOUNT_CODE_COLUMN].iloc[0])
    own_rows = emails[emails[config.PRODUCT_CODE_COLUMN] == product_code].head(3).copy()
    own_rows[config.ACCOUNT_CODE_COLUMN] = account_code
    own_rows[config.EMAIL_SEQUENCE_COLUMN] = ["10", "2", "1"]
    own_rows[config.EMAIL_TEXT_COLUMN] = [
        "Tenth email from us about this product, sent last of all.",
        "Second email from us about this product, <b>sent</b> next&nbsp;of all.",
        "FIRST EMAIL FROM US ABOUT THIS PRODUCT, SENT FIRST OF ALL."]
    own_rows[config.EMAIL_RECEIVED_AT_COLUMN] = ["Monday 10:00", "TUE 09:30", "wed 08:15"]
    ordered = fetch_open_emails(own_rows, product_code, [account_code])
    found = [(open_email[config.EMAIL_TEXT_COLUMN], open_email[config.EMAIL_RECEIVED_AT_COLUMN])
             for open_email in ordered]
    expected = [("first email from us about this product, sent first of all.", "Wednesday 08:15"),
                ("Second email from us about this product, sent next of all.", "Tuesday 09:30"),
                ("Tenth email from us about this product, sent last of all.", "Monday 10:00")]
    assert found == expected, (
        "Emails numbered 10, 2 and 1, the second with markup and the first in capitals, came "
        "back as "
        + str(found)
        + " where they are "
        + str(expected)
        + ". The place in the customer's emails is a number, and 10 comes after 2; each live "
        "email is cleaned as the training emails were."
    )
    planted_rows = own_rows.iloc[[0] * 8].reset_index(drop=True)
    planted_rows[config.EMAIL_ID_COLUMN] = ["EM-98000" + str(number) for number in range(1, 9)]
    planted_rows[config.ACCOUNT_CODE_COLUMN] = ["CUS-0034", "CUS-0034", "CUS-0034", "CUS-0034",
                                                "CUS-0021", "CUS-0021", "CUS-0099", "CUS-0034"]
    planted_rows[config.PRODUCT_CODE_COLUMN] = [product_code, None, product_code,
                                                str(counts.index[1]), product_code, product_code,
                                                product_code, product_code]
    planted_rows[config.EMAIL_SEQUENCE_COLUMN] = ["3", "2", "11", "4", "2", "1", "1", "1"]
    planted_read = [open_email[config.EMAIL_ID_COLUMN] for open_email in fetch_open_emails(
        planted_rows, product_code, ["CUS-0034", "CUS-0021"])]
    planted_expected = ["EM-980006", "EM-980005", "EM-980008", "EM-980002", "EM-980001",
                        "EM-980003"]
    assert planted_read == planted_expected, (
        "Eight planted emails about "
        + product_code
        + ", asked for CUS-0034 and CUS-0021: EM-980001 to EM-980004 from CUS-0034 numbered 3, "
        "2, 11 and 4, the second naming no product and the fourth naming "
        + str(counts.index[1])
        + ", EM-980005 and EM-980006 from CUS-0021 numbered 2 and 1, EM-980007 from CUS-0099 "
        "and EM-980008 from CUS-0034 numbered 1, came back as "
        + str(planted_read)
        + " where they are "
        + str(planted_expected)
        + ". The given customers' emails about the product or naming no product are read, "
        "customers in code order and each customer's by its place read as a number."
    )
    shared_place = own_rows.assign(**{config.EMAIL_SEQUENCE_COLUMN: ["1", "2", "2"]})
    blank_text = own_rows.assign(**{config.EMAIL_TEXT_COLUMN: [
        None, "Second email from us about this product, sent next of all.",
        "First email from us about this product, sent first of all."]})
    unread_code = own_rows.assign(**{config.ACCOUNT_CODE_COLUMN: "C" + account_code[-4:]})
    blank_sender = own_rows.assign(**{config.SENDER_ROLE_COLUMN: [None, "Buyer", "Buyer"]})
    for description, frame, asked_codes in [
            ("emails sharing a place in the customer's emails", shared_place, [account_code]),
            ("an email whose text is blank", blank_text, [account_code]),
            ("an email whose account code is not canonical", unread_code, [account_code]),
            ("an email with no sender", blank_sender, [account_code])]:
        refused = False
        try:
            fetch_open_emails(frame, product_code, asked_codes)
        except ValueError:
            refused = True
        assert refused, (
            "fetch_open_emails did not raise a ValueError for "
            + description
            + ". A review opened on emails that cannot be read for certain has to stop before it "
            "spends a language model call, and the service turns that ValueError into its "
            "refusal."
        )


def test_every_document_gives_up_its_text(document_texts, tmp_path):
    import os
    import shutil

    import pypdf

    from config import config
    from knowledge.pdf_extraction import extract_text_from_directory

    assert document_texts["problem"] is None, document_texts["problem"]
    in_folder = sorted(name for name in os.listdir(config.DOCS_DIRECTORY)
                       if name.lower().endswith(".pdf"))
    documents = document_texts["documents"]
    assert len(documents) > 0 and {"document_name", "document_text"} <= set(documents[0]), (
        "Reading the document folder returned "
        + str(len(documents))
        + " entries"
        + ("" if len(documents) == 0 else ", the first keyed " + str(sorted(documents[0])))
        + ", where every entry names its document and carries its text."
    )
    names = sorted(document["document_name"] for document in documents)
    assert names == in_folder, (
        "Reading the document folder returned "
        + str(names)
        + " where the folder holds "
        + str(in_folder)
        + ". A document never read is a question answered from general knowledge instead."
    )
    for document in documents:
        assert len(document["document_text"].split()) > 300, (
            document["document_name"]
            + " gave up only "
            + str(len(document["document_text"].split()))
            + " words. A PDF that will not give up its text is indexed as an empty passage."
        )
    total_words = sum(len(document["document_text"].split()) for document in documents)
    assert total_words >= 10000, (
        "The "
        + str(len(in_folder))
        + " documents gave up "
        + str(total_words)
        + " words between them, where every page read in full gives well over 10,000. A reader "
        "that stops early loses the later sections, and a rule in them is never read."
    )
    for required in (list(config.CATEGORY_POLICY_DOCUMENTS.values())
                     + list(config.REQUEST_GUIDE_DOCUMENTS.values())):
        assert required in names, (
            required
            + " was not read, and the configuration sends questions to it by name."
        )
    folder = tmp_path / "mixed_folder"
    folder.mkdir()
    policy_document = config.CATEGORY_POLICY_DOCUMENTS["Laundry"]
    guide_document = config.REQUEST_GUIDE_DOCUMENTS["Expedite request"]
    for name in [policy_document, guide_document]:
        shutil.copy(os.path.join(config.DOCS_DIRECTORY, name), str(folder / name))
    (folder / "reading-list.txt").write_text("Not a document of the supply planning team.")
    (folder / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (folder / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    mixed = extract_text_from_directory(str(folder))
    assert sorted(entry["document_name"] for entry in mixed) == sorted([policy_document,
                                                                        guide_document]), (
        "A folder holding two PDFs, a text file, an image and a hidden folder file was read as "
        + str(sorted(entry["document_name"] for entry in mixed))
        + ". Only the PDFs are the team's documents; anything else in the folder is passed over."
    )
    blank_folder = tmp_path / "blank_page_folder"
    blank_folder.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_page(pypdf.PdfReader(os.path.join(config.DOCS_DIRECTORY,
                                                 policy_document)).pages[0])
    writer.add_blank_page()
    with open(str(blank_folder / "Policy-With-A-Blank-Page.pdf"), "wb") as blank_file:
        writer.write(blank_file)
    refused = False
    try:
        extract_text_from_directory(str(blank_folder))
    except ValueError:
        refused = True
    assert refused, (
        "A PDF whose second page gives up no text was read without a ValueError. A page that "
        "cannot be read may be the one that states a rule, so the document is refused rather "
        "than indexed with a hole in it."
    )


def test_documents_are_cut_into_passages_of_about_the_configured_size_without_page_headers(
        document_passages):
    import re

    from config import config
    from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

    assert document_passages["problem"] is None, document_passages["problem"]
    documents = document_passages["documents"]
    sections = document_passages["sections"]
    chunks = document_passages["chunks"]
    assert len(chunks) > 0 and {"chunk_id", "chunk_text", "document_name"} <= set(chunks[0]), (
        "Cutting the documents returned "
        + str(len(chunks))
        + " passages"
        + ("" if len(chunks) == 0 else ", the first keyed " + str(sorted(chunks[0])))
        + ", where every passage carries its identifier, its text and its document."
    )
    assert len(sections) > 5 * len(documents), (
        str(len(documents))
        + " documents split into "
        + str(len(sections))
        + " sections. Each document is written in numbered sections, and a section is the unit "
        "a question is usually about."
    )
    with_header = [chunk["chunk_id"] for chunk in chunks
                   if re.search(config.DOCUMENT_HEADER_PATTERN, chunk["chunk_text"])]
    assert len(with_header) == 0, (
        "These passages still carry the document header: "
        + str(with_header[:5])
        + ". The header repeats the same words in every document, so it matches every question "
        "equally and pushes the passages that answer it out of the results."
    )
    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    average = sum(lengths) / len(lengths)
    assert config.CHUNK_SIZE_CHARACTERS / 5 <= average <= config.CHUNK_SIZE_CHARACTERS, (
        "Passages average "
        + str(int(average))
        + " characters where the configuration asks for passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + ". Passages far longer return a whole document for every question; far shorter lose "
        "the sentence that gave them their meaning."
    )
    assert max(lengths) <= 2 * config.CHUNK_SIZE_CHARACTERS, (
        "The longest passage runs to "
        + str(max(lengths))
        + " characters, far beyond the "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + " the configuration asks for."
    )
    identifiers = [chunk["chunk_id"] for chunk in chunks]
    assert len(identifiers) == len(set(identifiers)), (
        "Passage identifiers repeat, so storing them would overwrite one passage with another."
    )
    words_by_document = {document["document_name"]: set(document["document_text"].split())
                         for document in documents}
    for chunk in chunks:
        words = chunk["chunk_text"].split()
        foreign = [word for word in words
                   if word not in words_by_document.get(chunk["document_name"], set())]
        assert len(words) > 0, (
            "Passage "
            + str(chunk["chunk_id"])
            + " from "
            + str(chunk["document_name"])
            + " holds no words, and a passage with no text can answer nothing."
        )
        assert len(foreign) <= len(words) // 10, (
            "Passage "
            + str(chunk["chunk_id"])
            + " says it comes from "
            + str(chunk["document_name"])
            + " but "
            + str(len(foreign))
            + " of its "
            + str(len(words))
            + " words are not in that document, for example "
            + str(foreign[:5])
            + ". A passage has to cite the document it was cut from."
        )
    rule_patterns = [config.HISTORY_SENTENCE_PATTERN, config.CAP_SENTENCE_PATTERN,
                     config.PROTECTED_SENTENCE_PATTERN, config.NO_PROTECTION_SENTENCE_PATTERN,
                     config.RESERVE_SENTENCE_PATTERN, config.NO_RESERVE_SENTENCE_PATTERN]
    for document in documents:
        if document["document_name"] not in config.CATEGORY_POLICY_DOCUMENTS.values():
            continue
        flowing_text = re.sub(r"\s+", " ", document["document_text"])
        own_chunks = [re.sub(r"\s+", " ", chunk["chunk_text"]) for chunk in chunks
                      if chunk["document_name"] == document["document_name"]]
        stated = []
        for pattern in rule_patterns:
            for found in re.finditer(pattern, flowing_text):
                stated.append(found.group(0))
        broken = []
        for sentence in stated:
            if not any(sentence in chunk_text for chunk_text in own_chunks):
                broken.append(sentence)
        assert len(stated) == 4, (
            str(len(stated))
            + " rule sentences that the patterns in config read were found in the text "
            "extracted from "
            + document["document_name"]
            + ", where the policy states each of its four rules once, in a sentence of its own, "
            "so the extraction has broken or repeated them."
        )
        assert len(broken) == 0, (
            "These rule sentences of "
            + document["document_name"]
            + " are not whole inside any one of its passages: "
            + str(broken[:3])
            + ". The weeks of history, the cap, the protected channels or the reserve cut "
            "across two passages can no longer be read from either."
        )

    header = ("Document Allocation 99 | Issue 1 | Owner: Supply Planning | Review cycle: every "
              "six months")
    lines = []
    for first in "abcdefgh":
        for second in "abcdefg":
            lines.append("Reading " + first + second + " of the overlap check names break "
                         + first + second + ".")
    synthetic = [{"document_name": "Overlap-Check.pdf",
                  "document_text": "WRENFIELD HYGIENE\nOverlap Check\n" + header
                                   + "\n1. Overlap check\n" + " ".join(lines)}]
    pieces = [str(chunk["chunk_text"]) for chunk in build_chunk_records(
        split_documents_into_sections(synthetic))]
    assert len(pieces) >= 3, (
        "One numbered section of "
        + str(len(" ".join(lines)))
        + " characters was cut into "
        + str(len(pieces))
        + " passages, where passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + " characters need at least three."
    )
    unshared = [position for position in range(1, len(pieces))
                if pieces[position].strip()[:20] not in pieces[position - 1]]
    assert len(unshared) == 0, (
        "Passages "
        + str(unshared)
        + " of one long section open with text the passage before them does not carry. "
        "Neighbouring passages of a section overlap by up to "
        + str(config.CHUNK_OVERLAP_CHARACTERS)
        + " characters, so a sentence cut at a boundary is still whole in one of them."
    )
    for description, document_text in [
            ("no header line", "WRENFIELD HYGIENE\n1. Overlap check\n" + " ".join(lines)),
            ("two header lines", header + "\n" + header + "\n1. Overlap check\n"
             + " ".join(lines)),
            ("sections numbered from 2", header + "\n2. Overlap check\n" + " ".join(lines))]:
        refused = False
        try:
            split_documents_into_sections([{"document_name": "Broken-Layout.pdf",
                                            "document_text": document_text}])
        except ValueError:
            refused = True
        assert refused, (
            "A document with "
            + description
            + " was split without a ValueError. Every document carries one header line and "
            "sections numbered from 1, so a document laid out otherwise has been read wrongly "
            "and is refused."
        )


def test_the_index_holds_every_passage_once_with_the_document_it_came_from(
        knowledge_collection, document_passages):
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    documents = document_passages["documents"]
    chunks = document_passages["chunks"]
    stored = collection.get(include=["metadatas", "documents"])
    assert collection.count() == len(chunks) == len(stored["ids"]), (
        "The vector store holds "
        + str(collection.count())
        + " passages where cutting the documents gives "
        + str(len(chunks))
        + ". More means the index was added to twice and repeats itself in every result; fewer "
        "means passages no search can reach."
    )
    assert len(set(stored["ids"])) == len(stored["ids"]), (
        "Stored passage identifiers repeat, so one passage has overwritten another and its text "
        "can no longer be found."
    )
    names = {document["document_name"] for document in documents}
    cited = [(metadata or {}).get("document_name") for metadata in stored["metadatas"]]
    unnamed = [value for value in cited if value not in names]
    assert len(unnamed) == 0, (
        str(len(unnamed))
        + " stored passages name no document from the folder, for example "
        + str(unnamed[:3])
        + ". The name is what lets an answer cite its source and a policy be read from its own "
        "document alone."
    )
    by_document = {}
    for chunk in chunks:
        by_document[chunk["document_name"]] = by_document.get(chunk["document_name"], 0) + 1
    stored_by_document = {name: cited.count(name) for name in names}
    assert stored_by_document == by_document, (
        "Passages stored per document "
        + str(stored_by_document)
        + " differ from passages cut per document "
        + str(by_document)
        + ". A document with passages missing answers less than it says."
    )
    assert sorted(stored["documents"]) == sorted(chunk["chunk_text"] for chunk in chunks), (
        "The texts in the vector store are not the passages the documents are cut into, so a "
        "rule read from the store is not the rule the policy states."
    )


def test_a_search_narrowed_to_one_document_reads_only_that_document(knowledge_collection,
                                                                     tmp_path):
    from config import config
    from knowledge.vector_index import (build_vector_index, find_nearest_passage,
                                        open_vector_index, read_document_chunks,
                                        search_documents)

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    question = "How much may a customer have, and which customers are served first?"
    planted = {
        "Planted-Policy.pdf": [
            "Who decides. A Supply Planning Manager who did not open the review decides how the "
            "cases are released.",
            "What a customer may have and who is served first. A customer may have up to a cap "
            "on its usual demand, and protected customers are served before the rest.",
            "Questions about this policy. The supply planning team answers them.",
        ],
        "Planted-Notice.pdf": [
            question,
            "Telling customers. The customer service team sends the allocation notice.",
        ],
        "Planted-Canteen.pdf": ["The canteen at the depot opens at seven and closes at three."],
    }
    chunk_records = []
    for document_name, texts in planted.items():
        for position, text in enumerate(texts):
            chunk_records.append({"chunk_id": document_name + "-" + str(position),
                                  "chunk_text": text, "document_name": document_name})
    store_path = str(tmp_path / "planted_passages")
    stored_count = build_vector_index(chunk_records, store_path)
    assert stored_count == len(chunk_records), (
        "build_vector_index stored "
        + str(stored_count)
        + " passages of the "
        + str(len(chunk_records))
        + " it was given, so it does not report what the store holds."
    )
    collection = open_vector_index(store_path)
    whole = read_document_chunks(collection, "Planted-Policy.pdf")
    assert sorted(whole) == sorted(planted["Planted-Policy.pdf"]), (
        "Reading every passage of a planted policy of three passages returned "
        + str(whole)[:300]
        + ". A policy is read whole, every passage of it and nothing from another document, "
        "never from the few passages a search ranks first."
    )
    passage = find_nearest_passage(collection, question, "Planted-Policy.pdf")
    assert passage in planted["Planted-Policy.pdf"], (
        "Asked "
        + repr(question)
        + " of a planted policy alone, while another planted document holds the question word "
        "for word, the search returned "
        + repr(str(passage)[:100])
        + ", which is not a passage of the policy. A search narrowed to one document reads that "
        "document only: the policies share their headings and most of their sentences, and a "
        "passage from another states another category's rules."
    )
    for own_passage in planted["Planted-Policy.pdf"]:
        found = find_nearest_passage(collection, own_passage, "Planted-Policy.pdf")
        assert found == own_passage, (
            "Searching the planted policy with the words of its own passage "
            + repr(own_passage[:60])
            + " returned "
            + repr(str(found)[:60])
            + ". Nothing is nearer to a passage's own words than that passage, so the search "
            "does not rank the document's passages by how close they are to the question."
        )
    for reader, arguments in [(read_document_chunks, (collection, "No-Such-Policy.pdf")),
                              (find_nearest_passage, (collection, question,
                                                      "No-Such-Policy.pdf"))]:
        refused = False
        try:
            reader(*arguments)
        except ValueError:
            refused = True
        assert refused, (
            reader.__name__
            + " did not raise a ValueError for a document the index does not hold. An empty "
            "answer would be taken for a document that states nothing."
        )
    questions = ["In past shortages, how were the cases divided and which customers went short?",
                 "What does the planner check for an urgent clinical need?"]
    results = search_documents(knowledge_collection["collection"], questions)
    assert len(results) == len(questions), (
        "Searching with "
        + str(len(questions))
        + " questions returned "
        + str(len(results))
        + " result lists, one per question."
    )
    for asked, passages in zip(questions, results):
        assert len(passages) == config.CHAT_RETRIEVAL_RESULT_COUNT and all(
                {"chunk_id", "chunk_text", "document_name", "distance"} <= set(found)
                for found in passages), (
            "Asking "
            + repr(asked)
            + " returned "
            + str(len(passages))
            + " passages, the first keyed "
            + str(sorted(passages[0]) if passages else [])
            + ", where "
            + str(config.CHAT_RETRIEVAL_RESULT_COUNT)
            + " are returned, each with its identifier, its text, its document and its distance."
        )
        distances = [found["distance"] for found in passages]
        assert distances == sorted(distances), (
            "Passages came back in the order "
            + str(distances)
            + ", not closest first. The first passage is the one an answer is grounded on."
        )
        excluded = sorted({found["document_name"] for found in passages}
                          & set(config.CHAT_EXCLUDED_DOCUMENTS))
        assert len(excluded) == 0, (
            "Asking "
            + repr(asked)
            + " returned passages of "
            + str(excluded)
            + ". The casebook names past shortages and the customers in them and must never "
            "reach an answer, even to a question about past shortages."
        )


def test_the_planner_checks_customer_advice_and_briefing_guidance_come_from_their_documents(
        knowledge_collection):
    from config import config
    from knowledge.vector_index import find_nearest_passage, read_document_chunks
    from tools.retrieval_tools import (read_customer_advice, read_planner_checks,
                                       read_sales_briefing_guidance)

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    collection = knowledge_collection["collection"]
    for reader, question, documents in [
            (read_planner_checks, config.PLANNER_CHECKS_QUESTION, config.REQUEST_GUIDE_DOCUMENTS),
            (read_customer_advice, config.CUSTOMER_ADVICE_QUESTION,
             config.CATEGORY_POLICY_DOCUMENTS),
            (read_sales_briefing_guidance, config.SALES_BRIEFING_QUESTION,
             config.CATEGORY_POLICY_DOCUMENTS)]:
        passages = {}
        for subject, document_name in documents.items():
            passage = reader(collection, subject)
            own = read_document_chunks(collection, document_name)
            nearest = find_nearest_passage(collection, question, document_name)
            assert passage in own, (
                reader.__name__
                + " for "
                + subject
                + " returned "
                + repr(str(passage)[:80])
                + ", which is not a passage of "
                + document_name
                + ". Every request type has its own guide and every category its own policy, "
                "and a passage from another tells the planner, the customer or the sales team "
                "the wrong thing."
            )
            assert passage == nearest, (
                reader.__name__
                + " for "
                + subject
                + " returned a passage of "
                + document_name
                + " that is not the one find_nearest_passage finds closest to the configured "
                "question "
                + repr(question)
                + ". The passage is the document's answer to that question, not any passage of "
                "it."
            )
            passages[subject] = passage
        assert len(set(passages.values())) == len(documents), (
            reader.__name__
            + " gave two subjects the same passage word for word, so at least one was answered "
            "with text its document shares with others rather than with its own section."
        )
    for reader in [read_planner_checks, read_customer_advice, read_sales_briefing_guidance]:
        refused = False
        try:
            reader(collection, "Air care")
        except ValueError:
            refused = True
        assert refused, (
            reader.__name__
            + " did not raise a ValueError for 'Air care', which the configuration gives no "
            "document. A passage from some other document would be passed off as its own."
        )


def test_each_category_is_read_from_its_own_policy(planted_policies, knowledge_collection):
    from config import config
    from knowledge.policy_reading import read_allocation_policy

    assert planted_policies["problem"] is None, planted_policies["problem"]
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    for store_name, collection in [("the planted store", planted_policies["valid"]),
                                   ("the knowledge index", knowledge_collection["collection"])]:
        fingerprints = {}
        for category, document_name in config.CATEGORY_POLICY_DOCUMENTS.items():
            reading_problem = None
            try:
                reading = read_allocation_policy(collection, category)
            except ValueError as error:
                reading_problem = str(error)[:300]
            assert reading_problem is None, (
                "Reading the policy for "
                + category
                + " from "
                + store_name
                + " raised a ValueError: "
                + str(reading_problem)
                + ". The policy states each rule once in its own document; a reader that also "
                "takes passages from another document of the store, such as a superseded notice, "
                "finds rules that are not the policy's."
            )
            missing = sorted({"document_name", "history_weeks", "cap", "protected_channels",
                              "reserve_percent", "rule_sentences"} - set(reading or {}))
            assert len(missing) == 0, (
                "Reading the policy for "
                + category
                + " from "
                + store_name
                + " returned no "
                + str(missing)
                + ", so the division has nothing to apply that rule with."
            )
            assert reading["document_name"] == document_name, (
                "The policy for "
                + category
                + " was read from "
                + repr(reading["document_name"])
                + " where the configuration names "
                + document_name
                + ". Every rule belongs to the category whose own policy states it."
            )
            fingerprints[category] = str([reading["history_weeks"], str(reading["cap"]),
                                          sorted(reading["protected_channels"]),
                                          reading["reserve_percent"]])
            if store_name == "the planted store":
                decoy = planted_policies["decoy"]
                borrowed = (reading["history_weeks"] == decoy["history_weeks"]
                            or reading["reserve_percent"] == decoy["reserve_percent"]
                            or sorted(reading["protected_channels"])
                            == decoy["protected_channels"])
                assert not borrowed, (
                    "Reading the planted policy for "
                    + category
                    + " picked up "
                    + str(reading)[:300]
                    + ", which only a superseded notice in the same store states. A policy is "
                    "read from its own document, never from whichever passage of the store "
                    "mentions the same rule."
                )
        assert len(set(fingerprints.values())) == len(fingerprints), (
            "Two categories were read with identical rules from "
            + store_name
            + ": "
            + str(fingerprints)
            + ". Each policy states its own rules, so one was read for another."
        )
    refused = False
    try:
        read_allocation_policy(planted_policies["valid"], "Air care")
    except ValueError:
        refused = True
    assert refused, (
        "Reading the policy for 'Air care', a category the configuration gives no policy, did "
        "not raise a ValueError. Some other category's rules would be applied to it."
    )


def test_the_weeks_of_history_and_the_cap_are_read_from_the_policys_own_sentences(
        planted_policies):
    from fractions import Fraction

    from knowledge.policy_reading import read_allocation_policy

    assert planted_policies["problem"] is None, planted_policies["problem"]
    for category, expected in planted_policies["expected"].items():
        reading_problem = None
        try:
            reading = read_allocation_policy(planted_policies["valid"], category)
        except ValueError as error:
            reading_problem = str(error)[:300]
        assert reading_problem is None, (
            "Reading the planted policy for "
            + category
            + ", which states each rule once, raised a ValueError: "
            + str(reading_problem)
            + ". A policy is read from its own document only, so a superseded notice elsewhere "
            "in the store neither changes nor contradicts its rules."
        )
        found = reading if isinstance(reading, dict) else {}
        sentences = (found["rule_sentences"] if isinstance(found.get("rule_sentences"), dict)
                     else {})
        weeks = found.get("history_weeks")
        assert type(weeks) is int and weeks == expected["history_weeks"], (
            "From a store in which the policy for "
            + category
            + " works out usual demand over the last "
            + str(expected["history_weeks"])
            + " weeks, history_weeks was read as "
            + repr(weeks)
            + ". It is the whole number of weeks the policy's own sentence writes, wherever in "
            "the policy that sentence stands."
        )
        assert found.get("cap") == Fraction(expected["cap"]), (
            "From a store in which the policy for "
            + category
            + " caps an order at "
            + expected["cap"]
            + " times the customer's usual demand, cap was read as "
            + repr(found.get("cap"))
            + ". The cap is kept exactly as written, so the eligible quantity can be worked out "
            "exactly: at a cap of 1.4, 55 cases over seven weeks is exactly 11, and in floating "
            "point 10.999999999999998."
        )
        for rule_name in ["history", "cap"]:
            assert sentences.get(rule_name) == planted_policies["sentences"][category][rule_name], (
                "The "
                + rule_name
                + " rule of "
                + category
                + " was read from the sentence "
                + repr(sentences.get(rule_name))
                + " where the policy states "
                + repr(planted_policies["sentences"][category][rule_name])
                + ". The sentence a rule was read from is kept with it, so the notice and the "
                "page can quote the rule as the policy states it."
            )


def test_the_protected_channels_are_read_or_else_that_no_customer_is_protected(planted_policies):
    from knowledge.policy_reading import read_allocation_policy

    assert planted_policies["problem"] is None, planted_policies["problem"]
    for category, expected in planted_policies["expected"].items():
        reading_problem = None
        try:
            reading = read_allocation_policy(planted_policies["valid"], category)
        except ValueError as error:
            reading_problem = str(error)[:300]
        assert reading_problem is None, (
            "Reading the planted policy for "
            + category
            + ", which states each rule once, raised a ValueError: "
            + str(reading_problem)
            + ". A policy is read from its own document only, so a superseded notice elsewhere "
            "in the store neither changes nor contradicts its rules."
        )
        found = reading if isinstance(reading, dict) else {}
        sentences = (found["rule_sentences"] if isinstance(found.get("rule_sentences"), dict)
                     else {})
        channels = found.get("protected_channels")
        assert isinstance(channels, list) and sorted(channels) == sorted(
                expected["protected_channels"]), (
            "From a store in which the policy for "
            + category
            + (" protects the " + " and ".join(expected["protected_channels"]) + " channels"
               if expected["protected_channels"] else " protects no customer")
            + ", protected_channels was read as "
            + repr(channels)
            + ". They are Wrenfield Hygiene's channel names, every one the sentence writes, an "
            "empty list where the policy protects no customer, and never another category's."
        )
        assert sentences.get("protected") == planted_policies["sentences"][category]["protected"], (
            "The protection rule of "
            + category
            + " was read from "
            + repr(sentences.get("protected"))
            + " where the policy states "
            + repr(planted_policies["sentences"][category]["protected"])
            + "."
        )


def test_the_reserve_is_read_or_else_that_no_reserve_is_held(planted_policies):
    from knowledge.policy_reading import read_allocation_policy

    assert planted_policies["problem"] is None, planted_policies["problem"]
    for category, expected in planted_policies["expected"].items():
        reading_problem = None
        try:
            reading = read_allocation_policy(planted_policies["valid"], category)
        except ValueError as error:
            reading_problem = str(error)[:300]
        assert reading_problem is None, (
            "Reading the planted policy for "
            + category
            + ", which states each rule once, raised a ValueError: "
            + str(reading_problem)
            + ". A policy is read from its own document only, so a superseded notice elsewhere "
            "in the store neither changes nor contradicts its rules."
        )
        found = reading if isinstance(reading, dict) else {}
        sentences = (found["rule_sentences"] if isinstance(found.get("rule_sentences"), dict)
                     else {})
        percent = found.get("reserve_percent")
        assert (percent is None if expected["reserve_percent"] is None
                else type(percent) is int and percent == expected["reserve_percent"]), (
            "From a store in which the policy for "
            + category
            + (" holds back up to " + str(expected["reserve_percent"]) + " percent as a reserve"
               if expected["reserve_percent"] is not None else " holds no reserve")
            + ", reserve_percent was read as "
            + repr(percent)
            + ". It is the whole percentage the policy's own sentence writes, or None where the "
            "policy holds no reserve, which closes RELEASE WITH RESERVE; never nought, which "
            "would offer a reserve of no cases."
        )
        assert sentences.get("reserve") == planted_policies["sentences"][category]["reserve"], (
            "The reserve rule of "
            + category
            + " was read from "
            + repr(sentences.get("reserve"))
            + " where the policy states "
            + repr(planted_policies["sentences"][category]["reserve"])
            + "."
        )


def test_a_policy_that_states_a_rule_two_ways_is_refused(planted_policies):
    from config import config
    from knowledge.policy_reading import read_allocation_policy

    assert planted_policies["problem"] is None, planted_policies["problem"]
    for category, contradiction in planted_policies["contradictions"].items():
        valid_problem = None
        try:
            read_allocation_policy(planted_policies["valid"], category)
        except ValueError as error:
            valid_problem = str(error)[:200]
        assert valid_problem is None, (
            "The planted policy for "
            + category
            + " that states each rule once was refused with "
            + repr(valid_problem)
            + ". Only a policy that states a rule two ways is refused; a reader that refuses "
            "every policy refuses nothing in particular."
        )
        message = ""
        try:
            read_allocation_policy(planted_policies["contradictory"], category)
        except ValueError as error:
            message = str(error)
        assert message, (
            "A policy for "
            + category
            + " that states "
            + contradiction
            + " was read without a ValueError. A policy that contradicts itself is refused, "
            "never settled by picking one of its sentences: either reading may be the wrong one, "
            "and every customer's allocation would be decided by it."
        )
        assert config.CATEGORY_POLICY_DOCUMENTS[category] in message, (
            "Refusing the contradictory policy for "
            + category
            + " said "
            + repr(message[:200])
            + ", which does not name "
            + config.CATEGORY_POLICY_DOCUMENTS[category]
            + ". The refusal says which document has to be put right."
        )


def test_usual_demand_is_worked_out_exactly_over_the_policys_weeks_a_silent_week_counting_nought(
        planted_shortages):
    from fractions import Fraction

    assert planted_shortages["problem"] is None, planted_shortages["problem"]
    for label, shortage in planted_shortages["shortages"].items():
        assert shortage["reading_problem"] is None, shortage["reading_problem"]
        assert shortage["usual_problem"] is None, shortage["usual_problem"]
        computed = shortage["computed_usual"]
        expected = shortage["usual"]
        assert isinstance(computed, dict) and sorted(computed) == sorted(expected), (
            "compute_usual_demand gave the planted shortage '"
            + label
            + "' usual demand for "
            + str(sorted(computed) if isinstance(computed, dict) else computed)
            + " where its customers are "
            + str(sorted(expected))
            + ". Every customer with an open order has a usual demand, nought where it took "
            "nothing in the policy's weeks."
        )
        differing = {account_code: (str(computed[account_code]), str(expected[account_code]))
                     for account_code in expected
                     if computed[account_code] != expected[account_code]}
        assert len(differing) == 0, (
            "Over the last "
            + str(shortage["rules"]["history_weeks"])
            + " weeks, the planted shortage '"
            + label
            + "' gives usual demand (worked out, stated) of "
            + str(differing)
            + ". Usual demand is the cases shipped to the customer over the policy's last weeks, "
            "a shipment exactly that many weeks ago included and older ones not, divided by the "
            "number of weeks: a week with no shipment counts as nought, so 70 cases every other "
            "week is 30 a week, and 55 cases over seven weeks is exactly 55/7, kept as a fraction "
            "and never rounded."
        )
    shortages = planted_shortages["shortages"]
    shortage = shortages["every other week and 55 cases over seven weeks at 1.4"]
    assert shortage["division_problem"] is None, shortage["division_problem"]
    shown = shortage["lines"]["CUS-0131"]["usual_demand"]
    assert shown == 7.8571 and shortage["computed_usual"]["CUS-0131"] == Fraction(55, 7), (
        "A customer that took 55 cases over seven weeks shows a usual demand of "
        + repr(shown)
        + " on its allocation line, where it is shown to four decimal places, 7.8571, and "
        "kept exactly as 55/7 for the division."
    )


def test_eligible_is_the_lower_of_the_order_and_the_cap_times_usual_demand_rounded_down_exactly(
        planted_shortages):
    assert planted_shortages["problem"] is None, planted_shortages["problem"]
    stated = {
        "every notation and an order sent twice": (
            {"CUS-0021": 150, "CUS-0034": 60, "CUS-0046": 15, "CUS-0062": 100},
            {"CUS-0021": 42, "CUS-0034": 17, "CUS-0046": 15, "CUS-0062": 34},
            "1200 units by EDI from C0021 at twelve to a case, 60 cs by the portal, 2 plt by "
            "email from Acct 62 at fifty to a pallet, 15 cs by email from Acct 46 sent again by "
            "EDI as 240 units and kept as it first arrived, and 600 units more by EDI from "
            "C0021; 1.7 times 25 is 42.5, so 42"),
        "every other week and 55 cases over seven weeks at 1.4": (
            {"CUS-0021": 300, "CUS-0046": 50, "CUS-0131": 80},
            {"CUS-0021": 140, "CUS-0046": 42, "CUS-0131": 11},
            "1.4 times 55 cases over seven weeks is exactly 11, where floating point gives "
            "10.999999999999998 and rounds it down to 10"),
        "cases left in stock beside a customer with no usual demand": (
            {"CUS-0021": 40, "CUS-0034": 30}, {"CUS-0021": 40, "CUS-0034": 0},
            "a customer with no shipment in the policy's weeks may have nothing while the "
            "product is on allocation"),
        "no customer may have any case": (
            {"CUS-0021": 40, "CUS-0105": 20}, {"CUS-0021": 0, "CUS-0105": 0},
            "neither customer took anything in the policy's weeks"),
        "the worked example": (
            {"CUS-0021": 1000, "CUS-0034": 600, "CUS-0046": 20, "CUS-0062": 500, "CUS-0077": 150,
             "CUS-0089": 40, "CUS-0105": 300, "CUS-0118": 150, "CUS-0131": 200},
            {"CUS-0021": 400, "CUS-0034": 300, "CUS-0046": 20, "CUS-0062": 120, "CUS-0077": 100,
             "CUS-0089": 40, "CUS-0105": 250, "CUS-0118": 150, "CUS-0131": 90},
            "the question's own table for HS-0500 under the hand hygiene policy"),
    }
    for label, (ordered, eligible, meaning) in stated.items():
        shortage = planted_shortages["shortages"][label]
        assert shortage["reading_problem"] is None, shortage["reading_problem"]
        assert shortage["division_problem"] is None, shortage["division_problem"]
        found_ordered = {account_code: allocation_line.get("cases_ordered")
                         for account_code, allocation_line in shortage["lines"].items()}
        found_eligible = {account_code: allocation_line.get("eligible_cases")
                          for account_code, allocation_line in shortage["lines"].items()}
        assert found_ordered == ordered and found_eligible == eligible, (
            "The planted shortage '"
            + label
            + "' gives cases ordered "
            + str(found_ordered)
            + " and eligible "
            + str(found_eligible)
            + " where they are "
            + str(ordered)
            + " and "
            + str(eligible)
            + ": "
            + meaning
            + ". A customer's cases ordered are every order it sent, each counted once, and it "
            "may have the lower of those and the cap times its usual demand, worked out exactly "
            "and rounded down."
        )
        codes = [allocation_line["account_code"] for allocation_line
                 in shortage["allocation"]["allocation_lines"]]
        assert codes == sorted(ordered), (
            "The allocation lines of the planted shortage '"
            + label
            + "' come in the order "
            + str(codes)
            + ", where every customer has one line, in account-code order."
        )


def test_protected_customers_are_served_first_in_full_or_dividing_the_cases_among_themselves(
        planted_shortages):
    from config import config

    assert planted_shortages["problem"] is None, planted_shortages["problem"]
    release = config.DECISION_RELEASE
    reserve_release = config.DECISION_RELEASE_WITH_RESERVE
    stated = [
        ("a protected group too large for the cases", release, ["CUS-0105", "CUS-0118"],
         {"CUS-0021": 0, "CUS-0105": 80, "CUS-0118": 40},
         "two Healthcare customers that may have 100 and 50 between 120 cases divide them 80 and "
         "40 by usual demand, and the Supermarket customer gets nothing"),
        ("a protected group too large for the cases", reserve_release, ["CUS-0105", "CUS-0118"],
         {"CUS-0021": 0, "CUS-0105": 80, "CUS-0118": 40},
         "with no case left after the protected customers there is no reserve to hold"),
        ("a reserve of 8.5 cases", release, ["CUS-0105"], {"CUS-0105": 20},
         "the Healthcare customer gets its 20 in full before anyone else"),
        ("the worked example", release, ["CUS-0105", "CUS-0118"],
         {"CUS-0105": 250, "CUS-0118": 150},
         "Northgate and Oakhaven, both Healthcare, get their 250 and 150 in full"),
    ]
    for label, release_name, wanted_protected, expected, meaning in stated:
        shortage = planted_shortages["shortages"][label]
        assert shortage["reading_problem"] is None, shortage["reading_problem"]
        assert shortage["division_problem"] is None, shortage["division_problem"]
        protected = sorted(account_code for account_code, allocation_line
                           in shortage["lines"].items() if allocation_line.get("protected") is True)
        assert protected == wanted_protected, (
            "The planted shortage '"
            + label
            + "' marks "
            + str(protected)
            + " as protected where "
            + str(wanted_protected)
            + ", its customers in the Healthcare channel the policy protects, are the protected "
            "ones. A customer is protected by its channel in the customer master."
        )
        found = {}
        for account_code in expected:
            if account_code in shortage["lines"]:
                found[account_code] = shortage["lines"][account_code]["cases_by_release"].get(
                    release_name)
        assert found == expected, (
            "Under "
            + release_name
            + " the planted shortage '"
            + label
            + "' gives "
            + str(found)
            + " where it gives "
            + str(expected)
            + ": "
            + meaning
            + ". Protected customers are served first: in full when their eligible quantities "
            "fit in the cases, and otherwise the cases are divided among them alone, by the same "
            "division, and no other customer gets anything."
        )
    shortage = planted_shortages["shortages"]["a protected group too large for the cases"]
    held = shortage["summary"]["releases"][reserve_release]["reserve_cases"]
    assert held == 0, (
        "With every case gone to the protected customers, the planted shortage holds "
        + str(held)
        + " cases in reserve under RELEASE WITH RESERVE, where the reserve is a share of the "
        "cases left after the protected customers, and none are left."
    )


def test_the_rest_are_divided_by_usual_demand_passing_on_every_share_at_its_limit_until_none_is(
        planted_shortages):
    from config import config

    assert planted_shortages["problem"] is None, planted_shortages["problem"]
    release = config.DECISION_RELEASE
    stated = [
        ("three customers capped over two passes",
         {"CUS-0001": 50, "CUS-0003": 40, "CUS-0021": 65, "CUS-0034": 145},
         "300 cases in proportion to 100 : 100 : 100 : 200 give 60 each to the first three and "
         "120 to the fourth; the first two may have only 50 and 40, so they take those and 210 "
         "are divided again, 70 and 140; the third may have only 65, so it takes that, and the "
         "fourth gets the 145 left"),
        ("every notation and an order sent twice",
         {"CUS-0021": 21, "CUS-0034": 8, "CUS-0046": 15, "CUS-0062": 16},
         "Saverly's share of 16 reaches the 15 it may have, so it takes 15 and 45 are divided "
         "again among the other three"),
        ("every other week and 55 cases over seven weeks at 1.4",
         {"CUS-0021": 140, "CUS-0046": 42, "CUS-0131": 11},
         "193 cases is exactly what the three may have, so each gets all of it"),
        ("the worked example",
         {"CUS-0021": 293, "CUS-0034": 220, "CUS-0046": 20, "CUS-0062": 88, "CUS-0077": 73,
          "CUS-0089": 40, "CUS-0105": 250, "CUS-0118": 150, "CUS-0131": 66},
         "BasketDrop's 57.1429 and Saverly's 21.4286 reach their 40 and 20, and the 740 left go "
         "to the other five by usual demand"),
    ]
    for label, expected, meaning in stated:
        shortage = planted_shortages["shortages"][label]
        assert shortage["reading_problem"] is None, shortage["reading_problem"]
        assert shortage["division_problem"] is None, shortage["division_problem"]
        found = {account_code: allocation_line["cases_by_release"].get(release)
                 for account_code, allocation_line in shortage["lines"].items()}
        assert found == expected, (
            "Under RELEASE the planted shortage '"
            + label
            + "' gives "
            + str(found)
            + " where it gives "
            + str(expected)
            + ": "
            + meaning
            + ". The cases are divided in proportion to usual demand; a share at or above its "
            "customer's eligible quantity takes that quantity, and the rest is divided again "
            "among the others, until no share reaches its limit."
        )
    for label, shortage in planted_shortages["shortages"].items():
        assert shortage["reading_problem"] is None, shortage["reading_problem"]
        assert shortage["division_problem"] is None, shortage["division_problem"]
        over = []
        for account_code, allocation_line in shortage["lines"].items():
            for release_name, cases in allocation_line["cases_by_release"].items():
                if cases > allocation_line["eligible_cases"] or cases < 0:
                    over.append((account_code, release_name, cases))
        assert len(over) == 0, (
            "The planted shortage '"
            + label
            + "' gives "
            + str(over)
            + " (customer, release, cases), more than the customer may have or fewer than none. "
            "An order is never given more than its customer may have; what nobody may have stays "
            "in stock."
        )


def test_whole_cases_go_by_largest_remainder_the_reserve_is_held_and_every_case_is_counted(
        planted_shortages):
    from config import config

    assert planted_shortages["problem"] is None, planted_shortages["problem"]
    release = config.DECISION_RELEASE
    reserve_release = config.DECISION_RELEASE_WITH_RESERVE
    escalate = config.DECISION_ESCALATE
    stated = [
        ("a tie in the remainders settled by usual demand", release,
         {"CUS-0001": 2, "CUS-0003": 8}, None,
         "shares of 2.5 and 7.5 tie on their remainders, and the case left goes to the larger "
         "usual demand"),
        ("a tie in the remainders settled by account code", release,
         {"CUS-0001": 6, "CUS-0003": 5}, None,
         "shares of 5.5 and 5.5 tie on their remainders and their usual demand, and the case "
         "left goes to the lower account code, though CUS-0003 ordered first"),
        ("a reserve of 8.5 cases", release, {"CUS-0021": 50, "CUS-0034": 30, "CUS-0105": 20},
         (0, 5), "everyone fits, and the 5 cases nobody may have stay in stock"),
        ("a reserve of 8.5 cases", reserve_release,
         {"CUS-0021": 48, "CUS-0034": 29, "CUS-0105": 20}, (8, 0),
         "ten percent of the 85 cases left after the protected customer is 8.5, held as 8, and "
         "the 77 left are divided 48.125 and 28.875, the case left going to the larger "
         "remainder"),
        ("a cover of 49 percent", release, {"CUS-0021": 29, "CUS-0034": 20}, (0, 0),
         "49 cases divided 29.4 and 19.6, the case left going to the larger remainder"),
        ("cases left in stock beside a customer with no usual demand", release,
         {"CUS-0021": 40, "CUS-0034": 0}, (0, 60),
         "the customer that may have 40 gets 40, the one that may have nothing gets nothing, "
         "and 60 stay in stock"),
        ("no customer may have any case", release, {"CUS-0021": 0, "CUS-0105": 0}, (0, 50),
         "no customer may have any case, so all 50 stay in stock"),
        ("the worked example", reserve_release,
         {"CUS-0021": 262, "CUS-0034": 196, "CUS-0046": 20, "CUS-0062": 78, "CUS-0077": 65,
          "CUS-0089": 40, "CUS-0105": 250, "CUS-0118": 150, "CUS-0131": 59}, (80, 0),
         "80 held, ten percent of the 800 left after the protected customers, and 677 placed by "
         "rounding down with the 3 left going to Brightline, Saverly and Greenacre"),
    ]
    for label, release_name, expected, reserve_and_stock, meaning in stated:
        shortage = planted_shortages["shortages"][label]
        assert shortage["reading_problem"] is None, shortage["reading_problem"]
        assert shortage["division_problem"] is None, shortage["division_problem"]
        found = {account_code: allocation_line["cases_by_release"].get(release_name)
                 for account_code, allocation_line in shortage["lines"].items()}
        summary_release = shortage["summary"]["releases"].get(release_name, {})
        found_reserve_and_stock = (summary_release.get("reserve_cases"),
                                   summary_release.get("cases_in_stock"))
        assert found == expected and (reserve_and_stock is None
                                      or found_reserve_and_stock == reserve_and_stock), (
            "Under "
            + release_name
            + " the planted shortage '"
            + label
            + "' gives "
            + str(found)
            + ", with "
            + str(found_reserve_and_stock)
            + " cases held in reserve and left in stock, where it gives "
            + str(expected)
            + ("" if reserve_and_stock is None else ", with " + str(reserve_and_stock))
            + ": "
            + meaning
            + ". Every share is rounded down and the cases left go one each to the largest "
            "remainders, a tie to the larger usual demand and then to the lower account code; "
            "the reserve is the policy's percentage of the cases left after the protected "
            "customers, rounded down."
        )
    for label, shortage in planted_shortages["shortages"].items():
        assert shortage["reading_problem"] is None, shortage["reading_problem"]
        assert shortage["division_problem"] is None, shortage["division_problem"]
        for release_name, summary_release in shortage["summary"]["releases"].items():
            placed = sum(allocation_line["cases_by_release"][release_name]
                         for allocation_line in shortage["lines"].values())
            counted = placed + summary_release["reserve_cases"] + summary_release["cases_in_stock"]
            assert (counted == shortage["cases"] and summary_release["cases_placed"] == placed
                    and summary_release["cases_in_stock"] >= 0), (
                "Under "
                + release_name
                + " the planted shortage '"
                + label
                + "' places "
                + str(placed)
                + " cases, holds "
                + str(summary_release["reserve_cases"])
                + " and leaves "
                + str(summary_release["cases_in_stock"])
                + " in stock, "
                + str(counted)
                + " in all, where "
                + str(shortage["cases"])
                + " are available. Every case is placed, held or left in stock, and none twice."
            )
    stated_decisions = [
        ("a tie in the remainders settled by usual demand", [release, escalate], 50.0,
         {reserve_release: config.NO_RESERVE_HELD_REASON},
         "a cover of exactly 50 percent leaves RELEASE open, and a policy with no reserve never "
         "offers RELEASE WITH RESERVE"),
        ("a reserve of 8.5 cases", [release, reserve_release, escalate], 105.0, {},
         "every protected customer is served in full and the cover is above half"),
        ("a cover of 49 percent", [escalate], 49.0,
         {release: config.COVER_BELOW_MINIMUM_REASON,
          reserve_release: config.COVER_BELOW_MINIMUM_REASON + config.REASON_SEPARATOR
          + config.NO_RESERVE_HELD_REASON},
         "a cover below half closes RELEASE"),
        ("a protected group too large for the cases", [escalate], 63.16,
         {release: config.PROTECTED_CUSTOMER_SHORT_REASON,
          reserve_release: config.PROTECTED_CUSTOMER_SHORT_REASON},
         "a protected customer going short closes both releases, whatever the cover"),
        ("no customer may have any case", [escalate], None,
         {release: config.NOTHING_ELIGIBLE_REASON,
          reserve_release: config.NOTHING_ELIGIBLE_REASON},
         "with nothing anyone may have there is no cover to work out, and only ESCALATE is "
         "open"),
        ("cases left in stock beside a customer with no usual demand", [release, escalate], 250.0,
         {reserve_release: config.NO_RESERVE_HELD_REASON},
         "100 cases against the 40 customers may have"),
        ("the worked example", [release, reserve_release, escalate], 81.63, {},
         "1,200 cases against the 1,470 customers may have"),
    ]
    for label, decisions_open, cover, closed, meaning in stated_decisions:
        summary = planted_shortages["shortages"][label]["summary"]
        found_reasons = summary.get("decisions_not_open")
        if isinstance(found_reasons, dict):
            found_reasons = {decision: sorted(str(reasons).split(config.REASON_SEPARATOR))
                             for decision, reasons in found_reasons.items()}
        wanted_reasons = {decision: sorted(reasons.split(config.REASON_SEPARATOR))
                          for decision, reasons in closed.items()}
        found = (summary.get("decisions_open"), summary.get("cover_percent"), found_reasons)
        assert found == (decisions_open, cover, wanted_reasons), (
            "The planted shortage '"
            + label
            + "' gives the decisions open, the cover and the reasons for each closed decision as "
            + str(found)
            + " where they are "
            + str((decisions_open, cover, wanted_reasons))
            + ": "
            + meaning
            + ". ESCALATE is always open; RELEASE when every protected customer is served in full "
            "and the cover is at least 50 percent; RELEASE WITH RESERVE when RELEASE is open and "
            "the policy holds a reserve; each closed decision names every reason, in the "
            "configuration's words."
        )
    short = planted_shortages["shortages"]["every notation and an order sent twice"]["summary"]
    assert short["releases"][release]["customers_short"] == ["CUS-0021", "CUS-0034", "CUS-0062"], (
        "Under RELEASE the customers short of what they may have are given as "
        + str(short["releases"][release]["customers_short"])
        + " where three of the four get less than they may have, CUS-0021, CUS-0034 and "
        "CUS-0062, and Saverly gets all its 15."
    )


def test_each_reviewing_agent_was_called_once_and_in_order(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    listed = saved_run["calls_at_gate"]
    assert (listed["status"] == 200 and isinstance(listed["body"], dict)
            and {"calls", "call_count", "expected_call_count"} <= set(listed["body"])
            and all("agent_name" in call for call in listed["body"]["calls"])), (
        "Reading the run's model calls at the pause returned "
        + str(listed["status"])
        + " with "
        + str(listed["body"])[:300]
        + ", where the audit area lists every call under the agent that made it, with their "
        "count and the count expected."
    )
    at_gate = listed["body"]
    reviewing = [config.AGENT_NAME_PLANNER, config.AGENT_NAME_DEMAND, config.AGENT_NAME_REQUESTS,
                 config.AGENT_NAME_ADVISOR, config.AGENT_NAME_WRITER]
    called = [call["agent_name"] for call in at_gate["calls"]]
    assert at_gate["call_count"] == len(at_gate["calls"]) == config.LLM_CALLS_BEFORE_DECISION, (
        "Reaching the Supply Planning Manager recorded a call count of "
        + str(at_gate["call_count"])
        + " and listed "
        + str(len(at_gate["calls"]))
        + " language model calls where the workflow allows "
        + str(config.LLM_CALLS_BEFORE_DECISION)
        + ", one per reviewing agent. More means an agent ran twice; fewer, that one never ran."
    )
    assert called == reviewing, (
        "The agents ran in the order "
        + str(called)
        + ". Each one reads what the one before it concluded: the demand is analysed against "
        "the plan, the requests are reviewed against it, the allocation is advised on with the "
        "plan in hand, and the recommendation draws on all of them."
    )
    assert at_gate["expected_call_count"] == config.LLM_CALLS_BEFORE_DECISION, (
        "Before any decision the service expected "
        + str(at_gate["expected_call_count"])
        + " calls. Until a decision is taken only the reviewing agents' calls are expected."
    )


def test_every_model_call_is_recorded_with_what_it_cost(saved_run):
    from config import config
    from persistence.database import session_factory
    from persistence.models import LlmCall

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    listed = saved_run["final_calls"]
    assert listed["status"] == 200 and isinstance(listed["body"], dict) and {
            "calls", "total_latency_milliseconds", "run_id"} <= set(listed["body"]), (
        "Reading the finished run's model calls returned "
        + str(listed["status"])
        + " with "
        + str(listed["body"])[:300]
        + ", where the audit area lists every call, the run it belongs to and their total "
        "latency."
    )
    final = listed["body"]
    agents = {config.AGENT_NAME_PLANNER, config.AGENT_NAME_DEMAND, config.AGENT_NAME_REQUESTS,
              config.AGENT_NAME_ADVISOR, config.AGENT_NAME_WRITER, config.AGENT_NAME_NOTICE,
              config.AGENT_NAME_BRIEFING}
    assert len(final["calls"]) > 0 and final["run_id"] == saved_run["at_gate"]["run_id"], (
        "The calls listed belong to run "
        + str(final["run_id"])
        + " where the review was made by run "
        + str(saved_run["at_gate"]["run_id"])
        + ", or none was recorded at all, so nothing can say what the review cost."
    )
    for call in final["calls"]:
        missing = sorted({"agent_name", "model_name", "prompt_characters", "response_characters",
                          "latency_milliseconds"} - set(call))
        assert len(missing) == 0, (
            "A call listed for the run carries no "
            + str(missing)
            + ", so who made it and what it cost cannot be read."
        )
        assert call["agent_name"] in agents, (
            "A call was recorded against "
            + str(call["agent_name"])
            + ", which is not one of the seven agents."
        )
        assert str(call["model_name"]).strip(), (
            "The call from "
            + call["agent_name"]
            + " does not say which model answered it, so an audit asking what reviewed the "
            "shortage has nothing to read."
        )
        assert call["prompt_characters"] > 0 and call["response_characters"] > 0, (
            "The call from "
            + call["agent_name"]
            + " records a prompt of "
            + str(call["prompt_characters"])
            + " and a reply of "
            + str(call["response_characters"])
            + " characters. A call that was made sent something and got something back."
        )
        assert call["latency_milliseconds"] > 0, (
            "The call from "
            + call["agent_name"]
            + " took "
            + str(call["latency_milliseconds"])
            + " milliseconds by its own record."
        )
    assert final["total_latency_milliseconds"] == sum(
            call["latency_milliseconds"] for call in final["calls"]), (
        "The run's total latency "
        + str(final["total_latency_milliseconds"])
        + " is not the sum of its calls."
    )
    assert "llm_call_count" in saved_run["final"] and "llm_call_count" in saved_run["at_gate"], (
        "The record carries no llm_call_count at the pause or after the decision, so the run's "
        "own count of its calls cannot be read."
    )
    assert saved_run["final"]["llm_call_count"] == len(final["calls"]), (
        "The run records "
        + str(saved_run["final"]["llm_call_count"])
        + " calls against the "
        + str(len(final["calls"]))
        + " logged for it. The run's own count is what the audit trail reads."
    )
    gate_body = saved_run["calls_at_gate"]["body"]
    gate_calls = gate_body.get("calls") if isinstance(gate_body, dict) else None
    assert (isinstance(gate_calls, list)
            and saved_run["at_gate"]["llm_call_count"] == len(gate_calls)), (
        "At the pause the run recorded "
        + str(saved_run["at_gate"]["llm_call_count"])
        + " calls against the "
        + (str(len(gate_calls)) if isinstance(gate_calls, list) else "unreadable list of")
        + " calls logged for it by then. The count is kept up to date while the run waits, "
        "not only when it finishes."
    )
    with session_factory() as session:
        stored = session.query(LlmCall).filter(
            LlmCall.run_id == saved_run["at_gate"]["run_id"]).count()
    assert stored == len(final["calls"]), (
        "The database holds "
        + str(stored)
        + " model calls for the run where the service lists "
        + str(len(final["calls"]))
        + ". Every call is written to llm_calls as it is made, and the list is read from there."
    )


def test_the_review_reaches_the_manager_as_a_recommendation_among_the_decisions_open(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    missing = sorted({"recommended_decision", "decisions_open", "planner_request", "planner_id",
                      "review_plan_text", "recommendation_text", "product_category",
                      "policy_document", "inflated_order_scores", "customer_requests"}
                     - set(at_gate))
    assert len(missing) == 0, (
        "The record returned when the review reached a Supply Planning Manager carries no "
        + str(missing)
        + ", so what the agents concluded cannot be read."
    )
    assert (at_gate["recommended_decision"] in config.DECISIONS
            and at_gate["recommended_decision"] in at_gate["decisions_open"]), (
        "The writer recommended "
        + repr(at_gate["recommended_decision"])
        + " where the decisions open for this product are "
        + str(at_gate["decisions_open"])
        + ". A recommendation the allocation does not allow, or one in other words, cannot be "
        "acted on."
    )
    assert (at_gate["planner_request"] == saved_run["planner_request"]
            and at_gate["planner_id"] == saved_run["opener"]["username"]), (
        "The record holds the request the review was opened with as "
        + repr(at_gate["planner_request"])
        + ", made by "
        + repr(at_gate["planner_id"])
        + ", where "
        + saved_run["opener"]["username"]
        + " asked "
        + repr(saved_run["planner_request"])
        + ". The plan is only readable against what was actually asked, and by whom."
    )
    plan = str(at_gate["review_plan_text"]).strip()
    recommendation = str(at_gate["recommendation_text"]).strip()
    assert "\n" in plan and len(recommendation) > 0, (
        "The review plan stored runs to "
        + str(len(plan))
        + " characters and the recommendation to "
        + str(len(recommendation))
        + ", or the plan sits on one line, so an agent's reply never reached the record whole."
    )
    assert plan != recommendation and recommendation not in plan, (
        "The recommendation stored is the plan over again, so the writer's reply was thrown away."
    )
    bands = {config.INFLATED_ORDER_BAND_HIGH, config.INFLATED_ORDER_BAND_MEDIUM,
             config.INFLATED_ORDER_BAND_LOW}
    assert (at_gate["product_category"] in config.CATEGORY_POLICY_DOCUMENTS
            and at_gate["policy_document"]
            == config.CATEGORY_POLICY_DOCUMENTS[at_gate["product_category"]]
            and len(at_gate["inflated_order_scores"]) > 0
            and all(score.get("inflated_order_band") in bands
                    for score in at_gate["inflated_order_scores"])
            and set(at_gate["customer_requests"].values())
            <= set(config.REQUEST_GUIDE_DOCUMENTS)), (
        "The record names the category "
        + repr(at_gate["product_category"])
        + ", the policy "
        + repr(at_gate["policy_document"])
        + ", the order bands "
        + str(sorted({str(score.get("inflated_order_band"))
                      for score in at_gate["inflated_order_scores"]}))
        + " and the customers' requests "
        + str(sorted(set(at_gate["customer_requests"].values())))
        + ", which are not the product's own policy, the configured bands or the eight request "
        "types."
    )


def test_a_reply_decorated_the_way_models_write_is_read_back_whole():
    from agentic.agents import parse_labelled_response
    from config import config

    labels = ["PROTECTED CUSTOMERS", "LIMITS REACHED", "ALLOCATION SUMMARY"]
    first_protected = "Northgate Health Supplies (CUS-0105): may have 250; gets 250 under RELEASE"
    second_protected = "Oakhaven Care Homes (CUS-0118): may have 150; gets 150 under RELEASE"
    limits = "BasketDrop Online and Saverly Discount reach what they may have on the first pass"
    summary_start = "The protected customers are served in full."
    summary_end = "The other seven share the 800 cases left by usual demand."
    reply = ("Sure - here is my advice on how the cases were divided.\n"
             "\n"
             "**PROTECTED CUSTOMERS:**\n"
             + first_protected + "\n"
             + second_protected + "\n"
             "## LIMITS REACHED:\n"
             + limits + "\n"
             "\n"
             "> ALLOCATION SUMMARY: " + summary_start + "\n"
             + summary_end)
    refusal = ""
    try:
        parsed = parse_labelled_response(reply, labels, "decoration_check")
    except ValueError as error:
        parsed, refusal = {}, str(error)
    assert refusal == "", (
        "A reply holding all three fields was refused: "
        + refusal
        + ". A model wraps its labels in markdown, headings and quotes, and a reader that cannot "
        "see past them stops every review at the first agent."
    )
    assert sorted(parsed) == sorted(labels), (
        "Reading the reply returned the fields "
        + str(sorted(parsed))
        + " where "
        + str(sorted(labels))
        + " were asked for."
    )
    for label in labels:
        value = str(parsed[label])
        assert value.strip() and value == value.strip(config.RESPONSE_DECORATION_CHARACTERS), (
            label
            + " was read back as "
            + repr(value)
            + ". The decoration a model puts around a label belongs to neither the label nor its "
            "value, and an empty value is no value."
        )
    protected = parsed["PROTECTED CUSTOMERS"]
    assert (first_protected in protected and second_protected in protected
            and limits not in protected and summary_start not in protected), (
        "PROTECTED CUSTOMERS was read back as "
        + repr(protected)
        + ". A value that runs over several lines keeps every line until the next label, and "
        "nothing from any other field."
    )
    assert limits in parsed["LIMITS REACHED"] and first_protected not in parsed["LIMITS REACHED"], (
        "LIMITS REACHED was read back as "
        + repr(parsed["LIMITS REACHED"])
        + " where the reply gives it as "
        + repr(limits)
        + "."
    )
    assert summary_start in parsed["ALLOCATION SUMMARY"] and summary_end in parsed[
            "ALLOCATION SUMMARY"], (
        "ALLOCATION SUMMARY was read back as "
        + repr(parsed["ALLOCATION SUMMARY"])
        + ". A label behind a quote mark is still a label, and its value runs on to the next "
        "line."
    )
    try:
        written_twice = parse_labelled_response(
            "PROTECTED CUSTOMERS: " + first_protected + "\nLIMITS REACHED: " + limits
            + "\nPROTECTED CUSTOMERS: " + second_protected + "\nALLOCATION SUMMARY: "
            + summary_start, labels, "decoration_check")["PROTECTED CUSTOMERS"]
    except ValueError as error:
        written_twice = "refused: " + str(error)
    assert first_protected in written_twice and second_protected in written_twice, (
        "A reply that wrote the PROTECTED CUSTOMERS label again on its third line was read back "
        "as "
        + repr(written_twice)
        + ". A label written again adds its line to the field and never throws away what came "
        "before it, or advice written that way loses every protected customer but the last."
    )
    try:
        hyphenated = parse_labelled_response(
            "HEADLINE: Hand sanitiser is on allocation this week\n"
            "**Why-it-is-short:** A filling line broke down.",
            ["HEADLINE", "WHY IT IS SHORT"], "decoration_check")["WHY IT IS SHORT"]
    except ValueError as error:
        hyphenated = "refused: " + str(error)
    assert hyphenated == "A filling line broke down.", (
        "A label written 'Why-it-is-short' in bold was read back as "
        + repr(hyphenated)
        + " for the field WHY IT IS SHORT. A model writes a label in its own case and with the "
        "hyphens the words sometimes carry, and it is still the same label."
    )
    for incomplete in ["PROTECTED CUSTOMERS: One\nLIMITS REACHED: Two",
                       "PROTECTED CUSTOMERS: One\nLIMITS REACHED:\nALLOCATION SUMMARY: Three",
                       "LIMITS REACHED: Two\nALLOCATION SUMMARY: Three"]:
        refused = False
        try:
            parse_labelled_response(incomplete, labels, "decoration_check")
        except ValueError:
            refused = True
        assert refused, (
            "A reply missing a field, or leaving one empty, was accepted: "
            + repr(incomplete)
            + ". The next agent reads every field, so a reply without one has to be refused with "
            "a ValueError rather than passed on."
        )


def test_a_release_drafts_the_notice_then_a_briefing_quoting_it_and_neither_names_a_customer(
        saved_run):
    import re

    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    listed = {name: saved_run[name] for name in ["calls_at_gate", "final_calls"]}
    for name, entry in listed.items():
        assert (entry["status"] == 200 and isinstance(entry["body"], dict)
                and {"calls", "call_count", "expected_call_count"} <= set(entry["body"])
                and all("agent_name" in call for call in entry["body"]["calls"])), (
            "Reading the run's model calls "
            + ("at the pause" if name == "calls_at_gate" else "after the decision")
            + " returned "
            + str(entry["status"])
            + " with "
            + str(entry["body"])[:300]
            + ", where the audit area lists every call under the agent that made it, with their "
            "count and the count expected."
        )
    before = listed["calls_at_gate"]["body"]["calls"]
    final = listed["final_calls"]["body"]
    drafting = [call["agent_name"] for call in final["calls"][len(before):]]
    assert drafting == [config.AGENT_NAME_NOTICE, config.AGENT_NAME_BRIEFING], (
        "After a decision of "
        + saved_run["decision"]
        + " the agents called were "
        + str(drafting)
        + ". A release writes the allocation notice to the customers first and the sales "
        "briefing second, so the briefing can tell the sales team what the customers were told."
    )
    assert (final["call_count"] == len(final["calls"]) == final["expected_call_count"]
            == config.LLM_CALLS_BY_DECISION[saved_run["decision"]]), (
        "A decision of "
        + saved_run["decision"]
        + " ended with "
        + str(final["call_count"])
        + " calls recorded, "
        + str(len(final["calls"]))
        + " listed and "
        + str(final["expected_call_count"])
        + " expected, where the workflow allows "
        + str(config.LLM_CALLS_BY_DECISION[saved_run["decision"]])
        + ". The expected figure follows the decision taken."
    )
    missing = sorted({"notice_text", "briefing_text", "recommendation_text"}
                     - set(saved_run["final"]))
    assert len(missing) == 0, (
        "The record returned once the decision was recorded carries no "
        + str(missing)
        + ", so what the drafting agents wrote cannot be read."
    )
    notice = str(saved_run["final"]["notice_text"] or "").strip()
    briefing = str(saved_run["final"]["briefing_text"] or "").strip()
    assert "\n" in notice and "\n" in briefing, (
        "A release stored an allocation notice of "
        + str(len(notice))
        + " characters and a sales briefing of "
        + str(len(briefing))
        + ", or one of them on a single line, so a drafting agent's reply never reached the "
        "record whole."
    )
    assert notice != briefing and notice not in briefing and briefing not in notice, (
        "The allocation notice stored is the sales briefing, or holds the whole of it, so one "
        "text was written into the other's place."
    )
    headline_lines = [line for line in notice.split("\n") if line.upper().startswith("HEADLINE")]
    headline_words = (re.findall(r"[a-z0-9]+", headline_lines[0].split(":", 1)[-1].lower())
                      if headline_lines else [])
    briefing_words = re.findall(r"[a-z0-9]+", briefing.lower())
    quoted = any(briefing_words[start:start + len(headline_words)] == headline_words
                 for start in range(len(briefing_words)))
    assert len(headline_words) > 0 and quoted, (
        "The sales briefing "
        + repr(briefing[:200])
        + " does not quote the allocation notice's headline, whose words are "
        + repr(" ".join(headline_words))
        + ". The sales team tells customers what the notice says, and only that, so the "
        "briefing quotes it."
    )
    names = [str(line["customer_name"]) for line in saved_run["detail"]["body"]["allocation_lines"]]
    named = sorted(name for name in names
                   if re.search(re.escape(name.lower()), (notice + " " + briefing).lower()))
    assert len(names) > 0 and len(named) == 0, (
        "The allocation notice or the sales briefing names "
        + str(named)
        + ". The notice goes to every customer of the product and the briefing to the whole "
        "sales team, so neither names a customer: each customer hears its own figure on its "
        "own notice line."
    )
    recommendation = str(saved_run["final"]["recommendation_text"]).strip()
    assert recommendation == str(saved_run["at_gate"]["recommendation_text"]).strip(), (
        "After the decision the stored recommendation is no longer the one the Supply Planning "
        "Manager decided on."
    )


def test_the_workflow_has_ten_reachable_nodes_and_sends_each_decision_down_its_own_path(tmp_path):
    from agentic.graph import build_workflow
    from agentic.state import AllocationReviewState
    from config import config

    compiled = build_workflow(str(tmp_path / "workflow_check.db"))
    node_names = sorted(str(name) for name in compiled.builder.nodes)
    expected_nodes = sorted(["load_shortage_context", "plan_allocation_review", "analyse_demand",
                             "review_customer_requests", "compute_allocation",
                             "compose_allocation_recommendation", "request_allocation_decision",
                             "draft_allocation_notice", "draft_sales_briefing",
                             "record_allocation_outcome"])
    assert node_names == expected_nodes, (
        "The workflow declares the nodes "
        + str(node_names)
        + " where the design has "
        + str(expected_nodes)
        + "."
    )
    branches = compiled.builder.branches.get("request_allocation_decision", {})
    assert len(branches) == 1, (
        "The decision node has "
        + str(len(branches))
        + " conditional branches where it has one, choosing the path from the decision."
    )
    branch = list(branches.values())[0]
    destinations = {}
    for decision in config.DECISIONS:
        state = {key: None for key in AllocationReviewState.__annotations__}
        state["allocation_decision"] = {"decision": decision, "decided_by": "Routing Check",
                                        "decided_by_role": config.ROLE_MANAGER,
                                        "decision_reason": "Routing is being checked."}
        routed = branch.path.invoke(state)
        if isinstance(routed, (list, tuple)) and len(routed) == 1:
            routed = routed[0]
        destinations[decision] = branch.ends[routed] if branch.ends else routed
    expected_paths = {config.DECISION_RELEASE: "draft_allocation_notice",
                      config.DECISION_RELEASE_WITH_RESERVE: "draft_allocation_notice",
                      config.DECISION_ESCALATE: "draft_sales_briefing"}
    assert destinations == expected_paths, (
        "The decisions route to "
        + str(destinations)
        + ". Either release writes the allocation notice and then the sales briefing; an "
        "escalation tells no customer anything and writes the sales briefing alone."
    )
    edges = {(str(source), str(target)) for source, target in compiled.builder.edges}
    edges |= {("request_allocation_decision", str(target)) for target in destinations.values()}
    reached = {"__start__"}
    grew = True
    while grew:
        grew = False
        for source, target in edges:
            if source in reached and target not in reached:
                reached.add(target)
                grew = True
    unreachable = sorted(set(node_names) - reached)
    assert len(unreachable) == 0, (
        "These nodes cannot be reached from the start: "
        + str(unreachable)
        + ". A node with no way in never runs, whatever it contains."
    )
    for source, target, reason in [
            ("compose_allocation_recommendation", "request_allocation_decision",
             "the review is stored before the run waits for a Supply Planning Manager"),
            ("draft_allocation_notice", "draft_sales_briefing",
             "the sales briefing tells the sales team what the notice told the customers, so it "
             "is written after it"),
            ("draft_sales_briefing", "record_allocation_outcome",
             "what was written has to reach the record"),
            ("record_allocation_outcome", "__end__", "the run ends once the outcome is recorded")]:
        assert (source, target) in edges, (
            "Nothing leads from " + source + " to " + target + ": " + reason + "."
        )


def test_the_run_waits_for_a_manager_who_did_not_open_it_before_any_customer_is_told(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    undecided = ["decision", "decided_by", "decided_by_role", "decision_reason",
                 "cases_confirmed", "reserve_cases", "cases_in_stock", "notice_text",
                 "notice_lines", "briefing_text"]
    missing = sorted(({"status"} | set(undecided)) - set(at_gate))
    assert len(missing) == 0, (
        "The record returned when the run paused carries no "
        + str(missing)
        + ", so what has and has not been decided while it waits cannot be read."
    )
    assert at_gate["status"] == config.REVIEW_STATUS_PENDING, (
        "When the five reviewing agents finished, the review stood at "
        + str(at_gate["status"])
        + " where it waits at "
        + config.REVIEW_STATUS_PENDING
        + ". The Supply Allocation Policy gives the decision to a Supply Planning Manager, so no "
        "case is confirmed and no notice is drafted until one decides."
    )
    pending = saved_run["pending_at_gate"]
    assert pending["status"] == 200 and isinstance(pending["body"], list), (
        "Listing the reviews pending a decision returned "
        + str(pending["status"])
        + " with "
        + str(pending["body"])[:300]
        + ", so the managers' decision queue cannot be read."
    )
    assert at_gate["review_id"] in [str(row.get("review_id")) for row in pending["body"]], (
        "The review waiting for a decision is missing from the list of reviews pending a "
        "decision, so the decision queue would not show it."
    )
    nothing_yet = {field_name: at_gate[field_name] for field_name in undecided
                   if at_gate[field_name] is not None}
    assert len(nothing_yet) == 0, (
        "Before anyone decided, the record already holds "
        + str(nothing_yet)[:400]
        + ". Nothing may be decided, confirmed, held back or written to a customer while the "
        "run waits."
    )
    assert saved_run["run_record"]["problem"] is None, saved_run["run_record"]["problem"]
    status_at_gate = saved_run["run_record"]["status_at_gate"]
    assert status_at_gate == config.AGENT_RUN_STATUS_AWAITING_DECISION, (
        "While it waited the run was recorded as "
        + str(status_at_gate)
        + " where it should read "
        + config.AGENT_RUN_STATUS_AWAITING_DECISION
        + ". A paused run that still reads as running tells nobody it is waiting for them."
    )
    refused = saved_run["own_decision"]
    assert (refused["status"] == 403 and refused["still_pending"] is True
            and refused["calls_after"] == refused["calls_before"]
            and config.OWN_REVIEW_REFUSED_MESSAGE in str(refused["body"])), (
        "The Supply Planning Manager who opened "
        + at_gate["review_id"]
        + " was answered "
        + str(refused["status"])
        + " with "
        + str(refused["body"])[:200]
        + " when they tried to decide it; afterwards the review "
        + ("was still" if refused["still_pending"] else "was no longer")
        + " waiting, with "
        + str(refused["calls_after"])
        + " model calls recorded where "
        + str(refused["calls_before"])
        + " were recorded before. The Supply Allocation Policy separates opening a review from "
        "deciding it, so the person who opened it is refused with 403 and the configuration's "
        "message before anything resumes, and another Supply Planning Manager decides it."
    )


def test_resuming_finished_the_run_from_its_checkpoint_on_disk_and_closed_it_complete(saved_run):
    from agentic.graph import build_workflow
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["run_record"]["problem"] is None, saved_run["run_record"]["problem"]
    assert saved_run["run_record"]["status_final"] == config.AGENT_RUN_STATUS_COMPLETE, (
        "The run finished at "
        + str(saved_run["run_record"]["status_final"])
        + " rather than "
        + config.AGENT_RUN_STATUS_COMPLETE
        + ". A finished run that still reads as waiting stays in somebody's queue."
    )
    reopened = build_workflow(config.CHECKPOINT_DATABASE_FILE).get_state(
        {"configurable": {"thread_id": saved_run["run_record"]["thread_id"]}})
    assert reopened.values.get("review_id") == saved_run["at_gate"]["review_id"], (
        "A workflow built afresh on "
        + config.CHECKPOINT_DATABASE_FILE
        + " does not find the run's state under its thread, so a paused run would not survive "
        "the service being restarted overnight."
    )
    assert len(reopened.next) == 0 and (
            (reopened.values.get("allocation_decision") or {}).get("decision")
            == saved_run["decision"]), (
        "The checkpoint of the finished run still waits at "
        + str(reopened.next)
        + " or holds a different decision, so the run on disk is not the one that finished."
    )


def test_the_decision_is_recorded_once_with_what_it_confirms_and_holds_and_a_closed_one_refused(
        saved_run, service_client, staff):
    from config import config
    from persistence.database import session_factory
    from persistence.models import AllocationDecision
    from tools.allocation_tools import compute_allocation_outcome

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert staff["problem"] is None, staff["problem"]
    at_gate = saved_run["at_gate"]
    closed = saved_run["closed_decision"]
    expected_refusal = config.DECISION_NOT_OPEN_MESSAGE.format(
        decision=config.DECISION_RELEASE_WITH_RESERVE, reasons=config.NO_RESERVE_HELD_REASON)
    assert (closed["status"] == 409 and isinstance(closed["body"], dict)
            and closed["body"].get("detail") == expected_refusal
            and closed["calls_after"] == closed["calls_before"]), (
        "Deciding RELEASE WITH RESERVE on "
        + at_gate["review_id"]
        + ", whose policy holds no reserve and whose decisions open are "
        + str(at_gate.get("decisions_open"))
        + ", was answered "
        + str(closed["status"])
        + " with "
        + str(closed["body"])[:300]
        + ", and the calls went from "
        + str(closed["calls_before"])
        + " to "
        + str(closed["calls_after"])
        + ". A decision the allocation does not allow is refused with 409 and the "
        "configuration's message naming why, "
        + repr(expected_refusal)
        + ", before the run resumes."
    )
    final = saved_run["final"]
    missing = sorted({"status", "decision", "decided_by", "decided_by_role", "decision_reason",
                      "cases_confirmed", "reserve_cases", "cases_in_stock", "notice_lines"}
                     - set(final))
    assert len(missing) == 0, (
        "The record returned once the decision was recorded carries no "
        + str(missing)
        + ", so what the decision confirms, holds and leaves in stock cannot be read."
    )
    assert final["status"] == config.REVIEW_STATUS_BY_DECISION[saved_run["decision"]], (
        "After a decision of "
        + saved_run["decision"]
        + " the review stands at "
        + str(final["status"])
        + " where the configuration maps it to "
        + config.REVIEW_STATUS_BY_DECISION[saved_run["decision"]]
        + "."
    )
    assert (final["decision"] == saved_run["decision"]
            and final["decided_by"] == saved_run["decider"]["display_name"]
            and final["decided_by_role"] == config.ROLE_MANAGER
            and final["decision_reason"] == saved_run["decision_reason"]), (
        "The decision is recorded as "
        + str({key: final[key] for key in ["decision", "decided_by", "decided_by_role",
                                           "decision_reason"]})
        + ". It has to name the decision, the person who took it, their role and their reason "
        "as written."
    )
    detail = saved_run["detail"]["body"]
    stored_allocation = {
        "cases_available": detail["summary"]["cases_available"],
        "allocation_lines": detail["allocation_lines"],
        "reserves_by_release": {release_name: release["reserve_cases"] for release_name, release
                                in detail["summary"]["releases"].items()},
    }
    worked_out = compute_allocation_outcome(saved_run["decision"], stored_allocation,
                                            detail["summary"],
                                            detail["profile"][config.PRODUCT_NAME_COLUMN])
    expected_record = {key: (worked_out or {}).get(key) for key in [
        "cases_confirmed", "reserve_cases", "cases_in_stock", "notice_lines"]}
    recorded = {key: final[key] for key in expected_record}
    assert (recorded == expected_record and isinstance(recorded["cases_confirmed"], dict)
            and len(recorded["cases_confirmed"]) == len(detail["allocation_lines"])), (
        "A decision of "
        + saved_run["decision"]
        + " recorded "
        + str(recorded)[:500]
        + " where compute_allocation_outcome, given the product's own allocation, gives "
        + str(expected_record)[:500]
        + ". The outcome is worked out from the allocation the moment the decision arrives, never "
        "chosen by an agent: each customer is confirmed its cases under the release decided, on "
        "its own notice line in the configuration's words, and what nobody may have stays in "
        "stock."
    )
    allocation = {
        "cases_available": 100,
        "allocation_lines": [
            {"account_code": "CUS-0008", "customer_name": "Nettleham Foodstores",
             "channel": "Supermarket", "protected": False, "cases_ordered": 40,
             "usual_demand": 25.0, "eligible_cases": 30,
             "cases_by_release": {config.DECISION_RELEASE: 25,
                                  config.DECISION_RELEASE_WITH_RESERVE: 23}},
            {"account_code": "CUS-0009", "customer_name": "Merriton Supermarkets",
             "channel": "Supermarket", "protected": False, "cases_ordered": 30,
             "usual_demand": 25.0, "eligible_cases": 30,
             "cases_by_release": {config.DECISION_RELEASE: 25,
                                  config.DECISION_RELEASE_WITH_RESERVE: 22}},
            {"account_code": "CUS-0021", "customer_name": "Greenacre Foodstores",
             "channel": "Supermarket", "protected": False, "cases_ordered": 50,
             "usual_demand": 40.0, "eligible_cases": 40,
             "cases_by_release": {config.DECISION_RELEASE: 30,
                                  config.DECISION_RELEASE_WITH_RESERVE: 27}},
            {"account_code": "CUS-0089", "customer_name": "BasketDrop Online",
             "channel": "Online", "protected": False, "cases_ordered": 10, "usual_demand": 0.0,
             "eligible_cases": 0,
             "cases_by_release": {config.DECISION_RELEASE: 0,
                                  config.DECISION_RELEASE_WITH_RESERVE: 0}},
            {"account_code": "CUS-0105", "customer_name": "Northgate Health Supplies",
             "channel": "Healthcare", "protected": True, "cases_ordered": 20,
             "usual_demand": 20.0, "eligible_cases": 20,
             "cases_by_release": {config.DECISION_RELEASE: 20,
                                  config.DECISION_RELEASE_WITH_RESERVE: 20}},
        ],
        "reserves_by_release": {config.DECISION_RELEASE: 0,
                                config.DECISION_RELEASE_WITH_RESERVE: 8},
    }
    open_summary = {
        "cases_available": 100, "cases_ordered": 150, "cases_eligible": 120,
        "cover_percent": 83.33, "protected_served_in_full": True,
        "releases": {
            config.DECISION_RELEASE: {"cases_placed": 100, "reserve_cases": 0,
                                      "cases_in_stock": 0,
                                      "customers_short": ["CUS-0008", "CUS-0009", "CUS-0021"]},
            config.DECISION_RELEASE_WITH_RESERVE: {"cases_placed": 92, "reserve_cases": 8,
                                                   "cases_in_stock": 0,
                                                   "customers_short": ["CUS-0008", "CUS-0009",
                                                                       "CUS-0021"]}},
        "decisions_open": [config.DECISION_RELEASE, config.DECISION_RELEASE_WITH_RESERVE,
                           config.DECISION_ESCALATE],
        "decisions_not_open": {},
    }
    planted_cases = [
        (config.DECISION_RELEASE,
         {"cases_confirmed": {"CUS-0008": 25, "CUS-0009": 25, "CUS-0021": 30, "CUS-0089": 0,
                              "CUS-0105": 20},
          "cases_released": 100, "customers_receiving_cases": 4, "reserve_cases": 0,
          "cases_in_stock": 0,
          "channel_totals": {"Supermarket": {"customers": 3, "cases_released": 80}}}),
        (config.DECISION_RELEASE_WITH_RESERVE,
         {"cases_confirmed": {"CUS-0008": 23, "CUS-0009": 22, "CUS-0021": 27, "CUS-0089": 0,
                              "CUS-0105": 20},
          "cases_released": 92, "customers_receiving_cases": 4, "reserve_cases": 8,
          "cases_in_stock": 0,
          "channel_totals": {"Supermarket": {"customers": 3, "cases_released": 72}}}),
        (config.DECISION_ESCALATE,
         {"cases_confirmed": {}, "cases_released": 0, "customers_receiving_cases": 0,
          "reserve_cases": 0, "cases_in_stock": 100, "channel_totals": {}}),
    ]
    for decision, expected in planted_cases:
        outcome = compute_allocation_outcome(decision, allocation, open_summary,
                                             "Planted Hand Wash 500 ml")
        found = {key: (outcome or {}).get(key) for key in expected}
        if decision == config.DECISION_ESCALATE:
            if isinstance(found["cases_confirmed"], dict) and all(
                    cases == 0 for cases in found["cases_confirmed"].values()):
                found["cases_confirmed"] = {}
            if isinstance(found["channel_totals"], dict) and all(
                    isinstance(total, dict) and total.get("cases_released") == 0
                    for total in found["channel_totals"].values()):
                found["channel_totals"] = {}
        assert found == expected, (
            decision
            + " on a planted allocation of 100 cases, three Supermarket customers, one Online "
            "customer that may have nothing and one protected Healthcare customer, gave "
            + str(found)
            + " where it gives "
            + str(expected)
            + ". A release confirms each customer its cases under that release and holds that "
            "release's reserve; an escalation confirms nothing and leaves every case in stock; a "
            "customer confirmed nothing is not counted as receiving cases; and the sales "
            "briefing's channel totals cover only channels with at least "
            + str(config.SALES_BRIEFING_MINIMUM_CHANNEL_CUSTOMERS)
            + " customers."
        )
    released = compute_allocation_outcome(config.DECISION_RELEASE, allocation, open_summary,
                                          "Planted Hand Wash 500 ml")
    assert released.get("notice_lines", {}).get("CUS-0089") == config.NOTICE_LINE_FORMAT.format(
            customer_name="BasketDrop Online", cases_confirmed=0, cases_ordered=10,
            product_name="Planted Hand Wash 500 ml"), (
        "The notice line for a customer confirmed no cases reads "
        + repr(released.get("notice_lines", {}).get("CUS-0089"))
        + ". Every customer with an open order gets its own line, in the configuration's words, "
        "even one confirmed nothing."
    )
    closed_summary = dict(open_summary, cover_percent=41.67,
                          decisions_open=[config.DECISION_ESCALATE],
                          decisions_not_open={
                              config.DECISION_RELEASE: config.COVER_BELOW_MINIMUM_REASON,
                              config.DECISION_RELEASE_WITH_RESERVE:
                                  config.COVER_BELOW_MINIMUM_REASON})
    for decision, summary, wanted in [
            (config.DECISION_RELEASE, closed_summary, config.DECISION_NOT_OPEN_MESSAGE.format(
                decision=config.DECISION_RELEASE, reasons=config.COVER_BELOW_MINIMUM_REASON)),
            ("HOLD", open_summary, "")]:
        message = None
        try:
            compute_allocation_outcome(decision, allocation, summary, "Planted Hand Wash 500 ml")
        except ValueError as error:
            message = str(error)
        assert message is not None and wanted in message, (
            "Working out the outcome of "
            + repr(decision)
            + " for a planted allocation whose decisions open are "
            + str(summary["decisions_open"])
            + " gave "
            + ("no ValueError" if message is None else repr(message[:200]))
            + ". A decision the allocation does not leave open is refused in the configuration's "
            "words, naming why, and a value that is not a decision at all is refused."
        )
    client = service_client["client"]
    calls_path = "/reviews/" + at_gate["review_id"] + "/calls"
    calls_before = client.get(calls_path, headers=staff["planner"]["headers"])
    refusals = [
        ("a second decision on the review", "/reviews/" + at_gate["review_id"] + "/decision",
         {"decision": saved_run["decision"],
          "decision_reason": "A second decision on the same review."},
         saved_run["decider"]["headers"], 409, config.ALREADY_DECIDED_MESSAGE),
        ("a second review of the product", "/reviews",
         {"product_code": saved_run["product_code"],
          "planner_request": "Review this product again from the start."},
         staff["manager"]["headers"], 409,
         config.PRODUCT_ALREADY_REVIEWED_MESSAGE.format(product_code=saved_run["product_code"])),
        ("a review opened with a request of two characters", "/reviews",
         {"product_code": saved_run["product_code"], "planner_request": "ok"},
         staff["manager"]["headers"], 422, None),
        ("a decision that is not one of the three", "/reviews/" + at_gate["review_id"]
         + "/decision", {"decision": "HOLD", "decision_reason": "Hold everything for a week."},
         saved_run["decider"]["headers"], 422, None),
        ("a decision with a reason of five characters", "/reviews/" + at_gate["review_id"]
         + "/decision", {"decision": saved_run["decision"], "decision_reason": "Fine."},
         saved_run["decider"]["headers"], 422, None)]
    for description, path, payload, headers, wanted_status, wanted_detail in refusals:
        reply = client.post(path, json=payload, headers=headers)
        detail = (reply.json().get("detail")
                  if reply.headers.get("content-type") == "application/json" else reply.text[:200])
        assert reply.status_code == wanted_status and (wanted_detail is None
                                                       or detail == wanted_detail), (
            "Sending "
            + description
            + " returned "
            + str(reply.status_code)
            + " with "
            + reply.text[:200]
            + " where it is refused with "
            + str(wanted_status)
            + ("" if wanted_detail is None else " and " + repr(wanted_detail))
            + ". A product is reviewed once and a review decided once, and a request too short "
            "to act on, an unknown decision or a reason too short to explain it is refused "
            "before a language model call is spent."
        )
    calls_after = client.get(calls_path, headers=staff["planner"]["headers"])
    assert (calls_before.status_code == calls_after.status_code == 200
            and calls_after.json().get("call_count") == calls_before.json().get("call_count")), (
        "The refused requests changed the review's model calls from "
        + calls_before.text[:100]
        + " to "
        + calls_after.text[:100]
        + ". A refused request spends no call."
    )
    listed = client.get("/reviews", params={"status_value": config.REVIEW_STATUS_OPTIONS},
                        headers=staff["planner"]["headers"])
    same_product = ([record for record in listed.json()
                     if record.get("product_code") == saved_run["product_code"]]
                    if listed.status_code == 200 else [])
    assert (len(same_product) == 1
            and same_product[0].get("review_id") == at_gate["review_id"]
            and same_product[0].get("decision") == saved_run["decision"]
            and same_product[0].get("status") == final["status"]), (
        "After the second decision was refused, the product is listed as "
        + str([(record.get("review_id"), record.get("status"), record.get("decision"))
               for record in same_product])
        + ". It is listed once, with its one review, the first decision and its status."
    )
    with session_factory() as session:
        stored = session.query(AllocationDecision).filter(
            AllocationDecision.review_id == at_gate["review_id"]).count()
    assert stored == 1, (
        "allocation_decisions holds "
        + str(stored)
        + " rows for the review. Each decision is written once, when it is taken, and a refused "
        "one writes nothing."
    )


def test_the_stored_review_agrees_with_the_tools_that_produced_it(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["detail"]["status"] == 200 and isinstance(saved_run["detail"]["body"],
                                                               dict), (
        "The detail of "
        + str(saved_run.get("product_code"))
        + " returned "
        + str(saved_run["detail"]["status"])
        + " with "
        + str(saved_run["detail"]["body"])[:300]
        + ", so the facts the tools work out cannot be compared with the record."
    )
    stored = saved_run["at_gate"]
    detail = saved_run["detail"]["body"]
    missing = (sorted({"product_code", "product_name", "product_category", "policy_document",
                       "cases_available", "cases_ordered", "cases_eligible", "cover_percent",
                       "protected_served_in_full", "allocation_lines", "releases",
                       "decisions_open", "decisions_not_open", "inflated_order_scores",
                       "email_requests", "customer_requests", "forecast_shortfall"} - set(stored))
               + sorted({"profile", "policy_rules", "allocation_lines", "summary", "order_lines",
                         "email_requests", "customer_requests", "forecast_shortfall"}
                        - set(detail)))
    assert len(missing) == 0, (
        "The stored record, or the detail the tools work out again for the same product, "
        "carries no "
        + str(missing)
        + ", so the two cannot be compared."
    )
    profile = detail["profile"]
    summary = detail["summary"]
    comparisons = [
        ("product_code", stored["product_code"], profile[config.PRODUCT_CODE_COLUMN]),
        ("product_name", stored["product_name"], profile[config.PRODUCT_NAME_COLUMN]),
        ("product_category", stored["product_category"], profile[config.PRODUCT_CATEGORY_COLUMN]),
        ("policy_document", stored["policy_document"], detail["policy_rules"]["document_name"]),
        ("cases_available", stored["cases_available"],
         profile[config.CASES_AVAILABLE_THIS_WEEK_COLUMN]),
        ("cases_ordered", stored["cases_ordered"], summary["cases_ordered"]),
        ("cases_eligible", stored["cases_eligible"], summary["cases_eligible"]),
        ("cover_percent", stored["cover_percent"], summary["cover_percent"]),
        ("protected_served_in_full", stored["protected_served_in_full"],
         summary["protected_served_in_full"]),
        ("allocation_lines", stored["allocation_lines"], detail["allocation_lines"]),
        ("releases", stored["releases"], summary["releases"]),
        ("decisions_open", stored["decisions_open"], summary["decisions_open"]),
        ("decisions_not_open", stored["decisions_not_open"], summary["decisions_not_open"]),
        ("inflated_order_scores", sorted((score["order_line_id"], score["inflated_order_score"],
                                          score["inflated_order_band"])
                                         for score in stored["inflated_order_scores"]),
         sorted((line["order_line_id"], line["inflated_order_score"], line["inflated_order_band"])
                for line in detail["order_lines"])),
        ("email_requests", [(email["email_id"], email["request_type"])
                            for email in stored["email_requests"]],
         [(email["email_id"], email["request_type"]) for email in detail["email_requests"]]),
        ("customer_requests", stored["customer_requests"], detail["customer_requests"]),
        ("forecast_shortfall", stored["forecast_shortfall"], detail["forecast_shortfall"]),
    ]
    for field_name, stored_value, worked_out in comparisons:
        assert stored_value == worked_out, (
            "The stored "
            + field_name
            + " is "
            + str(stored_value)[:300]
            + " where the tools, run again on the same product for its detail, give "
            + str(worked_out)[:300]
            + ". The record holds the facts the decision was taken on, the tools' own and never "
            "an agent's."
        )


def test_signing_in_and_out_decides_who_the_service_will_serve(service_client):
    from access.password_security import hash_password
    from config import config
    from persistence.database import session_factory
    from persistence.models import UserAccount
    from persistence.repository import fetch_user_account

    assert service_client["problem"] is None, service_client["problem"]
    client = service_client["client"]
    planners = [account for account in config.SEED_USER_ACCOUNTS
                if account["role"] == config.ROLE_PLANNER]
    account = planners[-1]
    credentials = {"username": account["username"], "password": account["password"]}
    signed_in = client.post("/auth/signin", json=credentials)
    body = signed_in.json() if signed_in.status_code == 200 else {}
    assert ({"username", "session_token", "display_name", "role", "pages", "areas"} <= set(body)
            and str(body["session_token"]).strip()), (
        "Signing in as "
        + account["username"]
        + " with the seeded password returned "
        + str(signed_in.status_code)
        + " with "
        + signed_in.text[:200]
        + ", where a sign-in returns a session token with the user's name, role, pages and "
        "areas."
    )
    assert (body["username"] == account["username"] and body["role"] == account["role"]
            and body["display_name"] == account["display_name"]
            and sorted(body["pages"]) == sorted(config.ROLE_PAGE_ACCESS[account["role"]])
            and sorted(body["areas"]) == sorted(config.ROLE_AREA_ACCESS[account["role"]])), (
        "Signing in as a "
        + account["role"]
        + " returned the name "
        + repr(body["display_name"])
        + ", the role "
        + str(body["role"])
        + ", the pages "
        + str(body["pages"])
        + " and the areas "
        + str(body["areas"])
        + ", which are not the account's name and what the configuration grants that role."
    )
    wrong = client.post("/auth/signin", json={"username": account["username"],
                                              "password": "wrong-" + account["password"]})
    stranger = client.post("/auth/signin", json={"username": "no-such-person",
                                                 "password": account["password"]})
    assert wrong.status_code == 401 and stranger.status_code == 401, (
        "A wrong password returned "
        + str(wrong.status_code)
        + " and an unknown username "
        + str(stranger.status_code)
        + ". Both are refused."
    )
    assert wrong.json() == stranger.json(), (
        "A wrong password is refused with "
        + repr(wrong.json())
        + " and an unknown username with "
        + repr(stranger.json())
        + ". Two different refusals tell a stranger which usernames exist."
    )
    parameters = {"product_category": config.QUEUE_CATEGORY_ALL}
    first_headers = {config.SESSION_TOKEN_HEADER: body["session_token"]}
    accepted = client.get("/queue", params=parameters, headers=first_headers)
    assert accepted.status_code == 200, (
        "The token just issued to "
        + account["username"]
        + " was not accepted by the queue. A fresh session token opens the areas its role is "
        "granted."
    )
    for headers in [None, {config.SESSION_TOKEN_HEADER: "not-a-token-anyone-was-given"}]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert refused.status_code == 401, (
            "The queue answered "
            + str(refused.status_code)
            + " to a request "
            + ("with no token" if headers is None else "with an invented token")
            + ". Only a signed-in user may read it."
        )
    again = client.post("/auth/signin", json=credentials)
    second_headers = {config.SESSION_TOKEN_HEADER: again.json().get("session_token", "")}
    replaced = client.get("/queue", params=parameters, headers=first_headers)
    current = client.get("/queue", params=parameters, headers=second_headers)
    assert (again.status_code == 200 and replaced.status_code == 401
            and current.status_code == 200), (
        "After "
        + account["username"]
        + " signed in a second time, answered "
        + str(again.status_code)
        + ", the first token got "
        + str(replaced.status_code)
        + " and the second "
        + str(current.status_code)
        + " from the queue. An account holds one session: signing in somewhere else ends the "
        "session before it."
    )
    signed_out = client.post("/auth/signout", headers=second_headers)
    after = client.get("/queue", params=parameters, headers=second_headers)
    assert signed_out.status_code == 200 and after.status_code == 401, (
        "Signing out answered "
        + str(signed_out.status_code)
        + ", and the old token then got "
        + str(after.status_code)
        + " from the queue. Signing out answers 200 and ends the session."
    )
    with session_factory() as session:
        stored = fetch_user_account(session, UserAccount.username == account["username"])
        other = fetch_user_account(session, UserAccount.username == planners[0]["username"])
    missing = sorted({"password_hash", "password_salt"}
                     - (set(stored if isinstance(stored, dict) else {})
                        & set(other if isinstance(other, dict) else {})))
    assert len(missing) == 0, (
        "fetch_user_account read "
        + account["username"]
        + " and "
        + planners[0]["username"]
        + " back without "
        + str(missing)
        + ", so how their passwords are kept cannot be read."
    )
    assert (stored["password_hash"] != account["password"]
            and stored["password_hash"] == hash_password(account["password"],
                                                         stored["password_salt"])), (
        "The password stored for "
        + account["username"]
        + " is not the salted hash of the password. A password kept as written is read by "
        "anyone who can open the database."
    )
    assert (stored["password_salt"] != other["password_salt"]
            and stored["password_hash"] != other["password_hash"]), (
        account["username"]
        + " and "
        + planners[0]["username"]
        + " share a password and are stored with the same salt or the same hash, so one "
        "cracked hash opens both accounts."
    )


def test_every_role_reaches_exactly_the_areas_it_is_granted(service_client, staff):
    from config import config

    assert staff["problem"] is None, staff["problem"]
    client = service_client["client"]
    headers = {config.ROLE_MANAGER: staff["manager"]["headers"],
               config.ROLE_PLANNER: staff["planner"]["headers"],
               config.ROLE_ADVISER: staff["adviser"]["headers"]}
    unknown_product = "ZZ-9999"
    unknown_review = config.REVIEW_ID_PREFIX + "DOES-NOT-EXIST"
    probes = [
        (config.AREA_INTAKE, "GET", "/queue", {"product_category": config.QUEUE_CATEGORY_ALL},
         None, 200),
        (config.AREA_INTAKE, "GET", "/queue/" + unknown_product, None, None, 404),
        (config.AREA_REVIEW, "POST", "/reviews", None,
         {"product_code": unknown_product,
          "planner_request": "Divide the cases as the policy says."}, 404),
        (config.AREA_AUDIT, "GET", "/reviews",
         {"status_value": [config.REVIEW_STATUS_PENDING]}, None, 200),
        (config.AREA_AUDIT, "GET", "/reviews/" + unknown_review + "/calls", None, None, 404),
        (config.AREA_AUDIT, "GET", "/audit/chat-turns", {"turn_limit": 1}, None, 200),
        (config.AREA_DECISION, "POST", "/reviews/" + unknown_review + "/decision", None,
         {"decision": config.DECISION_ESCALATE,
          "decision_reason": "Nothing can go out this week."}, 404),
        (config.AREA_CHAT, "GET", "/chat/turns", {"turn_limit": 1}, None, 200),
    ]
    for role_name, role_headers in headers.items():
        for area_name, method, path, parameters, payload, status_when_granted in probes:
            reply = client.request(method, path, params=parameters, json=payload,
                                   headers=role_headers)
            granted = area_name in config.ROLE_AREA_ACCESS[role_name]
            assert (reply.status_code == 403) != granted and reply.status_code != 401, (
                "A signed-in "
                + role_name
                + " calling "
                + method
                + " "
                + path
                + " got "
                + str(reply.status_code)
                + ". The "
                + area_name
                + " area is "
                + ("granted" if granted else "not granted")
                + " to that role, and the guard runs before the endpoint: a role without the "
                "area is refused with 403 whether or not the thing asked for exists."
            )
            assert not granted or reply.status_code == status_when_granted, (
                "A signed-in "
                + role_name
                + " calling "
                + method
                + " "
                + path
                + " got "
                + str(reply.status_code)
                + " where it is "
                + str(status_when_granted)
                + (". A request that names a product or a review that does not exist is "
                   "answered 404, and a review opened on a product not in the queue stops "
                   "before it spends a language model call."
                   if status_when_granted == 404 else
                   ". A role granted an area is served by every endpoint in it.")
            )
    for role_name, area_name, method, path, parameters, payload in [
            (config.ROLE_ADVISER, config.AREA_INTAKE, "GET", "/queue",
             {"product_category": config.QUEUE_CATEGORY_ALL}, None),
            (config.ROLE_PLANNER, config.AREA_DECISION, "POST",
             "/reviews/" + unknown_review + "/decision", None,
             {"decision": config.DECISION_ESCALATE,
              "decision_reason": "Nothing can go out this week."})]:
        refused = client.request(method, path, params=parameters, json=payload,
                                 headers=headers[role_name])
        expected = config.AREA_REFUSED_MESSAGE.format(role=role_name, area=area_name)
        assert refused.status_code == 403 and refused.json().get("detail") == expected, (
            "A signed-in "
            + role_name
            + " calling "
            + method
            + " "
            + path
            + " was refused with "
            + refused.text[:200]
            + " where the refusal says, in the configuration's words, "
            + repr(expected)
            + "."
        )


def test_the_queue_puts_the_largest_forecast_shortfall_first(service_client, staff, live_files,
                                                             saved_models):
    from config import config
    from tools.allocation_tools import rank_shortage_queue
    from tools.prediction_tools import compute_forecast_shortfall, forecast_customer_demand
    from tools.shortage_data_tools import fetch_open_orders

    assert staff["problem"] is None, staff["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    planted_shortfalls = [
        (compute_forecast_shortfall({"CUS-0001": 10.4, "CUS-0003": 10.4, "CUS-0021": 10.4}, 20),
         11, "three forecasts of 10.4 cases against 20 next week: 31.2 rounded once is 31, where "
             "rounding each first would give 30"),
        (compute_forecast_shortfall({"CUS-0021": 5.0}, 20), -15,
         "a forecast of 5 cases against 20 next week: more supply than demand is a shortfall "
         "below nought, kept as it is"),
    ]
    for found, expected, meaning in planted_shortfalls:
        assert found == expected, (
            "compute_forecast_shortfall gave "
            + repr(found)
            + " for "
            + meaning
            + "; it is "
            + str(expected)
            + ". The forecasts are summed, rounded once to whole cases and next week's supply "
            "taken away."
        )
    planted_queue = [{"product_code": "PB-0002", "forecast_shortfall": 40},
                     {"product_code": "PA-0001", "forecast_shortfall": -5},
                     {"product_code": "PC-0003", "forecast_shortfall": 40},
                     {"product_code": "PA-0004", "forecast_shortfall": 90}]
    ranked = [entry["product_code"] for entry in rank_shortage_queue(planted_queue)]
    assert ranked == ["PA-0004", "PB-0002", "PC-0003", "PA-0001"], (
        "rank_shortage_queue put planted products with forecast shortfalls of 40, -5, 40 and 90 "
        "in the order "
        + str(ranked)
        + ". The largest shortfall comes first and a tie goes to the lower product code."
    )
    refused = False
    try:
        rank_shortage_queue(planted_queue + [{"product_code": "PA-0001",
                                              "forecast_shortfall": 7}])
    except ValueError:
        refused = True
    assert refused, (
        "A queue holding one product twice was ranked without a ValueError. A product on "
        "allocation is in the queue once, or it is reviewed twice."
    )
    client = service_client["client"]
    headers = staff["planner"]["headers"]
    reply = client.get("/queue", params={"product_category": config.QUEUE_CATEGORY_ALL},
                       headers=headers)
    assert reply.status_code == 200 and isinstance(reply.json(), list), (
        "The queue returned "
        + str(reply.status_code)
        + " with "
        + reply.text[:200]
        + " to a signed-in Supply Planner."
    )
    everything = reply.json()
    missing = sorted({"queue_position", "product_code", "product_name", "product_category",
                      "cases_available_this_week", "cases_available_next_week", "cases_ordered",
                      "cases_eligible", "cover_percent", "protected_served_in_full",
                      "decisions_open", "forecast_shortfall"}
                     - set(everything[0] if everything else {}))
    assert len(missing) == 0, (
        "A queue row carries no "
        + str(missing)
        + ", so the queue cannot be ordered, filtered or read."
    )
    codes = [row["product_code"] for row in everything]
    live_codes = live_files["products"][config.PRODUCT_CODE_COLUMN]
    assert sorted(codes) == sorted(live_codes), (
        "The queue holds "
        + str(len(everything))
        + " products where the live file holds "
        + str(len(live_codes))
        + ". Every product on allocation is in the queue exactly once."
    )
    keys = [(-row["forecast_shortfall"], row["product_code"]) for row in everything]
    out_of_order = [(codes[position], codes[position + 1])
                    for position in range(len(keys) - 1) if keys[position] > keys[position + 1]]
    assert len(out_of_order) == 0, (
        "The queue is not ordered by forecast shortfall, largest first, and then by product "
        "code: "
        + str(out_of_order[:3])
        + ". The planners work the queue from the top."
    )
    assert [row["queue_position"] for row in everything] == list(range(1, len(everything) + 1)), (
        "The queue positions run "
        + str([row["queue_position"] for row in everything][:10])
        + " ... where they count from 1 down the queue."
    )
    inconsistent = []
    for row in everything:
        cover = row["cover_percent"]
        if (config.DECISION_ESCALATE not in row["decisions_open"]
                or (config.DECISION_RELEASE_WITH_RESERVE in row["decisions_open"]
                    and config.DECISION_RELEASE not in row["decisions_open"])
                or (config.DECISION_RELEASE in row["decisions_open"]
                    and not (row["protected_served_in_full"] is True and cover is not None
                             and cover >= config.MINIMUM_RELEASE_COVER_PERCENT))
                or (cover is None) != (row["cases_eligible"] == 0)
                or (cover is not None
                    and abs(cover * row["cases_eligible"] - 100 * row["cases_available_this_week"])
                    > 0.0051 * row["cases_eligible"])):
            inconsistent.append(row["product_code"])
    assert len(inconsistent) == 0, (
        str(len(inconsistent))
        + " products, for example "
        + str(inconsistent[:3])
        + ", show a cover or decisions open that cannot follow from their cases. The cover is "
        "the cases available as a percentage of what customers may have, and there is none when "
        "they may have nothing; ESCALATE is always open, RELEASE only when every protected "
        "customer is served in full and the cover is at least half, and RELEASE WITH RESERVE "
        "never without RELEASE."
    )
    stated_rows = {
        "HS-0500": (81.63, [config.DECISION_RELEASE, config.DECISION_RELEASE_WITH_RESERVE,
                            config.DECISION_ESCALATE]),
        "HF-0400": (60.95, [config.DECISION_RELEASE, config.DECISION_RELEASE_WITH_RESERVE,
                            config.DECISION_ESCALATE]),
        "HS-0050": (9.46, [config.DECISION_ESCALATE]),
        "HW-5000": (15.33, [config.DECISION_ESCALATE]),
    }
    found_rows = {row["product_code"]: (row["cover_percent"], row["decisions_open"])
                  for row in everything if row["product_code"] in stated_rows}
    assert found_rows == stated_rows, (
        "The queue shows four hand hygiene products' cover and decisions open as "
        + str(found_rows)
        + " where, under the hand hygiene policy, they are "
        + str(stated_rows)
        + ": a cover at or above half with every protected customer served opens both releases "
        "under a policy that holds a reserve, and a cover below half leaves only ESCALATE."
    )
    for product_category in config.QUEUE_CATEGORY_OPTIONS[1:]:
        filtered = client.get("/queue", params={"product_category": product_category},
                              headers=headers).json()
        assert filtered == [row for row in everything
                            if row["product_category"] == product_category], (
            "Filtering the queue to "
            + product_category
            + " did not return exactly the "
            + product_category
            + " products of the full queue, in the same order and with the same positions."
        )
    unknown_filter = client.get("/queue", params={"product_category": "Air care"},
                                headers=headers)
    assert unknown_filter.status_code == 422, (
        "Filtering the queue to a category Wrenfield Hygiene does not sell returned "
        + str(unknown_filter.status_code)
        + " rather than 422. The filter takes a category the configuration names, or all of "
        "them, and nothing else."
    )
    for row in [everything[0], everything[len(everything) // 2], everything[-1]]:
        product_orders = fetch_open_orders(live_files["orders"], row["product_code"])
        forecast_shortfall = compute_forecast_shortfall(
            forecast_customer_demand(saved_models["demand"], saved_models["demand_schema"],
                                     product_orders),
            row["cases_available_next_week"])
        detail = client.get("/queue/" + row["product_code"], headers=headers).json()
        assert (detail.get("queue_position") == row["queue_position"]
                and detail.get("forecast_shortfall") == row["forecast_shortfall"]
                == forecast_shortfall), (
            "The detail of "
            + row["product_code"]
            + " gives its position and forecast shortfall as "
            + str({key: detail.get(key) for key in ["queue_position", "forecast_shortfall"]})
            + " and the queue shows "
            + str({key: row[key] for key in ["queue_position", "forecast_shortfall"]})
            + ", where the saved demand model's forecasts for its customers, summed and rounded "
            "once, less next week's "
            + str(row["cases_available_next_week"])
            + " cases, give "
            + str(forecast_shortfall)
            + ". The queue and the detail work from the trainee's own saved model."
        )


def test_a_policy_question_is_answered_from_the_documents_and_remembered_by_the_asker_alone(
        chat_replies, knowledge_collection, tmp_path):
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.conversation_memory import recall_conversation
    from config import config
    from knowledge.vector_index import search_documents
    from persistence.database import session_factory
    from persistence.models import Base
    from persistence.repository import insert_chat_turn

    assert chat_replies["problem"] is None, chat_replies["problem"]
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    for name, meaning in [("rules", "Asking the allocation assistant which customers the hand "
                                    "hygiene policy protects"),
                          ("turns", "Reading back the planner's own chat turns")]:
        assert chat_replies[name]["status"] == 200, (
            meaning
            + " returned "
            + str(chat_replies[name]["status"])
            + " with "
            + str(chat_replies[name]["body"])[:300]
            + "."
        )
    reply = chat_replies["rules"]["body"]
    missing = sorted({"question_type", "account_code", "product_code", "answer",
                      "source_documents", "was_refused"} - set(reply))
    assert len(missing) == 0, (
        "The assistant's reply carries no "
        + str(missing)
        + ", so what it answered and where from cannot be read."
    )
    assert reply["was_refused"] is False and reply["question_type"] in config.CHAT_ANSWERED_TYPES, (
        "Asking which customers the hand hygiene policy protects was answered as "
        + str(reply["question_type"])
        + (" and refused" if reply["was_refused"] else "")
        + ". The hand hygiene policy answers it in a sentence of its own, so the question is "
        "neither out of scope nor further from its nearest passage than the configured distance."
    )
    folder = {name for name in os.listdir(config.DOCS_DIRECTORY) if name.lower().endswith(".pdf")}
    cited = reply["source_documents"]
    searched = search_documents(knowledge_collection["collection"],
                                [chat_replies["rules_question"]])
    grounding = (searched[0][0].get("document_name")
                 if len(searched) > 0 and len(searched[0]) > 0 else None)
    assert len(cited) > 0 and set(cited) <= folder and grounding in cited, (
        "The answer cites "
        + str(cited)
        + ". An answer names the documents it came from, each of them in the document folder, "
        "and among them the one holding the passage search_documents finds closest to the "
        "question, "
        + repr(grounding)
        + "; an answer that cannot is an answer from general knowledge."
    )
    assert not set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS), (
        "The answer cites "
        + str(sorted(set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS)))
        + ", which the assistant must not read: it records what happened in past shortages and "
        "to which customers, not what the policies say."
    )
    assert len(reply["answer"].strip()) > 0 and reply["answer"] != config.CHAT_REFUSAL_MESSAGE, (
        "The answer given was "
        + repr(reply["answer"][:120])
        + ", which answers nothing."
    )
    turns = chat_replies["turns"]["body"]
    missing = sorted({"turn_id", "question_text", "question_type", "account_code", "product_code",
                      "answer_text", "was_refused", "closest_distance", "source_documents"}
                     - set(turns[0] if len(turns) > 0 else {}))
    assert len(turns) > 0 and len(missing) == 0, (
        "The planner's chat turns read back as "
        + str(len(turns))
        + " turns"
        + ("" if len(turns) == 0 else ", the first without " + str(missing))
        + ", where each turn read back carries its identifier, the question, its type, its "
        "account and product, the answer, whether it was refused, its closest distance and the "
        "documents it cites."
    )
    asked = [turn for turn in turns if turn["question_text"] == chat_replies["rules_question"]
             and turn["turn_id"] > chat_replies["latest_turn_before"]]
    assert len(asked) > 0, (
        "The planner's latest turns read back as "
        + str([turn["question_text"] for turn in turns])
        + ", which do not include the question just answered, so the audit trail and the chat "
        "page lose it."
    )
    latest = asked[0]
    assert (latest["question_type"] == reply["question_type"] and latest["was_refused"] is False
            and latest["answer_text"] == reply["answer"]
            and latest["account_code"] == reply["account_code"]
            and latest["product_code"] == reply["product_code"]), (
        "The planner's stored turn is "
        + str({key: latest[key] for key in ["question_text", "question_type", "account_code",
                                            "product_code", "was_refused"]})
        + ", which is not how the question was just answered."
    )
    assert latest["closest_distance"] <= config.CHAT_MAX_RELEVANCE_DISTANCE, (
        "An answered question is stored with its closest passage "
        + str(latest["closest_distance"])
        + " away, beyond the "
        + str(config.CHAT_MAX_RELEVANCE_DISTANCE)
        + " within which an answer counts as grounded."
    )
    assert all(name in latest["source_documents"] for name in cited), (
        "The stored turn cites "
        + repr(latest["source_documents"])
        + " where the answer cited "
        + str(cited)
        + ". The audit trail has to show where each answer came from."
    )
    follow_up = "And does the surface disinfection policy protect anyone else?"
    with session_factory() as session:
        remembered = recall_conversation(session, chat_replies["asker"], follow_up)
        someone_else = recall_conversation(session, chat_replies["someone_else"], follow_up)
    missing = sorted({"history_text", "search_text"}
                     - set(remembered if isinstance(remembered, dict) else {}))
    assert len(missing) == 0, (
        "recall_conversation returned no "
        + str(missing)
        + ", so what the next prompt is told and what is searched for it cannot be read."
    )
    assert chat_replies["rules_question"] in remembered["history_text"], (
        "The conversation recalled for the planner's next question does not include the "
        "question just asked, so a follow-up such as "
        + repr(follow_up)
        + " would be read with no idea what it follows."
    )
    assert (follow_up in remembered["search_text"]
            and chat_replies["rules_question"] in remembered["search_text"]), (
        "The text searched for the follow-up is "
        + repr(remembered["search_text"][:200])
        + ". It has to carry the new question and the question asked just before it."
    )
    assert (chat_replies["rules_question"] not in someone_else["history_text"]
            and chat_replies["rules_question"] not in someone_else["search_text"]), (
        "The conversation recalled for a Customer Service Adviser carries the planner's "
        "question "
        + repr(chat_replies["rules_question"])
        + ". Each user's memory is their own last exchanges: a desk answer can carry what the "
        "customer service line must never hear."
    )
    status_answer = ("Reviews of product HS-0500, newest first:\nREV-PLANTED0001, HS-0500 "
                     "Wrenfield Hand Sanitiser 500 ml, divided under "
                     "Allocation-Policy-Hand-Hygiene.pdf and opened by priya: Released; 1,200 "
                     "cases available and customers may have 1,470.\nGreenacre Foodstores "
                     "(CUS-0021, Supermarket): ordered 1,000, may have 400; confirmed 293.")
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_chat.db"))
    Base.metadata.create_all(planted_engine)
    with sessionmaker(bind=planted_engine)() as session:
        insert_chat_turn(session, {
            "username": chat_replies["asker"], "user_role": config.ROLE_PLANNER,
            "question_text": "How many cases has HS-0500 released to CUS-0021?",
            "question_type": config.CHAT_TYPE_ALLOCATION_STATUS, "account_code": "CUS-0021",
            "product_code": "HS-0500", "answer_text": status_answer, "source_documents": "",
            "closest_distance": 0.5, "was_refused": False})
        after_status = recall_conversation(session, chat_replies["asker"],
                                           "And who decided it?")
    carried = [fact for fact in ["REV-PLANTED0001", "Greenacre Foodstores", "confirmed 293",
                                 "1,470"]
               if fact in after_status["history_text"]]
    assert len(carried) == 0, (
        "After a planner was told a product's allocation records, the conversation recalled for "
        "their next question carries "
        + str(carried)
        + ". The recalled conversation goes into the next prompt, and nothing from the "
        "allocation records is ever put in front of the language model, so an allocation "
        "status answer is never replayed."
    )
    assert config.CHAT_STATUS_ANSWER_IN_HISTORY in after_status["history_text"], (
        "After a planner was told an allocation's status, the conversation recalled for their "
        "next question does not carry "
        + repr(config.CHAT_STATUS_ANSWER_IN_HISTORY)
        + " where that answer stood, so the next prompt cannot tell that the question was "
        "answered from the records."
    )


def test_an_adviser_hears_an_allocation_only_for_an_account_whose_code_and_postcode_match(
        saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["heard_at_gate"]["problem"] is None, saved_run["heard_at_gate"]["problem"]
    assert saved_run["heard_after"]["problem"] is None, saved_run["heard_after"]["problem"]
    customer = saved_run["customer"]
    other = saved_run["other_customer"]
    at_gate = saved_run["at_gate"]
    final = saved_run["final"]
    lines = {line["account_code"]: line for line in saved_run["detail"]["body"]["allocation_lines"]}
    product_code = saved_run["product_code"]
    product_name = saved_run["detail"]["body"]["profile"][config.PRODUCT_NAME_COLUMN]
    desk_facts = [at_gate["review_id"], at_gate["planner_id"], final["decided_by"],
                  final["decision_reason"]]
    for moment, heard in [("while the run waited", saved_run["heard_at_gate"]),
                          ("after the release", saved_run["heard_after"])]:
        for account_code, answer in [(customer, heard["adviser"]), (other, heard["adviser_other"])]:
            if moment == "while the run waited":
                own_line = config.CHAT_ORDER_UNDER_REVIEW_TEMPLATE.format(
                    product_code=product_code, product_name=product_name,
                    cases_ordered=lines[account_code]["cases_ordered"])
            else:
                own_line = config.CHAT_ORDER_RELEASED_TEMPLATE.format(
                    product_code=product_code, product_name=product_name,
                    cases_confirmed=(final.get("cases_confirmed") or {}).get(account_code),
                    cases_ordered=lines[account_code]["cases_ordered"])
            heading = config.CHAT_ACCOUNT_HEADING_TEMPLATE.format(
                account_code=account_code, customer_name=lines[account_code]["customer_name"])
            answer_lines = str(answer).split("\n")
            others = []
            for line in lines.values():
                if line["account_code"] != account_code:
                    others.append(line["customer_name"])
                    others.append(line["account_code"])
            leaked = []
            for fact in others + desk_facts:
                if str(fact) in str(answer):
                    leaked.append(fact)
            assert (answer_lines[0] == heading and own_line in answer_lines
                    and answer_lines[-1] == config.CHAT_OTHER_ORDERS_TEXT
                    and len(leaked) == 0), (
                "A Customer Service Adviser who gave account "
                + account_code
                + " and the delivery postcode on its record "
                + moment
                + " was told "
                + repr(str(answer)[:500])
                + ", which gives away "
                + str(leaked)
                + " or is not, in the configuration's words, "
                + repr(heading)
                + " first, "
                + repr(own_line)
                + " among the orders and "
                + repr(config.CHAT_OTHER_ORDERS_TEXT)
                + " last"
                + ". The caller has proved the account, so they hear its own orders on allocation, "
                "under review until a Supply Planning Manager decides and then with the cases "
                "confirmed, and nothing about any other customer or about the desk's records."
            )
    after = saved_run["heard_after"]
    released_line = config.CHAT_ORDER_RELEASED_TEMPLATE.format(
        product_code=product_code, product_name=product_name,
        cases_confirmed=(final.get("cases_confirmed") or {}).get(customer),
        cases_ordered=lines[customer]["cases_ordered"])
    narrowed = "\n".join([config.CHAT_ACCOUNT_HEADING_TEMPLATE.format(
        account_code=customer, customer_name=lines[customer]["customer_name"]),
        released_line, config.CHAT_OTHER_ORDERS_TEXT])
    assert after["adviser_written_otherwise"] == narrowed, (
        "An adviser who gave "
        + customer
        + " with its postcode in small letters and without its space, asking about "
        + product_code
        + ", was told "
        + repr(str(after["adviser_written_otherwise"])[:300])
        + " where they hear exactly "
        + repr(narrowed)
        + ". A postcode matches its record whatever its case and spacing, and a question about "
        "one product hears that product's order alone."
    )
    for description, answer, expected in [
            ("the account's code with another customer's postcode", after["adviser_wrong_postcode"],
             config.CHAT_NO_ALLOCATION_TEMPLATE.format(account=customer)),
            ("the account's code and no postcode", after["adviser_no_postcode"],
             config.CHAT_NO_ALLOCATION_TEMPLATE.format(account=customer)),
            ("an account the customer master does not hold", after["adviser_nowhere"],
             config.CHAT_NO_ALLOCATION_TEMPLATE.format(account="CUS-9999")),
            ("a postcode and no account", after["adviser_no_account"],
             config.CHAT_ACCOUNT_NOT_NAMED_MESSAGE)]:
        assert answer == expected, (
            "A Customer Service Adviser who gave "
            + description
            + " after the release was told "
            + repr(str(answer)[:300])
            + " where they hear exactly "
            + repr(expected)
            + ". An allocation is told only to a caller who gives the account's code and the "
            "delivery postcode on its record; anyone else hears the configuration's words, the "
            "same whether the account exists, was reviewed or was not."
        )
    review_line = config.CHAT_REVIEW_TEMPLATE.format(
        review_id=at_gate["review_id"], product_code=product_code, product_name=product_name,
        policy_document=at_gate["policy_document"], planner_id=at_gate["planner_id"],
        status=config.REVIEW_STATUS_LABELS[config.REVIEW_STATUS_PENDING],
        cases_available=at_gate["cases_available"], cases_eligible=at_gate["cases_eligible"])
    awaiting_line = config.CHAT_AWAITING_DECISION_TEMPLATE.format(
        decisions_open=", ".join(at_gate["decisions_open"]),
        recommended_decision=at_gate["recommended_decision"])
    decided_review_line = config.CHAT_REVIEW_TEMPLATE.format(
        review_id=at_gate["review_id"], product_code=product_code, product_name=product_name,
        policy_document=at_gate["policy_document"], planner_id=at_gate["planner_id"],
        status=config.REVIEW_STATUS_LABELS[final["status"]],
        cases_available=at_gate["cases_available"], cases_eligible=at_gate["cases_eligible"])
    decided_line = config.CHAT_DECIDED_TEMPLATE.format(
        decided_by=final["decided_by"], decided_by_role=final["decided_by_role"],
        decision=final["decision"], decision_reason=final["decision_reason"],
        reserve_cases=final["reserve_cases"], cases_in_stock=final["cases_in_stock"])
    for moment, desk, wanted_lines in [
            ("while the run waited", str(saved_run["heard_at_gate"]["desk"]),
             [review_line, awaiting_line]),
            ("after the release", str(after["desk"]), [decided_review_line, decided_line])]:
        heading = config.CHAT_REVIEWS_HEADING_TEMPLATE.format(
            subject=config.CHAT_SUBJECT_PRODUCT_TEMPLATE.format(product_code=product_code))
        desk_lines = desk.split("\n")
        assert desk_lines[0] == heading and all(line in desk_lines for line in wanted_lines), (
            "A Supply Planner asking about "
            + product_code
            + " "
            + moment
            + " was told "
            + repr(desk[:500])
            + ". The desk hears, in the configuration's words, "
            + repr(heading)
            + " and then the review, "
            + repr(wanted_lines[0][:200])
            + ", followed by "
            + repr(wanted_lines[1][:200])
            + "."
        )
