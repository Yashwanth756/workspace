"""Trains and saves all three models and reports whether each reaches the
minimum config sets for it.

Run from the Project folder, after run_data_pipeline.py:
    python3 run_model_training.py
"""
import sklearn
from sklearn.model_selection import train_test_split

from config import config
from data_pipeline.dataset_loader import load_processed_dataset
from data_pipeline.feature_builder import build_feature_frame
from modeling.demand_model import train_demand_model
from modeling.inflated_order_model import train_inflated_order_model
from modeling.model_registry import save_feature_schema, save_metrics, save_model
from modeling.request_model import train_request_model


def _record(algorithm, training_rows, held_out_rows, scores, score_name, minimum):
    return {"algorithm": algorithm, "training_rows": int(training_rows),
            "held_out_rows": int(held_out_rows), "scores": scores,
            "minimum_required": minimum, "meets_minimum": bool(scores[score_name] >= minimum),
            "scikit_learn_version": sklearn.__version__}


def main():
    print("training three models")
    metrics = {}

    orders_train = load_processed_dataset(config.CLEAN_ORDER_HISTORY_TRAIN_FILE)
    orders_held_out = load_processed_dataset(config.CLEAN_ORDER_HISTORY_EVAL_FILE)
    order_train_x = build_feature_frame(orders_train, config.INFLATED_ORDER_NUMERIC_FEATURES,
                                        config.INFLATED_ORDER_CATEGORICAL_FEATURES)
    order_held_x = build_feature_frame(orders_held_out, config.INFLATED_ORDER_NUMERIC_FEATURES,
                                       config.INFLATED_ORDER_CATEGORICAL_FEATURES)
    order_train_y = orders_train[config.ORDER_TARGET_COLUMN].map(
        config.ORDER_TARGET_ENCODING).astype(int)
    order_held_y = orders_held_out[config.ORDER_TARGET_COLUMN].map(
        config.ORDER_TARGET_ENCODING).astype(int)
    model, scores = train_inflated_order_model(order_train_x, order_train_y, order_held_x,
                                               order_held_y)
    save_model(model, config.INFLATED_ORDER_MODEL_FILE)
    save_feature_schema(list(order_train_x.columns), config.INFLATED_ORDER_SCHEMA_FILE)
    metrics["inflated_order"] = _record(type(model.steps[-1][1]).__name__, len(order_train_x),
                                        len(order_held_x), scores, "held_out_roc_auc",
                                        config.MINIMUM_INFLATED_ORDER_ROC_AUC)
    print(f"  inflated order  held-out ROC AUC {scores['held_out_roc_auc']:.4f}  "
          f"(train {scores['train_roc_auc']:.4f}, gap {scores['generalisation_gap']:+.4f})")

    demand_train = load_processed_dataset(config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE)
    demand_held_out = load_processed_dataset(config.CLEAN_WEEKLY_DEMAND_EVAL_FILE)
    demand_train_x = build_feature_frame(demand_train, config.DEMAND_NUMERIC_FEATURES,
                                         config.DEMAND_CATEGORICAL_FEATURES)
    demand_held_x = build_feature_frame(demand_held_out, config.DEMAND_NUMERIC_FEATURES,
                                        config.DEMAND_CATEGORICAL_FEATURES)
    model, scores = train_demand_model(demand_train_x,
                                       demand_train[config.DEMAND_TARGET_COLUMN].astype(float),
                                       demand_held_x,
                                       demand_held_out[config.DEMAND_TARGET_COLUMN].astype(float))
    save_model(model, config.DEMAND_MODEL_FILE)
    save_feature_schema(list(demand_train_x.columns), config.DEMAND_SCHEMA_FILE)
    metrics["demand"] = _record(type(model.steps[-1][1]).__name__, len(demand_train_x),
                                len(demand_held_x), scores, "held_out_r2",
                                config.MINIMUM_DEMAND_R2)
    print(f"  demand          held-out R2 {scores['held_out_r2']:.4f}  "
          f"MAE {scores['mean_absolute_error']:.2f} cases  "
          f"(train {scores['train_r2']:.4f}, gap {scores['generalisation_gap']:+.4f})")

    emails = load_processed_dataset(config.CLEAN_CUSTOMER_EMAILS_FILE)
    train_text, held_out_text, train_target, held_out_target = train_test_split(
        emails[config.EMAIL_TEXT_COLUMN], emails[config.EMAIL_TARGET_COLUMN],
        test_size=config.EMAIL_HELD_OUT_SHARE, stratify=emails[config.EMAIL_TARGET_COLUMN],
        random_state=config.RANDOM_SEED)
    model, scores = train_request_model(train_text, train_target, held_out_text,
                                        held_out_target)
    save_model(model, config.REQUEST_MODEL_FILE)
    algorithm = " + ".join(type(step).__name__ for _, step in model.steps)
    metrics["request"] = _record(algorithm, len(train_text), len(held_out_text), scores,
                                 "held_out_macro_f1", config.MINIMUM_REQUEST_MACRO_F1)
    print(f"  request         held-out macro F1 {scores['held_out_macro_f1']:.4f}  "
          f"(train {scores['train_macro_f1']:.4f}, gap {scores['generalisation_gap']:+.4f})")

    save_metrics(metrics)
    print()
    print(f"metrics written to {config.MODEL_METRICS_FILE}")
    below = [name for name, record in metrics.items() if not record["meets_minimum"]]
    if below:
        print("below the minimum the configuration sets: " + ", ".join(below))
    else:
        print("every model reaches the minimum the configuration sets for it")


if __name__ == "__main__":
    main()