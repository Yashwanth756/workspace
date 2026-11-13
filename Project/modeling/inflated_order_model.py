"""The model that predicts whether a customer will cut an order placed during a
shortage once supply recovers.

The estimator (team's choice): gradient-boosted trees. The score is held never
to fall as an order stands further above the customer's usual week, or when
the same order also arrived by another route.
"""
from sklearn.ensemble import HistGradientBoostingClassifier

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_classifier

SECOND_ROUTE_FLAG = "same_order_other_route_flag"
MONOTONIC_CONSTRAINTS = {
    "ordered_vs_usual_pct": 1,
    f"{SECOND_ROUTE_FLAG}_{config.BOOLEAN_YES}": 1,
    f"{SECOND_ROUTE_FLAG}_{config.BOOLEAN_NO}": -1,
}


def train_inflated_order_model(train_features, train_target, held_out_features,
                               held_out_target):
    """Fit on the training rows only; return (fitted pipeline, scores)."""
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, monotonic_cst=MONOTONIC_CONSTRAINTS, early_stopping=False,
        random_state=config.RANDOM_SEED)
    model = build_model_pipeline(estimator, config.INFLATED_ORDER_NUMERIC_FEATURES,
                                 config.INFLATED_ORDER_CATEGORICAL_FEATURES)
    model.fit(train_features, train_target)
    scores = evaluate_classifier(model, train_features, train_target, held_out_features,
                                 held_out_target)
    return model, scores