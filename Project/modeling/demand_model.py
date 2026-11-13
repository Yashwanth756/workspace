"""The model that forecasts the cases a customer will need next week.

The estimator (team's choice): gradient-boosted trees under a Poisson loss, so
no forecast is below nought. The forecast is held never to fall as the
four-week average rises, or when a promotion is planned next week.
"""
from sklearn.ensemble import HistGradientBoostingRegressor

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_regressor

PROMOTION_FLAG = "promotion_next_week_flag"
MONOTONIC_CONSTRAINTS = {
    "avg_4_week_cases": 1,
    f"{PROMOTION_FLAG}_{config.BOOLEAN_YES}": 1,
    f"{PROMOTION_FLAG}_{config.BOOLEAN_NO}": -1,
}


def train_demand_model(train_features, train_target, held_out_features, held_out_target):
    """Fit on the training rows only; return (fitted pipeline, scores)."""
    estimator = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=20, monotonic_cst=MONOTONIC_CONSTRAINTS, early_stopping=False,
        random_state=config.RANDOM_SEED)
    model = build_model_pipeline(estimator, config.DEMAND_NUMERIC_FEATURES,
                                 config.DEMAND_CATEGORICAL_FEATURES)
    model.fit(train_features, train_target)
    scores = evaluate_regressor(model, train_features, train_target, held_out_features,
                                held_out_target)
    return model, scores