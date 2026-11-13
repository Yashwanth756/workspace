"""The NLP model that reads a customer's email and names the request it makes.

The estimator (team's choice): TF-IDF of words and word pairs, then logistic
regression. The text handling is inside the one saved pipeline, so it takes
raw email texts, and any text is read as one of the request types it learned.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from modeling.model_evaluation import evaluate_text_classifier


def train_request_model(train_text, train_target, held_out_text, held_out_target):
    """Fit on the training emails only; return (fitted pipeline, scores)."""
    model = Pipeline([
        ("text", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True)),
        ("model", LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced",
                                     random_state=config.RANDOM_SEED)),
    ])
    model.fit(train_text, train_target)
    scores = evaluate_text_classifier(model, train_text, train_target, held_out_text,
                                      held_out_target)
    return model, scores