"""
train_ml.py
-----------
Train and evaluate ML baseline classifiers for sentiment analysis.

Models trained:
  - Logistic Regression  (best ML model: 90.2%)
  - Linear SVM           (89.8%)
  - Naive Bayes          (85.7%)
  - Random Forest        (86.7%)

All models use TF-IDF features from features.py.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import build_tfidf


def get_ml_models():
    """
    Return a dict of instantiated (unfitted) ML classifiers.

    Returns:
        dict: {model_name: sklearn_model}
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM':          LinearSVC(max_iter=2000, random_state=42),
        'Naive Bayes':         MultinomialNB(),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }


def train_all(X_train, X_test, y_train, y_test, max_features=5000, ngram_range=(1, 2)):
    """
    Vectorize text with TF-IDF, train all 4 ML models, and evaluate.

    Args:
        X_train:      Cleaned training review strings
        X_test:       Cleaned test review strings
        y_train:      Binary labels (1=positive, 0=negative) for train
        y_test:       Binary labels for test
        max_features: TF-IDF vocabulary size
        ngram_range:  TF-IDF n-gram range

    Returns:
        results: dict {model_name: accuracy}
        models:  dict {model_name: fitted_model}
        preds:   dict {model_name: prediction_array}
        tfidf:   fitted TfidfVectorizer (for inference)
        X_test_tfidf: sparse matrix (for ROC-AUC)
    """
    print("Building TF-IDF features...")
    tfidf, X_train_tfidf, X_test_tfidf = build_tfidf(
        X_train, X_test, max_features=max_features, ngram_range=ngram_range
    )

    ml_models = get_ml_models()
    results, models, preds = {}, {}, {}

    for name, model in ml_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)

        results[name] = acc
        models[name]  = model
        preds[name]   = y_pred

        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    return results, models, preds, tfidf, X_test_tfidf


def predict_single(text, tfidf, model):
    """
    Run inference on a single cleaned review string.

    Args:
        text:  Cleaned review (output of preprocessing.clean_text)
        tfidf: Fitted TfidfVectorizer
        model: Fitted sklearn classifier

    Returns:
        label:      'POSITIVE' or 'NEGATIVE'
        confidence: float (probability if available, else decision score)
    """
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    label = 'POSITIVE' if pred == 1 else 'NEGATIVE'

    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(vec)[0][pred]
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(vec)[0]
        # Normalize decision score to [0, 1] via sigmoid
        confidence = float(1 / (1 + np.exp(-abs(score))))
    else:
        confidence = 1.0

    return label, confidence
