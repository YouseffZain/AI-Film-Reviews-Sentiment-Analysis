"""
features.py
-----------
Feature engineering for movie review sentiment analysis.

Extracts two types of features:
  1. TF-IDF vectors  — sparse bag-of-words representation for ML models
  2. Linguistic features — handcrafted numerical signals derived from writing style

These are used by the ML baseline classifiers (Logistic Regression, SVM, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# TF-IDF Vectorization
# ---------------------------------------------------------------------------

def build_tfidf(X_train, X_test, max_features=5000, ngram_range=(1, 2)):
    """
    Fit a TF-IDF vectorizer on training data and transform both splits.

    TF-IDF (Term Frequency × Inverse Document Frequency):
      - TF:  How often a word appears in one document (signals local importance)
      - IDF: How rare a word is across all documents (signals distinctiveness)
      - ngram_range=(1,2): Also captures bigrams like 'not good', 'waste time'

    Args:
        X_train:      Array of cleaned training review strings
        X_test:       Array of cleaned test review strings
        max_features: Vocabulary size cap (top-N words by frequency)
        ngram_range:  Tuple (min_n, max_n) for n-gram extraction

    Returns:
        tfidf:          Fitted TfidfVectorizer (use to transform new text)
        X_train_tfidf:  Sparse matrix, shape (n_train, max_features)
        X_test_tfidf:   Sparse matrix, shape (n_test, max_features)
    """
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return tfidf, X_train_tfidf, X_test_tfidf


# ---------------------------------------------------------------------------
# Linguistic Feature Extraction
# ---------------------------------------------------------------------------

def extract_linguistic_features(texts):
    """
    Extract 7 writing-style features from raw (uncleaned) review texts.

    Features and their sentiment signal:
      word_count       — negative reviews tend to be longer (people rant more)
      char_count       — correlated with word_count
      punct_count      — emotional reviews use more punctuation
      avg_word_length  — more sophisticated vocabulary = more nuanced sentiment
      exclamation_count — excitement (positive) or frustration (negative)
      question_count   — rhetorical questions often signal criticism
      capital_ratio    — SHOUTING indicates strong (usually negative) emotion

    Args:
        texts: Iterable of raw (original, uncleaned) review strings

    Returns:
        pd.DataFrame with 7 feature columns, one row per review
    """
    records = []
    for text in texts:
        words = text.split()
        records.append({
            'word_count':        len(words),
            'char_count':        len(text),
            'punct_count':       sum(1 for c in text if c in '.,!?;:'),
            'avg_word_length':   np.mean([len(w) for w in words]) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count':    text.count('?'),
            'capital_ratio':     sum(1 for c in text if c.isupper()) / (len(text) + 1),
        })
    return pd.DataFrame(records)


LINGUISTIC_FEATURE_COLS = [
    'word_count', 'char_count', 'punct_count', 'avg_word_length',
    'exclamation_count', 'question_count', 'capital_ratio'
]
