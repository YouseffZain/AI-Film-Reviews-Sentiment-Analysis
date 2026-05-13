"""
preprocessing.py
----------------
Text cleaning pipeline for movie review sentiment analysis.

Steps applied:
  1. Remove HTML tags (IMDB reviews contain <br/> from web scraping)
  2. Lowercase
  3. Remove special characters and digits
  4. Tokenize (NLTK punkt)
  5. Negation-aware stopword removal — keeps 'not', 'no', 'never', etc.
     because they flip sentiment ('not good' != 'good')
  6. Lemmatization (WordNet)
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (silent if already present)
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

# Negation words that must NOT be removed — they reverse sentiment polarity
NEGATION_WORDS = {
    'not', 'no', 'nor', 'never', 'neither', 'nobody', 'nothing',
    'nowhere', 'none', "don't", "doesn't", "didn't", "won't",
    "wouldn't", "couldn't", "shouldn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "can't", "cannot", "mustn't", "needn't"
}

# Standard English stopwords minus the negation words above
STOP_WORDS = set(stopwords.words('english')) - NEGATION_WORDS


def clean_text(text: str) -> str:
    """
    Clean a single movie review for ML/DL models.

    Args:
        text: Raw review string (may contain HTML, punctuation, etc.)

    Returns:
        Cleaned, tokenized, lemmatized string ready for TF-IDF or embedding.

    Example:
        >>> clean_text("<br/>This was NOT a good film at all!")
        'not good film'
    """
    text = re.sub(r'<[^>]+>', '', text)       # Remove HTML tags
    text = text.lower()                        # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)      # Remove non-alpha characters
    tokens = word_tokenize(text)               # Tokenize
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in STOP_WORDS and len(w) > 2  # Remove stopwords and short tokens
    ]
    return ' '.join(tokens)


def clean_text_for_transformer(text: str) -> str:
    """
    Minimal cleaning for Transformer models (RoBERTa, BERT).

    Transformers use their own BPE tokenizer trained on raw text —
    they NEED punctuation, capitalization, and stopwords to understand context.
    We only remove HTML tags here.

    Args:
        text: Raw review string

    Returns:
        HTML-stripped text, preserving all other content.
    """
    return re.sub(r'<[^>]+>', '', text)


def clean_dataset(texts, transformer=False, verbose=True):
    """
    Apply cleaning to a list/array of review texts.

    Args:
        texts:       Iterable of raw review strings
        transformer: If True, uses minimal cleaning (for RoBERTa/BERT)
        verbose:     Print progress every 10,000 reviews

    Returns:
        List of cleaned strings
    """
    fn = clean_text_for_transformer if transformer else clean_text
    cleaned = []
    for i, text in enumerate(texts):
        cleaned.append(fn(text))
        if verbose and i > 0 and i % 10000 == 0:
            print(f"  Cleaned {i}/{len(texts)} reviews...")
    return cleaned
