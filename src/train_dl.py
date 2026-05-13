"""
train_dl.py
-----------
Train GloVe-based RNN variants for sentiment analysis.

Architectures:
  - SimpleRNN  (~52%  — fails due to vanishing gradients)
  - LSTM       (~76%  — memory cell helps, still struggles with long sequences)
  - GRU        (~50%  — similar vanishing gradient issue as SimpleRNN)
  - Bi-LSTM    (~88%  — best DL model; reads sequences both forward and backward)

Key engineering decisions:
  - Post-padding (not pre-padding): cuDNN requires zeros at the END when mask_zero=True
  - SpatialDropout1D: drops entire embedding dimensions, not random neurons
  - clipnorm=1.0: prevents exploding gradients
  - ReduceLROnPlateau: auto-halves LR when validation loss plateaus
  - EarlyStopping: stops training before overfitting
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, LSTM, GRU,
    Bidirectional, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Hyperparameters
MAX_VOCAB_SIZE      = 20000   # Top-N words by frequency
MAX_SEQUENCE_LENGTH = 100     # Longer sequences cause vanishing gradients in RNNs
EMBEDDING_DIM       = 100     # GloVe dimension (matches glove.6B.100d.txt)
GLOVE_PATH          = 'glove.6B.100d.txt'


# ---------------------------------------------------------------------------
# GloVe Embeddings
# ---------------------------------------------------------------------------

def load_glove(path=GLOVE_PATH):
    """
    Load GloVe pre-trained word vectors into a dictionary.

    GloVe maps every English word to a dense vector where semantically
    similar words are geometrically close (king - man + woman ≈ queen).
    This gives our model a head-start on understanding English meaning.

    Args:
        path: Path to glove.6B.100d.txt

    Returns:
        dict: {word: np.array of shape (100,)}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GloVe file not found at '{path}'.\n"
            "Download with: wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip glove.6B.100d.txt"
        )
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            embeddings[values[0]] = np.asarray(values[1:], dtype='float32')
    print(f"Loaded {len(embeddings):,} GloVe vectors.")
    return embeddings


def build_embedding_matrix(word_index, glove_embeddings, vocab_size):
    """
    Create the embedding matrix to initialize the Keras Embedding layer.

    Row i = GloVe vector for the word with index i.
    Words not in GloVe stay as zero vectors (model learns them during training).

    Args:
        word_index:       Tokenizer.word_index dict {word: integer_id}
        glove_embeddings: Dict from load_glove()
        vocab_size:       Number of rows in the matrix

    Returns:
        np.ndarray of shape (vocab_size, EMBEDDING_DIM)
    """
    matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    found = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        vec = glove_embeddings.get(word)
        if vec is not None:
            matrix[i] = vec
            found += 1
    print(f"Matched {found:,}/{vocab_size:,} words ({found/vocab_size:.1%}) to GloVe vectors.")
    return matrix


# ---------------------------------------------------------------------------
# Tokenization & Padding
# ---------------------------------------------------------------------------

def tokenize_and_pad(X_train, X_test):
    """
    Convert cleaned text to padded integer sequences.

    Post-padding is used (zeros at the END) because cuDNN's optimized LSTM/GRU
    kernel requires mask_zero=True + post-padding. Pre-padding would crash on GPU.

    Args:
        X_train: Array of cleaned training strings
        X_test:  Array of cleaned test strings

    Returns:
        tokenizer:    Fitted Keras Tokenizer
        vocab_size:   int
        X_train_pad:  np.ndarray (n_train, MAX_SEQUENCE_LENGTH)
        X_test_pad:   np.ndarray (n_test, MAX_SEQUENCE_LENGTH)
    """
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    # Post-padding (cuDNN requirement when mask_zero=True)
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Train: {X_train_pad.shape}, Test: {X_test_pad.shape}")
    return tokenizer, vocab_size, X_train_pad, X_test_pad


# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

def build_model(arch, vocab_size, embedding_matrix):
    """
    Build a GloVe-initialized RNN model.

    Architecture choices:
      - mask_zero=True: tells LSTM/GRU to ignore padded zeros (post-padding only!)
      - SpatialDropout1D: drops entire embedding dimensions, forces the model
        to not rely on any single feature (better than neuron dropout for sequences)
      - clipnorm=1.0: caps gradient norm to prevent exploding gradients

    Args:
        arch:             One of 'RNN', 'LSTM', 'GRU', 'Bi-LSTM'
        vocab_size:       int
        embedding_matrix: np.ndarray (vocab_size, EMBEDDING_DIM)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Embedding(
        vocab_size, EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True,   # Fine-tune GloVe vectors on our data
        mask_zero=True    # Ignore padded zeros during computation
    ))
    model.add(SpatialDropout1D(0.3))

    if arch == 'RNN':
        model.add(SimpleRNN(64))
    elif arch == 'LSTM':
        model.add(LSTM(64))
    elif arch == 'GRU':
        model.add(GRU(64))
    elif arch == 'Bi-LSTM':
        # Two LSTMs: one reads left→right, one reads right→left
        # Concatenates both hidden states → captures context from both directions
        model.add(Bidirectional(LSTM(64)))
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: RNN, LSTM, GRU, Bi-LSTM")

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_all(X_train_pad, X_test_pad, y_train, y_test, embedding_matrix, vocab_size):
    """
    Train all 4 RNN architectures and return results.

    Args:
        X_train_pad:      Padded training sequences
        X_test_pad:       Padded test sequences
        y_train / y_test: Binary labels
        embedding_matrix: GloVe matrix from build_embedding_matrix()
        vocab_size:       int

    Returns:
        dl_models:    dict {arch: fitted Keras model}
        dl_histories: dict {arch: Keras History object}
        dl_results:   dict {arch: test_accuracy}
        dl_preds:     dict {arch: binary prediction array}
    """
    architectures = ['RNN', 'LSTM', 'GRU', 'Bi-LSTM']
    dl_models, dl_histories, dl_results, dl_preds = {}, {}, {}, {}

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1)
    ]

    for arch in architectures:
        print(f"\n{'='*50}\nTraining {arch}...\n{'='*50}")
        model = build_model(arch, vocab_size, embedding_matrix)
        model.summary()

        history = model.fit(
            X_train_pad, y_train,
            validation_data=(X_test_pad, y_test),
            epochs=15, batch_size=128,
            callbacks=callbacks, verbose=1
        )

        dl_models[arch]    = model
        dl_histories[arch] = history

        _, acc = model.evaluate(X_test_pad, y_test, verbose=0)
        dl_results[arch] = acc
        probs = model.predict(X_test_pad, verbose=0).flatten()
        dl_preds[arch] = (probs > 0.5).astype(int)
        print(f"\n{arch} Test Accuracy: {acc:.4f}")

    return dl_models, dl_histories, dl_results, dl_preds
