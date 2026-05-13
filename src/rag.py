"""
rag.py
------
Retrieval-Augmented Generation (RAG) pipeline using FAISS + Sentence-Transformers.

How RAG works:
  1. EMBED: Encode a reference set of IMDB reviews into 384-dim vectors
             using all-MiniLM-L6-v2 (a fast, high-quality sentence transformer)
  2. INDEX: Store vectors in a FAISS L2 index for ultra-fast nearest-neighbor search
  3. RETRIEVE: For any new review, encode it and find the top-k most similar
               reviews from the index
  4. AUGMENT: Pass the retrieved reviews as context into the LLM prompt
              so explanations can reference real examples

Why RAG?
  Without RAG, the LLM explains predictions based only on general knowledge.
  With RAG, it can ground explanations in actual similar reviews from the dataset.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Embedding model: all-MiniLM-L6-v2
#   - 384-dimensional output vectors
#   - ~80MB, very fast on CPU
#   - Trained specifically for semantic similarity tasks
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'


def build_index(reviews, labels, model_name=EMBED_MODEL_NAME, batch_size=64):
    """
    Embed a set of reviews and store them in a FAISS index.

    Args:
        reviews:    List of review strings (raw or lightly cleaned)
        labels:     List/array of binary labels (1=positive, 0=negative)
        model_name: Sentence-transformer model name
        batch_size: Encoding batch size

    Returns:
        index:      FAISS IndexFlatL2 — supports fast L2 nearest-neighbor search
        embed_model: Fitted SentenceTransformer (reuse for query encoding)
        rag_reviews: Truncated review strings stored alongside the index
        rag_labels:  Corresponding labels
    """
    embed_model = SentenceTransformer(model_name)
    rag_reviews = [r[:500] for r in reviews]  # Truncate for speed

    print(f"Embedding {len(rag_reviews):,} reviews (model: {model_name})...")
    embeddings = embed_model.encode(rag_reviews, show_progress_bar=True, batch_size=batch_size)

    # Build FAISS L2 index (L2 = Euclidean distance; smaller = more similar)
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    print(f"FAISS index built: {index.ntotal:,} vectors, dim={dimension}")
    return index, embed_model, rag_reviews, np.array(labels)


def retrieve_similar(query, index, embed_model, rag_reviews, rag_labels, top_k=3):
    """
    Find the top-k reviews most semantically similar to the query.

    Args:
        query:       New review string to search for
        index:       FAISS index from build_index()
        embed_model: SentenceTransformer from build_index()
        rag_reviews: Review strings from build_index()
        rag_labels:  Labels from build_index()
        top_k:       Number of similar reviews to return

    Returns:
        List of dicts with keys: 'review', 'sentiment', 'distance'
    """
    query_vec = embed_model.encode([query[:500]]).astype('float32')
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'review':    rag_reviews[idx][:200] + '...',
            'sentiment': 'Positive' if rag_labels[idx] == 1 else 'Negative',
            'distance':  float(distances[0][i])
        })
    return results


def format_rag_context(similar_reviews):
    """
    Format retrieved reviews into a string for injection into LLM prompts.

    Args:
        similar_reviews: List of dicts from retrieve_similar()

    Returns:
        Formatted context string
    """
    context = ""
    for i, s in enumerate(similar_reviews):
        context += f"\nExample {i+1} [{s['sentiment']}]: \"{s['review']}\"\n"
    return context
