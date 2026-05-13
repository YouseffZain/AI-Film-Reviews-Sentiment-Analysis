"""
llm.py
------
LLM integration for generating human-readable sentiment explanations.

Uses Groq (Llama 3.3 70B) as the LLM backend.
Groq provides:
  - Very generous free tier
  - OpenAI-compatible API
  - Sub-second inference (runs on dedicated LPU hardware)

The LLM is used to explain WHY a review was classified a certain way,
combining the RoBERTa prediction with RAG-retrieved similar reviews
to ground the explanation in real examples from the dataset.
"""

from groq import Groq


def build_groq_client(api_key: str) -> Groq:
    """
    Initialize and return a Groq API client.

    Args:
        api_key: Groq API key (get free at console.groq.com)

    Returns:
        Groq client instance
    """
    return Groq(api_key=api_key)


def explain_prediction(
    review: str,
    label: str,
    confidence: float,
    rag_context: str,
    groq_client: Groq,
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Generate a human-readable explanation for a sentiment prediction.

    The prompt combines:
      - The original review text
      - RoBERTa's prediction and confidence
      - Top-3 similar reviews from the FAISS RAG database (as context)

    Args:
        review:      Raw review text (lightly cleaned, HTML removed)
        label:       'POSITIVE' or 'NEGATIVE'
        confidence:  float (e.g. 0.957)
        rag_context: Formatted similar reviews from rag.format_rag_context()
        groq_client: Groq client from build_groq_client()
        model:       Groq model name

    Returns:
        Explanation string (3 sentences)
    """
    prompt = f"""You are a sentiment analysis expert. A model predicted the following movie review as {label} with {confidence:.1%} confidence.

Review: "{review[:1000]}"

Here are 3 similar reviews from our database for context:
{rag_context}

In exactly 3 sentences, explain WHY this review is {label}. Reference the similar reviews to support your analysis. Quote specific emotionally charged words or phrases from the review."""

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Explanation unavailable: {e}"


def classify_and_explain(
    review: str,
    roberta_model,
    roberta_tokenizer,
    device,
    faiss_index,
    embed_model,
    rag_reviews,
    rag_labels,
    groq_client: Groq
):
    """
    Full pipeline: RoBERTa prediction + FAISS retrieval + Groq explanation.

    This is the main inference function used by the Gradio UI and the agent.

    Args:
        review:            Raw review string
        roberta_model:     Loaded HuggingFace RoBERTa model
        roberta_tokenizer: Loaded HuggingFace tokenizer
        device:            torch.device ('cuda' or 'cpu')
        faiss_index:       FAISS index from rag.build_index()
        embed_model:       SentenceTransformer from rag.build_index()
        rag_reviews:       Review strings from rag.build_index()
        rag_labels:        Labels from rag.build_index()
        groq_client:       Groq client from build_groq_client()

    Returns:
        label:       'POSITIVE' or 'NEGATIVE'
        confidence:  float
        explanation: str
        similar:     list of dicts from rag.retrieve_similar()
    """
    import re
    import torch
    from rag import retrieve_similar, format_rag_context

    # Step 1: Clean (HTML only for Transformer)
    clean = re.sub(r'<[^>]+>', '', review)

    # Step 2: RoBERTa prediction
    encoding = roberta_tokenizer(
        clean, truncation=True, padding=True,
        max_length=256, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = roberta_model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    label = 'POSITIVE' if pred_idx == 1 else 'NEGATIVE'

    # Step 3: RAG retrieval
    similar = retrieve_similar(review, faiss_index, embed_model, rag_reviews, rag_labels)
    rag_context = format_rag_context(similar)

    # Step 4: LLM explanation
    explanation = explain_prediction(clean, label, confidence, rag_context, groq_client)

    return label, confidence, explanation, similar
