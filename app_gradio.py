"""
app_gradio.py
-------------
Gradio web application for the Cinematic Sentiment AI system.

Launches a public Gradio interface with two tabs:
  1. Analyze Review — paste any movie review for instant sentiment analysis,
     LLM explanation, and similar review retrieval
  2. Chat with Agent — free-form conversation with the AI agent,
     which dynamically calls tools based on your question

Usage:
    python app_gradio.py

    Set these environment variables before running:
        GROQ_API_KEY  — from console.groq.com (free)
        MODEL_DIR     — path to roberta-sentiment-final/ folder (default: ./models/roberta-sentiment-final)
        RAG_SIZE      — number of reviews to embed in FAISS index (default: 5000)
"""

import os
import re
import sys
import getpass

import torch
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from rag import build_index, retrieve_similar
from llm import build_groq_client, explain_prediction
from tools import build_tool_map, ask_agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR    = os.environ.get('MODEL_DIR', './models/roberta-sentiment-final')
RAG_SIZE     = int(os.environ.get('RAG_SIZE', 5000))
GROQ_API_KEY = os.environ.get('GROQ_API_KEY') or getpass.getpass("Enter Groq API Key: ")
DATA_PATH    = os.environ.get('DATA_PATH', './data/IMDB Dataset.csv')


# ---------------------------------------------------------------------------
# Load models (done once at startup)
# ---------------------------------------------------------------------------

print("Loading RoBERTa model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
roberta_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
roberta_model.to(device)
roberta_model.eval()
print(f"RoBERTa loaded on {device}.")

print("Building RAG index...")
df = pd.read_csv(DATA_PATH)
rag_reviews_raw = df['review'].values[:RAG_SIZE]
rag_labels_raw  = (df['sentiment'].values[:RAG_SIZE] == 'positive').astype(int)
faiss_index, embed_model, rag_reviews, rag_labels = build_index(rag_reviews_raw, rag_labels_raw)
print("RAG index ready.")

print("Connecting to Groq...")
groq_client = build_groq_client(GROQ_API_KEY)
tool_map    = build_tool_map(
    roberta_model, roberta_tokenizer, device,
    faiss_index, embed_model, rag_reviews, rag_labels, groq_client
)
print("All systems ready!")


# ---------------------------------------------------------------------------
# Gradio handler functions
# ---------------------------------------------------------------------------

def analyze_review_ui(review: str):
    """Handler for the Analyze Review tab."""
    if not review.strip():
        return "⚠️ Please enter a review.", "", ""

    clean = re.sub(r'<[^>]+>', '', review)

    # RoBERTa prediction
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
    emoji = '🟢' if label == 'POSITIVE' else '🔴'
    prediction_text = f"{emoji} **{label}** ({confidence:.1%} confidence)"

    # RAG retrieval
    similar = retrieve_similar(review, faiss_index, embed_model, rag_reviews, rag_labels)
    similar_text = ""
    for i, s in enumerate(similar):
        e = "🟢" if s['sentiment'] == 'Positive' else "🔴"
        similar_text += f"{e} **Similar #{i+1}** [{s['sentiment']}]\n{s['review']}\n\n"

    # LLM explanation via Groq
    rag_context = "\n".join([f"- [{s['sentiment']}]: {s['review']}" for s in similar])
    explanation = explain_prediction(clean, label, confidence, rag_context, groq_client)

    return prediction_text, explanation, similar_text


def chat_ui(message: str, history):
    """Handler for the Chat with Agent tab."""
    if not message.strip():
        return "Please enter a question."
    try:
        return ask_agent(message, groq_client, tool_map)
    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Cinematic Sentiment AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Cinematic Sentiment AI")
    gr.Markdown(
        "*Powered by **RoBERTa** (95.75% accuracy) · "
        "**Groq/Llama 3.3 70B** · **FAISS RAG** · **Gradio***"
    )

    with gr.Tab("🔍 Analyze Review"):
        gr.Markdown("Paste any movie review to get a sentiment prediction, AI explanation, and similar reviews.")
        review_input = gr.Textbox(
            label="Movie Review",
            placeholder="Type or paste a movie review here...",
            lines=5
        )
        analyze_btn = gr.Button("🚀 Analyze", variant="primary")

        with gr.Row():
            prediction_output = gr.Markdown(label="Prediction")

        with gr.Row():
            with gr.Column():
                explanation_output = gr.Markdown(label="💡 LLM Explanation")
            with gr.Column():
                similar_output = gr.Markdown(label="📚 Similar Reviews (RAG)")

        analyze_btn.click(
            fn=analyze_review_ui,
            inputs=review_input,
            outputs=[prediction_output, explanation_output, similar_output]
        )

    with gr.Tab("💬 Chat with Agent"):
        gr.Markdown(
            "Ask the AI agent anything about movie reviews. "
            "It will dynamically call the right tools (classify, find similar, explain)."
        )
        gr.ChatInterface(
            fn=chat_ui,
            examples=[
                "Is this positive or negative? 'A stunning visual experience with a hollow plot.'",
                "Find reviews similar to: 'One of the worst movies I have ever seen.'",
                "Analyze in detail: 'The director outdid themselves with this masterful sequel.'",
            ]
        )

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
## Project Overview

This application demonstrates a complete NLP pipeline for movie review sentiment analysis.

### Models & Accuracy
| Model | Accuracy |
|---|---|
| 🏆 RoBERTa (fine-tuned) | **95.75%** |
| Logistic Regression | 90.20% |
| Bi-LSTM + GloVe | 88.05% |

### Pipeline
1. **Preprocessing** — HTML removal, lemmatization, negation-aware stopwords
2. **ML Baselines** — TF-IDF + Logistic Regression / SVM / Naive Bayes / Random Forest
3. **Deep Learning** — GloVe + SimpleRNN / LSTM / GRU / Bi-LSTM
4. **Transformer** — Fine-tuned RoBERTa-base (125M parameters)
5. **LLM** — Groq / Llama 3.3 70B for human-readable explanations
6. **RAG** — FAISS + Sentence-Transformers for context retrieval
7. **Agent** — Dynamic tool calling for complex queries
""")


if __name__ == '__main__':
    demo.launch(share=True, debug=False)
