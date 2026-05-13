"""
tools.py
--------
Tool definitions for the LangChain / Groq AI agent system.

The agent receives a user query and dynamically decides which tool(s) to call:
  - classify_review:    "Is this review positive or negative?"
  - find_similar:       "Find reviews similar to this one"
  - explain_sentiment:  "Why is this review negative?"

The agent uses Groq's OpenAI-compatible function calling API with
Llama 3.3 70B as the backbone LLM.
"""

import re
import json
import torch
from groq import Groq


# JSON schema definitions for Groq function calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "classify_review",
            "description": (
                "Classify a movie review as POSITIVE or NEGATIVE using our "
                "fine-tuned RoBERTa model (95.75% accuracy). Returns the label and confidence score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "review": {"type": "string", "description": "The movie review text to classify"}
                },
                "required": ["review"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar",
            "description": (
                "Find the top 3 most semantically similar movie reviews from the FAISS vector database. "
                "Useful for finding context or examples."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "review": {"type": "string", "description": "The movie review text to search for"}
                },
                "required": ["review"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_sentiment",
            "description": (
                "Explain why a movie review has a certain sentiment. "
                "Uses RoBERTa for prediction, FAISS for similar examples, and Groq LLM for reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "review": {"type": "string", "description": "The movie review text to explain"}
                },
                "required": ["review"]
            }
        }
    }
]

SYSTEM_PROMPT = (
    "You are a movie review sentiment analysis assistant. "
    "You have access to tools for classifying reviews, finding similar reviews, "
    "and explaining predictions. Use the appropriate tool(s) based on what the user asks."
)


def build_tool_map(roberta_model, roberta_tokenizer, device,
                   faiss_index, embed_model, rag_reviews, rag_labels,
                   groq_client):
    """
    Build a dict mapping tool names → callable functions.

    All tools are closures that capture the model/index references
    so they can be called with just the review text.

    Args:
        roberta_model / roberta_tokenizer: Loaded HuggingFace RoBERTa
        device:      torch.device
        faiss_index: FAISS index from rag.build_index()
        embed_model: SentenceTransformer from rag.build_index()
        rag_reviews / rag_labels: from rag.build_index()
        groq_client: Groq client

    Returns:
        dict: {tool_name: callable(review) -> str}
    """
    from rag import retrieve_similar, format_rag_context
    from llm import explain_prediction

    def classify_review(review: str) -> str:
        """RoBERTa sentiment classification."""
        clean = re.sub(r'<[^>]+>', '', review)
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
        return f"Prediction: {label} | Confidence: {confidence:.1%}"

    def find_similar(review: str) -> str:
        """FAISS semantic similarity search."""
        similar = retrieve_similar(review, faiss_index, embed_model, rag_reviews, rag_labels)
        result = ""
        for i, s in enumerate(similar):
            result += f"\n{i+1}. [{s['sentiment']}] {s['review']}\n"
        return result

    def explain_sentiment(review: str) -> str:
        """RoBERTa + RAG + Groq explanation."""
        clean = re.sub(r'<[^>]+>', '', review)
        # Get prediction
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
        # Get RAG context
        similar = retrieve_similar(review, faiss_index, embed_model, rag_reviews, rag_labels)
        rag_ctx = format_rag_context(similar)
        # Get explanation
        explanation = explain_prediction(clean, label, confidence, rag_ctx, groq_client)
        return f"Prediction: {label} ({confidence:.1%})\n\nExplanation: {explanation}"

    return {
        'classify_review':   classify_review,
        'find_similar':      find_similar,
        'explain_sentiment': explain_sentiment,
    }


def ask_agent(query: str, groq_client: Groq, tool_map: dict,
              model: str = "llama-3.3-70b-versatile") -> str:
    """
    Run a user query through the Groq function-calling agent.

    The agent:
      1. Sends the user query + tool schemas to the LLM
      2. The LLM decides which tool(s) to call and with what arguments
      3. We execute the tool and send results back
      4. Repeat until the LLM returns a final text response

    Args:
        query:       User's natural language question
        groq_client: Groq client
        tool_map:    Dict from build_tool_map()
        model:       Groq model name

    Returns:
        Final text response from the agent
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=model, messages=messages,
            tools=TOOLS_SCHEMA, tool_choice="auto"
        )
    except Exception:
        # Fallback: answer without tools if schema fails
        response = groq_client.chat.completions.create(
            model=model, messages=messages
        )
        return response.choices[0].message.content

    msg = response.choices[0].message

    # Handle tool call loop
    while msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"🔧 Agent calling: {fn_name}")
            result = tool_map[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

        try:
            response = groq_client.chat.completions.create(
                model=model, messages=messages,
                tools=TOOLS_SCHEMA, tool_choice="auto"
            )
        except Exception:
            response = groq_client.chat.completions.create(
                model=model, messages=messages
            )
            return response.choices[0].message.content

        msg = response.choices[0].message

    return msg.content
