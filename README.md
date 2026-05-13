# 🎬 Cinematic Sentiment AI — Movie Review Sentiment Analysis

An end-to-end NLP pipeline for binary sentiment classification of movie reviews, progressing from traditional Machine Learning baselines through Deep Learning (RNN/LSTM/GRU) to State-of-the-Art Transformer fine-tuning, with LLM-powered explanations and a RAG-enhanced interactive UI.

## 📊 Results Summary

| Model | Type | Test Accuracy |
|---|---|---|
| RoBERTa (fine-tuned) | Transformer | **95.75%** |
| BERT (fine-tuned) | Transformer | 92.19% |
| Logistic Regression | ML (TF-IDF) | 90.20% |
| Linear SVM | ML (TF-IDF) | 89.80% |
| Bi-LSTM + GloVe | DL (RNN) | 88.05% |
| Random Forest | ML (TF-IDF) | 86.70% |
| Naive Bayes | ML (TF-IDF) | 85.70% |
| LSTM + GloVe | DL (RNN) | 76.18% |
| SimpleRNN | DL (RNN) | 52.14% |
| GRU | DL (RNN) | 50.52% |

## 🗂️ Project Structure

```
movie-sentiment-ai/
├── notebooks/
│   ├── 01_Preprocessing_ML_Baselines.ipynb   # Data cleaning + ML models
│   ├── 02_RNN_Experiments.ipynb              # RNN/LSTM/GRU/Bi-LSTM + GloVe
│   ├── 03_BERT_Finetuning.ipynb              # BERT fine-tuning (Colab)
│   ├── 03b_RoBERTa_Finetuning.ipynb          # RoBERTa fine-tuning (Kaggle)
│   └── 04_LLM_RAG_UI.ipynb                  # Gradio app + Groq + FAISS
├── src/
│   ├── preprocessing.py                      # Text cleaning & tokenization
│   ├── features.py                           # Feature engineering
│   ├── train_ml.py                           # ML baseline training
│   ├── train_dl.py                           # DL model training
│   ├── evaluate.py                           # Metrics & visualization
│   ├── rag.py                                # FAISS vector database
│   ├── llm.py                                # Groq LLM integration
│   └── tools.py                              # Groq agent + tool definitions
├── models/                                   # Saved model weights (Google Drive)
├── vector_db/                                # FAISS index files
├── data/                                     # Dataset files (via kagglehub)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebooks
All notebooks are designed to run on **Google Colab** (with T4 GPU) or **Kaggle**.

- Open any notebook in Colab/Kaggle
- Follow the inline instructions
- Models are saved to Google Drive for persistence across sessions

### 3. Launch the Gradio UI
**Option A — Master Notebook (Colab):** Open `Master_Sentiment_Analysis.ipynb` in Google Colab, select T4 GPU, and run all cells. A public URL is generated at the end.

**Option B — Standalone app (local):**
```bash
export GROQ_API_KEY=your_key_here
export MODEL_DIR=./models/roberta-sentiment-final
export DATA_PATH=./data/IMDB\ Dataset.csv
python app_gradio.py
```

## 🔧 Pipeline Overview

1. **Preprocessing**: Lowercase → HTML removal → Tokenization → Stopword removal (with negation retention) → Lemmatization
2. **Feature Engineering**: TF-IDF (unigram + bigram), word count, character count, punctuation count
3. **ML Baselines**: Logistic Regression, Linear SVM, Naive Bayes, Random Forest
4. **Deep Learning**: SimpleRNN, LSTM, GRU, Bi-LSTM with GloVe embeddings (100d)
5. **Transformers**: BERT and RoBERTa fine-tuning with AdamW, warmup scheduling, and label smoothing
6. **LLM Integration**: Groq API (Llama 3) for prediction explanations
7. **RAG Pipeline**: FAISS + sentence-transformers for similar review retrieval
8. **Agent System**: Groq function-calling agent for dynamic tool selection
9. **Frontend**: Gradio web application with public share link

## 📦 Datasets

| Dataset | Size | Purpose |
|---|---|---|
| [IMDB 50K](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50,000 reviews | Primary training & evaluation |
| [Letterboxd 90K](https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000) | 90,000 reviews | Domain adaptation & generalization |
| [Metacritic 10K](https://www.kaggle.com/datasets/joyshil0599/movie-reviews-dataset-10k-scraped-data) | 10,000 reviews | Cross-platform generalization test |

## 🛠️ Technologies

- **ML**: scikit-learn, TF-IDF
- **DL**: TensorFlow/Keras, GloVe embeddings
- **Transformers**: HuggingFace (BERT, RoBERTa), PyTorch
- **LLM**: Groq API (Llama 3)
- **RAG**: FAISS, sentence-transformers
- **Agent**: Groq function calling
- **UI**: Gradio
- **Environment**: Google Colab (T4 GPU), Kaggle (T4 GPU)
