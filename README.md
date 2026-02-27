# 🐝 ArXHive : Semantic Search for ArXiv

![Image](public/arxhive_logo.png)

A lightweight **RAG-based semantic search app** for exploring ArXiv abstracts.

Ask a research question → get a grounded AI answer + the most relevant papers.

---

## 🧠 How It Works

```text
User Query
  ↓
Preprocessing (tokenize + lemmatize)
  ↓
Sentence Embeddings (all-MiniLM-L6-v2)
  ↓
Vector Search (Chroma + cosine similarity)
  ↓
Top-K Abstracts
  ↓
LLM (DeepSeek via OpenRouter)
  ↓
Answer + Sources
```

Abstracts are embedded into a high-dimensional vector space.
Your query is embedded the same way.
Nearest-neighbor search retrieves the most relevant research before generation.

---

## ✨ Features

* 🔎 Meaning-based (not keyword) search
* ⚡ GPU-supported batch embeddings
* 💾 Persistent Chroma vector DB (`db_v1/`)
* 🤖 RetrievalQA with LangChain
* 🖥 Simple Streamlit UI

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/arxhive-semantic-search.git
cd arxhive-semantic-search
pip install -r requirements.txt
export OPENAI_API_KEY="your_openrouter_key"
streamlit run app.py
```

Open `http://localhost:8501`

---

