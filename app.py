import streamlit as st
import os
import re
import json
import torch
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

# ---- HARD MEMORY CONTROLS ----
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INDEX_FILE = "faiss.index"
META_FILE = "metadata.jsonl"

st.set_page_config(page_title="ArXHive Semantic Search")
st.title("🐝 ArXHive Semantic Search 📚")

# -----------------------------
# Preprocess
# -----------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    return text.strip()

# -----------------------------
# Load embedding model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# -----------------------------
# Build or Load FAISS
# -----------------------------
@st.cache_resource(show_spinner=True)
def build_or_load_index(_model):

    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        return index

    st.info("Building FAISS index (first run only)...")

    dataset = load_dataset("neuralwork/arxiver", split="train", streaming=True)

    dim = 384  # all-MiniLM-L6-v2 dimension
    index = faiss.IndexFlatIP(dim)

    metadata_file = open(META_FILE, "w")

    batch_texts = []
    batch_meta = []
    batch_size = 512

    for row in dataset:
        text = preprocess(row["abstract"])

        batch_texts.append(text)
        batch_meta.append({
            "title": row["title"],
            "authors": row["authors"],
            "published_date": row["published_date"],
            "link": row["link"],
            "abstract": row["abstract"],
        })

        if len(batch_texts) >= batch_size:
            embeddings = _model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            index.add(embeddings)

            for m in batch_meta:
                metadata_file.write(json.dumps(m) + "\n")

            batch_texts = []
            batch_meta = []

    if batch_texts:
        embeddings = _model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
        index.add(embeddings)
        for m in batch_meta:
            metadata_file.write(json.dumps(m) + "\n")

    metadata_file.close()
    faiss.write_index(index, INDEX_FILE)

    return index

# -----------------------------
# Search
# -----------------------------
def search(query, model, index, k=5):
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vec, k)

    results = []
    with open(META_FILE, "r") as f:
        all_meta = f.readlines()

    for idx in indices[0]:
        if idx < len(all_meta):
            results.append(json.loads(all_meta[idx]))

    return results

# -----------------------------
# LLM
# -----------------------------
@st.cache_resource
def load_llm(api_key):
    return ChatOpenAI(
        model="deepseek/deepseek-r1",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

# -----------------------------
# API KEY
# -----------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = st.text_input("🔑 Enter OpenRouter API key", type="password")

if api_key:

    model = load_model()
    index = build_or_load_index(model)
    llm = load_llm(api_key)

    query = st.text_input("Ask a research question:")

    if query:
        with st.spinner("Searching..."):
            results = search(query, model, index)

            context = "\n\n".join(r["abstract"] for r in results)

            prompt = f"""
            Use the following research abstracts to answer the question.

            {context}

            Question: {query}
            """

            response = llm.invoke(prompt)

            st.markdown("### ✅ Answer")
            st.write(response.content)

            st.markdown("### 📄 Top Matches")
            for r in results:
                st.markdown(f"**{r['title']}**")
                st.write(r["abstract"][:400] + "...")
                st.markdown("---")