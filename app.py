import streamlit as st
import os
import sys
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

PERSIST_DIR = "db_v1"

# Custom embedding class with GPU support and batch processing
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: torch.device = "cuda", batch_size: int = 32):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        # Process in batches for large datasets
        return [
            emb.tolist()
            for i in range(0, len(documents), self.batch_size)
            for emb in self.model.encode(
                documents[i:i+self.batch_size], convert_to_tensor=False, show_progress_bar=False
            )
        ]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()

# Setup
st.set_page_config(page_title="Semantic Search App", layout="centered")
st.title("🐝 ArXHive Semantic Search 📚")

# Download NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@st.cache_resource
def load_and_prepare_docs():
    dataset = load_dataset("neuralwork/arxiver")
    df = dataset["train"].to_pandas()
    abstracts = dataset["train"]["abstract"]
    processed_abstracts = [preprocess_text(abs_) for abs_ in abstracts]
    df["processed_abstract"] = processed_abstracts

    documents = df.apply(lambda row: Document(
        page_content=row["processed_abstract"],
        metadata={
            "id": row["id"],
            "title": row["title"],
            "authors": row["authors"],
            "published_date": row["published_date"],
            "link": row["link"],
            "abstract": row["abstract"],
        }), axis=1).tolist()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    return chunks

@st.cache_resource
def load_vector_db(_chunks):
    embedding = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=128
    )
    if _chunks is None or len(_chunks) == 0:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        vectordb = Chroma.from_documents(
            documents=_chunks,
            embedding=embedding,
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )
        vectordb.persist()
    return vectordb

@st.cache_resource
def load_qa_chain(_vectordb, api_key):
    retriever = _vectordb.as_retriever()
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Get API key
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.text_input("🔑 Enter your OpenRouter API key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

if api_key:
    chunks = None
    if not os.path.exists(PERSIST_DIR) or not os.path.isdir(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        with st.spinner("Loading and preprocessing ArXiv dataset... this may take a few minutes..."):
            chunks = load_and_prepare_docs()
    with st.spinner("Loading embedding model and vector database... hang on for a few more minutes..."):
        vectordb = load_vector_db(chunks)
    with st.spinner("Loading LLM and Retrieval QA chain... almost there..."):
        qa_chain = load_qa_chain(vectordb, api_key)

    query = st.text_input("Ask a research question:", placeholder="e.g., What is the dominant approach for image segmentation?")
    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.invoke(query)
            st.markdown("### ✅ Answer")
            st.markdown(answer.get("result", "No answer found."))

            st.markdown("---\n### 📄 Top Matching Abstracts")
            matches = vectordb.similarity_search(query, k=5)
            for doc in matches:
                st.markdown(f"**Title:** {doc.metadata.get('title', 'N/A')}")
                st.markdown(f"**Authors:** {doc.metadata.get('authors', 'N/A')}")
                st.markdown(f"**Published date:** {doc.metadata.get('published_date', 'N/A')}")
                st.markdown(f"**Link:** {doc.metadata.get('link', 'N/A')}")
                st.markdown(f"> {doc.metadata.get('abstract', doc.page_content)[:500]}...")
                st.markdown("---")
