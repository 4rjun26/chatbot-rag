import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from translatepy import Translator
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# Setup OpenRouter API
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.default_headers = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "LLaMA 4 Maverick RAG Bot"
}

# Load and chunk text into 300-word chunks (limit total for memory)
def load_and_chunk_text(file_path, chunk_size=300, max_words=3000):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.split()[:max_words]
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed chunks in batches (reduce memory)
def embed_chunks(chunks, model, batch_size=8):
    return model.encode(chunks, convert_to_numpy=True, batch_size=batch_size)

# Create FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Search top-k chunks
def search_index(query, embedder, index, chunks, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# Ask OpenRouter (LLaMA-3)
def ask_llama(question, context):
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-8b-instruct:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Initialize RAG
def initialize_rag(text_file_path):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_and_chunk_text(text_file_path)
    embeddings = embed_chunks(chunks, embedder)
    index = create_faiss_index(embeddings)
    return embedder, index, chunks

# Ask question and translate
def ask_question(question, embedder, index, chunks):
    top_chunks = search_index(question, embedder, index, chunks)
    context = "\n".join(top_chunks)
    answer_en = ask_llama(question, context)
    translated_answer = Translator().translate(answer_en, "Gujarati").result
    return answer_en, translated_answer
