import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from translatepy import Translator
import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "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"
openai.default_headers = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "LLaMA 4 Maverick RAG Bot"
}

def load_and_chunk_text(file_path, chunk_size=300):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(query, embedder, index, chunks, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

def ask_llama(question, context):
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

def initialize_rag(text_file_path):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_and_chunk_text(text_file_path)
    embeddings = embed_chunks(chunks, embedder)
    index = create_faiss_index(embeddings)
    return embedder, index, chunks

def ask_question(question, embedder, index, chunks):
    top_chunks = search_index(question, embedder, index, chunks)
    context = "\n".join(top_chunks)
    answer_en = ask_llama(question, context)
    translated_answer = Translator().translate(answer_en, "Gujarati").result
    return answer_en, translated_answer
