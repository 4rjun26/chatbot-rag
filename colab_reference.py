
!pip install sentence-transformers faiss-cpu openai

# 🧠 STEP 3: Define the RAG pipeline
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# 🔑 STEP 3.1: Set up OpenRouter API
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "MY-API-KEY-GOES-HERE"  # <-- Replace with your actual key
openai.default_headers = {
    "HTTP-Referer": "http://localhost",  # Use your project URL if hosted
    "X-Title": "LLaMA 4 Maverick RAG Bot"
}

# 🧩 STEP 3.2: Load & chunk text
def load_and_chunk_text(file_path, chunk_size=300):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# 🧬 STEP 3.3: Embed chunks using Sentence Transformers
def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

# 📦 STEP 3.4: Create FAISS vector index
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 🔍 STEP 3.5: Search for top-k chunks for a query
def search_index(query, embedder, index, chunks, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# 🗣️ STEP 3.6: Ask LLaMA 4 Maverick via OpenRouter
def ask_llama(question, context):
    prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question:
{question}
"""
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# 🚀 STEP 4: Run the RAG pipeline

# Replace this with the uploaded filename
file_name = "/content/maynasundari.txt"

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Build the index
chunks = load_and_chunk_text(file_name)
embeddings = embed_chunks(chunks, embedder)
index = create_faiss_index(embeddings)

# Ask your question (Gujarati or English)
user_question = "જયેન્દ્ર સુહાસે શેના વિશે વાત કરી હતી?"  # Change this to your question
top_chunks = search_index(user_question, embedder, index, chunks)
context = "\n".join(top_chunks)

# Get the final answer from LLaMA 4
answer = ask_llama(user_question, context)
print("📜 Answer:\n", answer)

# 📥 Ask your question in English
english_question = "Who is Mayna Sundari? Explain in detail."

# 🌐 Step 1: Translate English → Gujarati
translation_prompt = f"Translate the following question into Gujarati:\n\n{english_question}"
translated_question = openai.ChatCompletion.create(
    model="meta-llama/llama-3-70b-instruct",
    messages=[{"role": "user", "content": translation_prompt}],
    temperature=0.7
)['choices'][0]['message']['content'].strip()

# 🔍 Step 2: Find relevant chunks in Gujarati
top_chunks = search_index(translated_question, embedder, index, chunks)
context = "\n".join(top_chunks)

# 🧠 Step 3: Ask LLaMA the Gujarati question
gujarati_answer = ask_llama(translated_question, context)

# 🌐 Step 4: Translate Gujarati → English
back_translation_prompt = f"Translate the following Gujarati answer into English:\n\n{gujarati_answer}"
english_answer = openai.ChatCompletion.create(
    model="meta-llama/llama-3-70b-instruct",
    messages=[{"role": "user", "content": back_translation_prompt}],
    temperature=0.7
)['choices'][0]['message']['content'].strip()

# 🖨️ Final Output
# print("🗣️ Translated Question (Gujarati):\n", translated_question)
print("\n📜 Answer (Gujarati):\n", gujarati_answer)
print("\n🔁 Translated Back to English:\n", english_answer)

from translatepy import Translator

# 📥 Step 1: Ask your question in English
english_question = "Who is Mayna Sundari? Explain in detail."

# 🔍 Step 2: Find relevant chunks (RAG in English)
top_chunks = search_index(english_question, embedder, index, chunks)
context = "\n".join(top_chunks)

# 🧠 Step 3: Ask LLaMA 4 Maverick the English question
english_answer = ask_llama(english_question, context)

# 🌐 Step 4: Translate English answer → Gujarati using Google Translate

translator = Translator()
translated_answer = translator.translate(english_answer, "Gujarati").result

# 🖨️ Final Output
print("📜 English Answer:\n", english_answer)
print("\n🌐 Gujarati Translation:\n", translated_answer)

english_question = "What happens in Episode 9?"

# 🔍 Step 2: Find relevant chunks (RAG in English)
top_chunks = search_index(english_question, embedder, index, chunks)
context = "\n".join(top_chunks)

# 🧠 Step 3: Ask LLaMA 4 Maverick the English question
english_answer = ask_llama(english_question, context)

# 🌐 Step 4: Translate English answer → Gujarati using Google Translate

translator = Translator()
translated_answer = translator.translate(english_answer, "Gujarati").result

# 🖨️ Final Output
print("📜 English Answer:\n", english_answer)
print("\n🌐 Gujarati Translation:\n", translated_answer)

english_question = "Why did Mayna Punish her father?"

# 🔍 Step 2: Find relevant chunks (RAG in English)
top_chunks = search_index(english_question, embedder, index, chunks)
context = "\n".join(top_chunks)

# 🧠 Step 3: Ask LLaMA 4 Maverick the English question
english_answer = ask_llama(english_question, context)

# 🌐 Step 4: Translate English answer → Gujarati using Google Translate

translator = Translator()
translated_answer = translator.translate(english_answer, "Gujarati").result

# 🖨️ Final Output
print("📜 English Answer:\n", english_answer)
print("\n🌐 Gujarati Translation:\n", translated_answer)