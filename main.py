from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_utils import initialize_rag, ask_question
import uvicorn

app = FastAPI()

# Load all models and index once
embedder, index, chunks = initialize_rag("maynasundari.txt")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer_en, answer_gu = ask_question(query.question, embedder, index, chunks)
    return {"english_answer": answer_en, "gujarati_translation": answer_gu}

# Run locally if needed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
