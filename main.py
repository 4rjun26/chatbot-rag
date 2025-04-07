from fastapi import FastAPI
from pydantic import BaseModel
from rag_utils import initialize_rag, ask_question
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Add CORS middleware before defining routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load RAG pipeline once at startup
embedder, index, chunks = initialize_rag("maynasundari.txt")

class QuestionRequest(BaseModel):
    question: str

# ✅ RAG endpoint
@app.post("/ask")
def ask(req: QuestionRequest):
    answer_en, answer_gu = ask_question(req.question, embedder, index, chunks)
    return {
        "answer_en": answer_en,
        "answer_gu": answer_gu
    }
