from fastapi import FastAPI
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer
import ollama
import os
from dotenv import load_dotenv
import torch

load_dotenv()
app = FastAPI(title="Inversion Protection RAG")

# Qdrant Connection Check
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# GE-M3 Embedding Model Load
tokenizer = AutoTokenizer.from_pretrained(os.getenv("EMBEDDING_MODEL"))

# Device Auto Choose (MAC -> mps, Windows -> cuda)
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
model = AutoModel.from_pretrained(os.getenv("EMBEDDING_MODEL")).to(device)
# Health Check
@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG is Running"}

# Local LLM(llama3) Test
@app.get("/test-llm")
def test_llm():
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': '안녕! 너는 누구니?'},
    ])
    return {"response": response['message']['content']}