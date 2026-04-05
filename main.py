from fastapi import FastAPI
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer
import ollama
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Secure RAG Project")

# 1. Qdrant 연결 확인
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# 2. BGE-M3 임베딩 모델 로드
tokenizer = AutoTokenizer.from_pretrained(os.getenv("EMBEDDING_MODEL"))
model = AutoModel.from_pretrained(os.getenv("EMBEDDING_MODEL")).to("mps") # Mac GPU 사용

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Secure RAG System is running on M4"}

@app.get("/test-llm")
def test_llm():
    # 로컬 Ollama(Llama-3) 연동 테스트
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': '안녕! 너는 누구니?'},
    ])
    return {"response": response['message']['content']}