import os
import json
from datasets import load_dataset
from dotenv import load_dotenv

# Load Hugging Face Access Token
load_dotenv()
token = os.getenv("HF_TOKEN")
if token is None:
    raise ValueError("Please set HF_TOKEN environment variable")
os.environ["HF_TOKEN"] = token

# [FiQA] USE orgrctera/beir_fiqa
fiqa = load_dataset("orgrctera/beir_fiqa", split="train")

# First Dataset For Test Parsing
example1 = fiqa[0]
query_text = example1['input']

# Parsing ID List(Type: JSON)
relevant_docs = json.loads(example1['expected_output'])

print(f"[FiQA] Query text: {query_text}")
print(f"[FiQA] Relevant docs IDs: {[doc['id'] for doc in relevant_docs]}")

# [PubMedQA] USE CesarCEOAI/PubMedQA
pubmedqa = load_dataset("CesarCEOAI/PubMedQA", "pqa_labeled", split="train")

# First Dataset For Test Parsing
example2 = pubmedqa[0]

print(f"[PubMedQA] Question: {example2['question']}")
print(f"[PubMedQA] Context: {example2['context']}")