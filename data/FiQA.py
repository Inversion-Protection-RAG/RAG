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

# USE orgrctera/beir_fiqa
dataset = load_dataset("orgrctera/beir_fiqa", split="train")

# First Dataset For Test Parsing
example = dataset[0]
query_text = example['input']

# Parsing ID List(Type: JSON)
relevant_docs = json.loads(example['expected_output'])

print(f"Query text: {query_text}")
print(f"Relevant docs IDs: {[doc['id'] for doc in relevant_docs]}")