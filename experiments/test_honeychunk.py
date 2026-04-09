from data.honeychunk import build_honey_chunks

real_chunks = [
    {
        "id": "chunk_1",
        "text": "The treatment increased survival rate by 12 percent in high-risk patients.",
        "metadata": {
            "dataset": "test",
            "is_honey": False
        }
    },
    {
        "id": "chunk_2",
        "text": "Positive outcomes were observed in 30 patients after 6 weeks.",
        "metadata": {
            "dataset": "test",
            "is_honey": False
        }
    },
    {
        "id": "chunk_3",
        "text": "The study reported more adverse effects in the control group.",
        "metadata": {
            "dataset": "test",
            "is_honey": False
        }
    }
]

honey_chunks = build_honey_chunks(real_chunks, ratio=0.5)

print("=== REAL CHUNKS ===")
for chunk in real_chunks:
    print(chunk["id"], chunk["metadata"]["is_honey"])
    print(chunk["text"])
    print()

print("=== HONEY CHUNKS ===")
for chunk in honey_chunks:
    print(chunk["id"], chunk["metadata"]["is_honey"])
    print(chunk["text"])
    print()