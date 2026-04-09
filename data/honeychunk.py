import re

SWAP_WORDS = {
    "increase": "decrease",
    "decrease": "increase",
    "high": "low",
    "low": "high",
    "positive": "negative",
    "negative": "positive",
    "more": "less",
    "less": "more",
}


def perturb_numbers(text: str) -> str:
    def replace_number(match):
        value = match.group()
        try:
            if "." in value:
                return str(round(float(value) + 1.0, 2))
            return str(int(value) + 1)
        except ValueError:
            return value

    return re.sub(r"\b\d+(\.\d+)?\b", replace_number, text, count=2)


def swap_keywords(text: str) -> str:
    new_text = text
    for src, dst in SWAP_WORDS.items():
        pattern = re.compile(rf"\b{src}\b", re.IGNORECASE)
        if pattern.search(new_text):
            new_text = pattern.sub(dst, new_text, count=1)
    return new_text


def make_decoy_text(text: str) -> str:
    decoy = perturb_numbers(text)
    decoy = swap_keywords(decoy)

    if decoy == text:
        decoy = text + " However, later evidence suggested the opposite conclusion."

    return decoy


def build_honey_chunks(real_chunks, ratio=0.3):
    if not real_chunks:
        return []

    honey_count = max(1, int(len(real_chunks) * ratio))
    honey_chunks = []

    for chunk in real_chunks[:honey_count]:
        honey_chunks.append({
            "id": f"honey_{chunk['id']}",
            "text": make_decoy_text(chunk["text"]),
            "metadata": {
                **chunk["metadata"],
                "is_honey": True,
                "source_chunk_id": chunk["id"]
            }
        })

    return honey_chunks