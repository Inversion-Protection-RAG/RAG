import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Define Constant
CORPUS_SIZE    = 500   # Indexing Corpus Doc
EVAL_SIZE      = 50    # Query
TOP_K          = 5     # NDCG@K / Recall@K
VECTOR_DIM     = 1024  # BAAI/bge-m3 Output Dimension
BATCH_SIZE     = 32
MAX_LENGTH     = 512

# DP Parameter
CLIP_NORM      = 1.0   # L2-norm Clipping threshold C
GAUSSIAN_SIGMA = 0.3   # Gaussian fix σ
DP_EPSILON     = 100   # sweep: ε=100 → NDCG≈0.50, InvRate=0.17
DP_DELTA       = 1e-5
DIM_SCALE_MIN  = 0.1
DIM_SCALE_MAX  = 2.0

# ε sweep
EPSILON_SWEEP  = [1, 10, 50, 100, 200, 500]

COLLECTIONS = {
    "Natural":        "rag_natural",
    "Gaussian Noise": "rag_gaussian",
    "Adaptive Noise": "rag_adaptive_dp",
}

FIQA_QUERY_DATASET  = "orgrctera/beir_fiqa"
FIQA_CORPUS_DATASET = "BeIR/fiqa"

# Init Model
device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"[Init] Using device: {device}")

model_name = "BAAI/bge-m3"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name).to(device)
model.eval()

client = QdrantClient(host="localhost", port=6333)

# Batch Embedding
def get_embeddings_batch(texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            return_tensors="pt", max_length=MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)


def get_embedding(text: str) -> np.ndarray:
    return get_embeddings_batch([text])[0]

# DP Embedding
def clip_embedding(v: np.ndarray) -> np.ndarray:
    """L2-norm Clipping: CLIP_NORM Fix"""
    norm = np.linalg.norm(v)
    return v / max(1.0, norm / CLIP_NORM)


def compute_dimension_sensitivity(v: np.ndarray) -> np.ndarray:
    """
    Dimension Noise Scale Calculation.
    Low Importance Dimension(Absolute Value Low) → High Noise.
    DIM_SCALE_MAX=2.0 -> explosion proof.
    """
    abs_v = np.abs(v) + 1e-8
    scale = (1.0 / abs_v) / (1.0 / abs_v).mean()
    return np.clip(scale, DIM_SCALE_MIN, DIM_SCALE_MAX)


def compute_sigma(epsilon: float) -> float:
    """
    Standard Noise Scale Calculation.:
    σ = C * √(2·ln(1.25/δ)) / ε
    ε High σ Low = Noise Low = Utility High (DP Low)
    ε Low σ High  = Noise High = Privacy High (Utility Sacrifice)
    """
    return CLIP_NORM * np.sqrt(2.0 * np.log(1.25 / DP_DELTA)) / epsilon


def inject_noise(
    v: np.ndarray,
    mode: str,
    epsilon: float = DP_EPSILON,
) -> np.ndarray:
    """
    Noise Injection Per Mode.
      Natural         : No Noise
      Gaussian Noise : Uniform Gaussian v' = v + N(0, σ²I)
      Adaptive Noise : L2-Clipping → Adaptive Noise Per Dimension → L2-norm Renormalization
    """
    if mode == "Natural":
        return v.copy()

    if mode == "Gaussian Noise":
        return v + np.random.normal(0.0, GAUSSIAN_SIGMA, size=v.shape)

    if mode == "Adaptive Noise":
        # L2-norm Clipping
        v_c = clip_embedding(v)
        original_norm = np.linalg.norm(v_c)

        # Standard Gaussian Mechanism σ Calculate
        sigma = compute_sigma(epsilon)

        # Responsiveness Scale Per Dimension (DIM_SCALE_MAX=2.0 Clipping)
        dim_scale = compute_dimension_sensitivity(v_c)

        # Inject Noise
        noise = np.random.normal(0.0, sigma, size=v_c.shape) * dim_scale
        v_noisy = v_c + noise

        # L2-norm Renormalization → Direction Change in Cosine Space
        noisy_norm = np.linalg.norm(v_noisy)
        if noisy_norm > 0:
            v_noisy = v_noisy / noisy_norm * original_norm

        return v_noisy

    raise ValueError(f"Unknown mode: {mode}")


def inject_noise_batch(
    embeddings: np.ndarray,
    mode: str,
    epsilon: float = DP_EPSILON,
) -> np.ndarray:
    return np.vstack([inject_noise(v, mode, epsilon=epsilon) for v in embeddings])

# Privacy Evaluation — Inversion Success Rate
def rouge_l_score(reference: list, hypothesis: list) -> float:
    if not reference or not hypothesis:
        return 0.0
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j-1] + 1) if reference[i-1] == hypothesis[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0.0
    rec  = lcs / m if m else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def compute_inversion_score(
    original_emb: np.ndarray,
    noisy_emb: np.ndarray,
    top_k_dims: int = 50,
) -> dict:
    n_o = np.linalg.norm(original_emb)
    n_n = np.linalg.norm(noisy_emb)
    cosine_sim = (
        float(np.dot(original_emb, noisy_emb) / (n_o * n_n))
        if n_o > 0 and n_n > 0 else 0.0
    )

    ref_tokens = list(np.argsort(original_emb)[-top_k_dims:])
    hyp_tokens = list(np.argsort(noisy_emb)[-top_k_dims:])

    smoother = SmoothingFunction().method1
    bleu = sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(0.5, 0.5),
        smoothing_function=smoother,
    )
    rl = rouge_l_score(ref_tokens, hyp_tokens)
    return {"cosine_sim": cosine_sim, "bleu": bleu, "rouge_l": rl}

# Retrieval Metrics (Utility)
def compute_utility_metrics(
    retrieved_ids: list,
    true_ids: list,
    k: int = TOP_K,
) -> dict:
    true_set = set(true_ids)
    hits   = sum(1 for rid in retrieved_ids[:k] if rid in true_set)
    recall = hits / len(true_set) if true_set else 0.0

    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in true_set:
            mrr = 1.0 / rank
            break

    dcg  = sum(1.0 / np.log2(r + 1)
               for r, rid in enumerate(retrieved_ids[:k], start=1)
               if rid in true_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_set), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"ndcg": ndcg, "recall": recall, "mrr": mrr}

# 5. Load Dataset
#    Corpus: BeIR/fiqa corpus split
#    Query : orgrctera/beir_fiqa train split
print("\n[Dataset] Loading FiQA query dataset...")
query_dataset = load_dataset(FIQA_QUERY_DATASET, split="train")

print("[Dataset] Loading FiQA corpus (BeIR/fiqa)...")
try:
    corpus_raw = load_dataset(FIQA_CORPUS_DATASET, "corpus", split="corpus")
    sample_c   = corpus_raw[0]
    print(f"  Corpus columns: {list(sample_c.keys())}")
    print(f"  Sample: {dict(list(sample_c.items())[:3])}")

    id_key   = next((k for k in ("_id", "id", "doc_id") if k in sample_c), None)
    text_key = next((k for k in ("text", "content", "body", "passage") if k in sample_c), None)
    if not id_key or not text_key:
        raise KeyError(f"ID/No Text Key. Column: {list(sample_c.keys())}")

    corpus_map: dict[str, str] = {}
    for row in corpus_raw.select(range(min(CORPUS_SIZE, len(corpus_raw)))):
        did  = str(row[id_key])
        text = str(row[text_key]).strip()
        if text:
            corpus_map[did] = text
    print(f"  Corpus loaded: {len(corpus_map)} documents.")

except Exception as e:
    print(f"  [WARN] BeIR/fiqa corpus failed to load: {e}")
    print("  [WARN] fallback: query input text - corpus")
    corpus_map = {}
    for item in query_dataset.select(range(min(300, len(query_dataset)))):
        try:
            docs = json.loads(item.get("expected_output", "[]"))
        except Exception:
            continue
        for doc in (docs if isinstance(docs, list) else []):
            if isinstance(doc, dict):
                did = str(doc.get("id", ""))
                if did and did not in corpus_map:
                    corpus_map[did] = item["input"]
    print(f"  [WARN] fallback corpus: {len(corpus_map)} documents.")

# Query Evaluation Set Construction
print(f"\n[Dataset] Parsing eval queries ...")
eval_queries: list[dict] = []

for item in query_dataset.select(range(min(EVAL_SIZE * 5, len(query_dataset)))):
    if len(eval_queries) >= EVAL_SIZE:
        break
    try:
        docs = json.loads(item.get("expected_output", "[]"))
    except Exception:
        continue
    true_ids = [
        str(doc["id"])
        for doc in (docs if isinstance(docs, list) else [])
        if isinstance(doc, dict) and str(doc.get("id", "")) in corpus_map
    ]
    if true_ids:
        eval_queries.append({"query_text": item["input"], "true_ids": true_ids})

print(f"  Valid queries: {len(eval_queries)}")
if len(eval_queries) == 0:
    raise RuntimeError("[FATAL] Valid Query 0 — corpus ID & query expected_output ID Not Correct")

# Corpus Embedding
relevant_ids = {tid for q in eval_queries for tid in q["true_ids"]}
other_ids    = [cid for cid in corpus_map if cid not in relevant_ids]
np.random.shuffle(other_ids)
index_ids    = list(relevant_ids) + other_ids[: max(0, CORPUS_SIZE - len(relevant_ids))]
index_ids    = index_ids[:CORPUS_SIZE]

doc_texts_all = [corpus_map[cid] for cid in index_ids]
doc_ids_all   = index_ids

print(f"\n[Embed] Embedding {len(doc_texts_all)} corpus documents ...")
original_doc_embeddings = get_embeddings_batch(doc_texts_all)
print(f"[Embed] Shape: {original_doc_embeddings.shape}")

str_to_int: dict[str, int] = {cid: i for i, cid in enumerate(doc_ids_all)}
int_to_str: dict[int, str] = {i: cid for cid, i in str_to_int.items()}

# Evaluate Mode Function

def run_evaluation(mode: str, collection_name: str, epsilon: float = DP_EPSILON) -> dict:
    """
    corpus indexing → eval_queries evaluate → Return aggregated metrics.
    Not Inject On Query.
    """
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    noisy_doc_embs = inject_noise_batch(original_doc_embeddings, mode, epsilon=epsilon)

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=str_to_int[cid], vector=emb.tolist(),
                        payload={"doc_id": cid, "text": text})
            for cid, emb, text in zip(doc_ids_all, noisy_doc_embs, doc_texts_all)
        ],
    )

    # Privacy
    inv_scores = [compute_inversion_score(o, n)
                  for o, n in zip(original_doc_embeddings, noisy_doc_embs)]
    avg_cos  = float(np.mean([s["cosine_sim"] for s in inv_scores]))
    avg_bleu = float(np.mean([s["bleu"]       for s in inv_scores]))
    avg_rl   = float(np.mean([s["rouge_l"]    for s in inv_scores]))
    inv_rate = (avg_bleu + avg_rl) / 2.0

    # Utility + Latency
    ndcg_l, recall_l, mrr_l, lat_l = [], [], [], []
    for q in eval_queries:
        true_int = [str_to_int[s] for s in q["true_ids"] if s in str_to_int]
        if not true_int:
            continue
        # Query -> No Noise
        qemb = get_embedding(q["query_text"])
        try:
            t0 = time.time()
            res = client.query_points(
                collection_name=collection_name,
                query=qemb.tolist(),
                limit=TOP_K,
            ).points
            lat_l.append((time.time() - t0) * 1000)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue
        m = compute_utility_metrics([r.id for r in res], true_int, k=TOP_K)
        ndcg_l.append(m["ndcg"]); recall_l.append(m["recall"]); mrr_l.append(m["mrr"])

    return {
        f"NDCG@{TOP_K}":   float(np.mean(ndcg_l))          if ndcg_l  else 0.0,
        f"Recall@{TOP_K}": float(np.mean(recall_l))         if recall_l else 0.0,
        f"MRR@{TOP_K}":    float(np.mean(mrr_l))            if mrr_l   else 0.0,
        "Inversion Rate":  inv_rate,
        "Cosine Sim":      avg_cos,
        "BLEU":            avg_bleu,
        "ROUGE-L":         avg_rl,
        "Avg Latency(ms)": float(np.mean(lat_l))            if lat_l   else 0.0,
        "P95 Latency(ms)": float(np.percentile(lat_l, 95))  if lat_l   else 0.0,
        "epsilon":         epsilon,
    }


# Main Evaluation — Natural / Gaussian / Adaptive
results_summary: dict[str, dict] = {}

for mode, col in COLLECTIONS.items():
    print(f"\n{'='*60}")
    print(f"[Mode] {mode}  (ε={DP_EPSILON if mode=='Adaptive Noise' else 'N/A'})")
    print(f"{'='*60}")
    r = run_evaluation(mode, col)
    results_summary[mode] = r
    print(f"  ── Utility ──────────────────────────────────")
    print(f"    NDCG@{TOP_K}          : {r[f'NDCG@{TOP_K}']:.4f}")
    print(f"    Recall@{TOP_K}        : {r[f'Recall@{TOP_K}']:.4f}")
    print(f"    MRR@{TOP_K}           : {r[f'MRR@{TOP_K}']:.4f}")
    print(f"  ── Privacy (↓) ───────────")
    print(f"    Inversion Rate    : {r['Inversion Rate']:.4f}")
    print(f"    Cosine Sim        : {r['Cosine Sim']:.4f}")
    print(f"    BLEU              : {r['BLEU']:.4f}")
    print(f"    ROUGE-L           : {r['ROUGE-L']:.4f}")
    print(f"  ── Efficiency ───────────────────────────────")
    print(f"    Avg Latency (ms)  : {r['Avg Latency(ms)']:.2f}")
    print(f"    P95 Latency (ms)  : {r['P95 Latency(ms)']:.2f}")

# 8. ε Sweep — Adaptive Noise trade-off curve
print(f"\n\n{'='*60}")
print(f"[ε Sweep] Adaptive Noise  ε ∈ {EPSILON_SWEEP}")
print(f"{'='*60}")

sweep_results: list[dict] = []
for eps in EPSILON_SWEEP:
    col = f"rag_sweep_eps{str(eps).replace('.','_')}"
    print(f"  ε={eps:5.1f}  σ={compute_sigma(eps):.5f} ...", end=" ", flush=True)
    r = run_evaluation("Adaptive Noise", col, epsilon=eps)
    sweep_results.append(r)
    print(f"NDCG={r[f'NDCG@{TOP_K}']:.3f}, InvRate={r['Inversion Rate']:.3f}")

df_sweep = pd.DataFrame(sweep_results)


# Save Result
print(f"\n\n{'='*60}")
print("Final Comparison Table")
print(f"{'='*60}")
df_main = pd.DataFrame(results_summary).T.drop(columns=["epsilon"]).round(4)
print(df_main.to_string())
df_main.to_csv("rag_evaluation_results.csv")

print(f"\n{'='*60}")
print("ε Sweep Table")
print(f"{'='*60}")
sweep_cols = ["epsilon", f"NDCG@{TOP_K}", "Inversion Rate", "Cosine Sim",
              "BLEU", "ROUGE-L", "Avg Latency(ms)"]
print(df_sweep[sweep_cols].to_string(index=False))
df_sweep.to_csv("rag_sweep_results.csv", index=False)
print("\n[Saved] rag_evaluation_results.csv / rag_sweep_results.csv")


# Visualization
modes  = list(COLLECTIONS.keys())
colors = ["steelblue", "tomato", "seagreen"]

# Figure 1: 3×3 Comparison with Main
fig1 = plt.figure(figsize=(18, 14))
fig1.suptitle(
    f"Adaptive DP-Embedding RAG Evaluation\n"
    f"(Model: {model_name}, Top-K={TOP_K}, "
    f"ε_adaptive={DP_EPSILON}, δ={DP_DELTA}, C={CLIP_NORM}, σ_gauss={GAUSSIAN_SIGMA})",
    fontsize=13, fontweight="bold",
)
gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.55, wspace=0.38)


def bar_subplot(ax, title, ylabel, key, ylim=None):
    vals = [results_summary[m][key] for m in modes]
    bars = ax.bar(modes, vals, color=colors, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (ylim[1] * 0.01 if ylim else h * 0.02),
                f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    if ylim:
        ax.set_ylim(*ylim)


bar_subplot(fig1.add_subplot(gs1[0, 0]), f"NDCG@{TOP_K} ↑",   "Score", f"NDCG@{TOP_K}",   (0, 1.15))
bar_subplot(fig1.add_subplot(gs1[0, 1]), f"Recall@{TOP_K} ↑", "Score", f"Recall@{TOP_K}", (0, 1.15))
bar_subplot(fig1.add_subplot(gs1[0, 2]), f"MRR@{TOP_K} ↑",    "Score", f"MRR@{TOP_K}",   (0, 1.15))
bar_subplot(fig1.add_subplot(gs1[1, 0]), "Inversion Rate ↓", "Score", "Inversion Rate", (0, 1.15))
bar_subplot(fig1.add_subplot(gs1[1, 1]), "BLEU ↓",    "Score", "BLEU",    (0, 1.15))
bar_subplot(fig1.add_subplot(gs1[1, 2]), "ROUGE-L ↓", "Score", "ROUGE-L", (0, 1.15))
bar_subplot(fig1.add_subplot(gs1[2, 0]), "Avg Latency (ms) ↓", "ms", "Avg Latency(ms)")
bar_subplot(fig1.add_subplot(gs1[2, 1]), "P95 Latency (ms) ↓", "ms", "P95 Latency(ms)")

ax_sc = fig1.add_subplot(gs1[2, 2])
for m, color in zip(modes, colors):
    x = results_summary[m][f"NDCG@{TOP_K}"]
    y = 1.0 - results_summary[m]["Inversion Rate"]
    ax_sc.scatter(x, y, color=color, s=140, label=m, zorder=5)
    ax_sc.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
ax_sc.set_xlabel(f"NDCG@{TOP_K}  (Utility ↑)", fontsize=8)
ax_sc.set_ylabel("Privacy Defense Rate\n(1 − Inversion Rate ↑)", fontsize=8)
ax_sc.set_title("Utility vs Privacy Trade-off", fontsize=10)
ax_sc.legend(fontsize=7)
ax_sc.set_xlim(0, 1.05); ax_sc.set_ylim(0, 1.05)
ax_sc.annotate("← Ideal", xy=(1.0, 1.0), fontsize=7, color="gray",
               ha="right", va="top", style="italic")
fig1.savefig("rag_evaluation_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Saved] rag_evaluation_results.png")

# Figure 2: ε Sweep Trade-off curve
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    f"Adaptive DP-Embedding: ε Sweep (Privacy–Utility Trade-off)\n"
    f"(δ={DP_DELTA}, C={CLIP_NORM}, DIM_SCALE_MAX={DIM_SCALE_MAX})",
    fontsize=12, fontweight="bold",
)
eps_vals  = df_sweep["epsilon"].tolist()
ndcg_vals = df_sweep[f"NDCG@{TOP_K}"].tolist()
inv_vals  = df_sweep["Inversion Rate"].tolist()
lat_vals  = df_sweep["Avg Latency(ms)"].tolist()
sigma_vals = [compute_sigma(e) for e in eps_vals]

ax0 = axes2[0]
ax0.plot(eps_vals, ndcg_vals, "o-", color="steelblue", lw=2, ms=6)
for x, y in zip(eps_vals, ndcg_vals):
    ax0.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax0.set_xlabel("ε (Privacy Budget — log scale)", fontsize=9)
ax0.set_ylabel(f"NDCG@{TOP_K} (Utility ↑)", fontsize=9)
ax0.set_title(f"ε vs NDCG@{TOP_K}", fontsize=11)
ax0.set_xscale("log"); ax0.set_ylim(0, 1.05); ax0.grid(True, alpha=0.3)

ax0b = ax0.twiny()
ax0b.set_xlim(ax0.get_xlim())
ax0b.set_xscale("log")
ax0b.set_xticks(eps_vals)
ax0b.set_xticklabels([f"σ={s:.3f}" for s in sigma_vals], fontsize=6, rotation=30)
ax0b.set_xlabel("Corresponding σ", fontsize=7)

ax1 = axes2[1]
ax1.plot(eps_vals, inv_vals, "s-", color="tomato", lw=2, ms=6)
for x, y in zip(eps_vals, inv_vals):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax1.set_xlabel("ε (Privacy Budget — log scale)", fontsize=9)
ax1.set_ylabel("Inversion Rate ↓", fontsize=9)
ax1.set_title("ε vs Inversion Rate", fontsize=11)
ax1.set_xscale("log"); ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)

ax2 = axes2[2]
sc = ax2.scatter(ndcg_vals, [1 - v for v in inv_vals],
                 c=np.log10(eps_vals), cmap="plasma", s=120, zorder=5)
for eps, x, y in zip(eps_vals, ndcg_vals, [1 - v for v in inv_vals]):
    ax2.annotate(f"ε={eps}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
plt.colorbar(sc, ax=ax2, label="log₁₀(ε)")
ax2.set_xlabel(f"NDCG@{TOP_K}  (Utility ↑)", fontsize=9)
ax2.set_ylabel("Privacy Defense Rate (1 − Inversion ↑)", fontsize=9)
ax2.set_title("Trade-off Curve (colored by ε)", fontsize=11)
ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("rag_sweep_tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Saved] rag_sweep_tradeoff.png")