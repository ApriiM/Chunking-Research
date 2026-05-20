from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from FlagEmbedding import FlagReranker
import torch

app = Flask(__name__)

# ===== DEVICE =====
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"

print(f"Using device: {device}")

# ===== MODELE =====
embed_model = SentenceTransformer("BAAI/bge-m3", device=device)

if use_fp16:
    embed_model.half()

embed_model.max_seq_length = 8192

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=use_fp16
)

# (FlagEmbedding sam wykrywa GPU, ale można wymusić)
if device == "cuda":
    reranker.model.to(device)


# ===== ENDPOINT =====
@app.route("/similarity", methods=["POST"])
def compute_similarity():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Brak JSON"}), 400

    query = data.get("query")
    documents = data.get("documents")

    if not query or not isinstance(documents, list):
        return jsonify({"error": "query i documents są wymagane"}), 400

    if not documents:
        return jsonify({"results": []})

    # ===== BEFORE =====
    query_emb = embed_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device
    )

    doc_embs = embed_model.encode(
        documents,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
        batch_size=32
    )

    scores_before = util.cos_sim(query_emb, doc_embs)[0]

    # ===== AFTER =====
    pairs = [[query, doc] for doc in documents]
    scores_after = reranker.compute_score(pairs)

    # ===== RESPONSE =====
    results = []
    for doc, s_before, s_after in zip(documents, scores_before, scores_after):
        results.append({
            "document": doc,
            "score_before": float(s_before),
            "score_after": float(s_after)
        })

    return jsonify({"results": results})


# ===== RUN =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)