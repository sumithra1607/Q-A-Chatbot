import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# ========= CONFIG =========
EMB_FILE = "C:/Users/sumit/Downloads/Q&A/data/embeddings.npy"
META_FILE = "C:/Users/sumit/Downloads/Q&A/data/metadata.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_DIR = "C:/Users/sumit/Downloads/Q&A/vectorstores/faiss_index"
# ==========================

# 1) Make sure output directory exists
os.makedirs(FAISS_DIR, exist_ok=True)

# 2) Load embeddings + metadata
embeddings = np.load(EMB_FILE)
metadata = [json.loads(line) for line in open(META_FILE, "r", encoding="utf-8")]

print(f"📥 Loaded {len(metadata)} metadata entries")
print(f"📥 Embedding matrix shape: {embeddings.shape}")

# 3) Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)   # L2 distance (Euclidean)
index.add(embeddings)

print(f"✅ FAISS index ready with {index.ntotal} chunks")

# 4) Save FAISS index to disk
faiss.write_index(index, os.path.join(FAISS_DIR, "faiss.index"))
print(f"💾 Saved FAISS index → {FAISS_DIR}/faiss.index")

# 5) Save metadata alongside index
with open(os.path.join(FAISS_DIR, "metadata.jsonl"), "w", encoding="utf-8") as f:
    for m in metadata:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"💾 Saved metadata → {FAISS_DIR}/metadata.jsonl")

# 6) Reload embedding model for query-time use
model = SentenceTransformer(MODEL_NAME)

# (Optional) quick test query
def search(query, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": metadata[idx]["text"],
            "file": metadata[idx]["file"],
            "page": metadata[idx]["page"],
            "score": float(distances[0][i])
        })
    return results

if __name__ == "__main__":
    query = "What are the side effects of BENTYL?"
    results = search(query, top_k=3)
    for r in results:
        print(f"\n📖 From {r['file']} (page {r['page']}) [Score: {r['score']:.2f}]")
        print(r['text'][:300], "...")
