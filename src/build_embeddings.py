import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ========= CONFIG =========
INPUT_FOLDER = "C:/Users/sumit/Downloads/Q&A/data/processed_chunks"     # your JSON chunk files
OUT_EMB_NPY = "C:/Users/sumit/Downloads/Q&A/data/embeddings.npy"        # output embeddings
OUT_META_JSONL = "C:/Users/sumit/Downloads/Q&A/data/metadata.jsonl"     # output metadata
MODEL_NAME = "all-MiniLM-L6-v2"       # small + fast + accurate
BATCH_SIZE = 64
# ==========================

# 1) Load chunks
texts = []
metadata = []

for fname in os.listdir(INPUT_FOLDER):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(INPUT_FOLDER, fname)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        for ch in chunks:
            text = ch.get("content", "").strip()
            if not text:
                continue
            texts.append(text)
            metadata.append({
                "file": ch.get("file"),
                "page": ch.get("page"),
                "chunk_id": ch.get("chunk_id"),
                "text": text
            })

print(f"📥 Loaded {len(texts)} chunks from {INPUT_FOLDER}")

# 2) Create embeddings
model = SentenceTransformer(MODEL_NAME)
embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
    batch = texts[i:i+BATCH_SIZE]
    vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    embeddings.append(vecs)

embeddings = np.vstack(embeddings) if embeddings else np.zeros((0, 384), dtype=np.float32)
print(f"✅ Embeddings shape: {embeddings.shape}")

# 3) Save to disk
np.save(OUT_EMB_NPY, embeddings)

with open(OUT_META_JSONL, "w", encoding="utf-8") as f:
    for m in metadata:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"✅ Saved embeddings → {OUT_EMB_NPY}")
print(f"✅ Saved metadata   → {OUT_META_JSONL}")
