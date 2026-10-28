import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")
RAW = Path("data/raw/filings")


def chunk_words(text: str, size: int = 300, overlap: int = 50):
    """Split text into overlapping chunks of `size` words."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_index(size: int = 300, overlap: int = 50):
    name = f"filings_{size}"
    client = chromadb.PersistentClient(path="data/chroma")
    col = client.get_or_create_collection(name)
    total = 0
    for txt in sorted(RAW.glob("*.txt")):
        meta = json.loads(txt.with_suffix(".json").read_text())
        chunks = chunk_words(txt.read_text(), size, overlap)
        emb = EMBEDDER.encode(chunks, batch_size=64, normalize_embeddings=True)
        col.upsert(
            ids=[f"{txt.stem}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=emb.tolist(),
            metadatas=[{**meta, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        print(f"{txt.stem}: {len(chunks):,} chunks")
    print(f"{name}: {total:,} chunks indexed")


if __name__ == "__main__":
    build_index(size=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
