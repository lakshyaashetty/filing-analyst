import re
import sys
from functools import lru_cache

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from src.index import EMBEDDER

load_dotenv()
LLM = Anthropic()
MODEL = "claude-haiku-5-5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOPWORDS = {"what", "how", "did", "does", "do", "the", "a", "an", "of", "in", "to", "and", "is",
             "was", "its", "it", "for", "from", "on", "say", "says", "many", "much", "which", "with"}

SYSTEM = """You answer questions about company 10-K annual reports using ONLY the numbered sources provided.
Cite every claim with its source number, like [2]. If the sources don't contain the answer,
say exactly: "The filings provided don't answer this." Never use outside knowledge for facts or figures."""


def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def get_collection(name: str):
    return chromadb.PersistentClient(path="data/chroma").get_collection(name)


def vector_search(question, n, ticker, collection):
    q = EMBEDDER.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    res = get_collection(collection).query(
        query_embeddings=q, n_results=n, where={"ticker": ticker} if ticker else None)
    return [{"id": i, "text": d, **m}
            for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]


@lru_cache(maxsize=4)
def bm25_index(collection):
    """Build the keyword index once per collection (all chunks, from Chroma)."""
    data = get_collection(collection).get(include=["documents", "metadatas"])
    bm25 = BM25Okapi([tokenize(d) for d in data["documents"]])
    return bm25, data["ids"], data["documents"], data["metadatas"]


def keyword_search(question, n, ticker, collection):
    bm25, ids, docs, metas = bm25_index(collection)
    scores = bm25.get_scores(tokenize(question))
    hits = []
    for i in scores.argsort()[::-1]:
        if ticker and metas[i]["ticker"] != ticker:
            continue
        hits.append({"id": ids[i], "text": docs[i], **metas[i]})
        if len(hits) == n:
            break
    return hits


def rrf(*ranked_lists, c=60):
    """Reciprocal rank fusion: items ranked high in several lists win."""
    scores, items = {}, {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            scores[hit["id"]] = scores.get(hit["id"], 0) + 1 / (c + rank + 1)
            items[hit["id"]] = hit
    return [items[i] for i in sorted(scores, key=scores.get, reverse=True)]


def retrieve(question, k=5, ticker=None, collection="filings_300", mode="vector"):
    if mode == "vector":
        return vector_search(question, k, ticker, collection)
    if mode == "keyword":
        return keyword_search(question, k, ticker, collection)
    fused = rrf(vector_search(question, 20, ticker, collection),
                keyword_search(question, 20, ticker, collection))
    return fused[:k]


def answer(question: str, **kw) -> dict:
    hits = retrieve(question, **kw)
    sources = "\n\n".join(
        f"[{n}] ({h['ticker']} 10-K filed {h['filing_date']})\n{h['text']}"
        for n, h in enumerate(hits, 1))
    msg = LLM.messages.create(
        model=MODEL, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    return {"answer": text, "sources": hits, "usage": msg.usage, "stop_reason": msg.stop_reason}


if __name__ == "__main__":
    question = sys.argv[1]
    ticker = sys.argv[2] if len(sys.argv) > 2 else None
    mode = sys.argv[3] if len(sys.argv) > 3 else "vector"
    result = answer(question, ticker=ticker, mode=mode)
    print(result["answer"], "\n")
    if result["stop_reason"] == "max_tokens":
        print("WARNING: answer was cut off by max_tokens\n")
    for n, h in enumerate(result["sources"], 1):
        print(f"[{n}] {h['ticker']} 10-K filed {h['filing_date']} (chunk {h['chunk']})")
    u = result["usage"]
    print(f"\ntokens: {u.input_tokens} in, {u.output_tokens} out | "
          f"cost ≈ ${u.input_tokens * 0.10 / 1e6 + u.output_tokens * 0.50 / 1e6:.5f}")
