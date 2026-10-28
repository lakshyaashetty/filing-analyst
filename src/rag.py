import sys

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

from src.index import EMBEDDER

load_dotenv()
LLM = Anthropic()
MODEL = "claude-haiku-5-5"

# bge models search better when the QUESTION (not the passages) gets this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM = """You answer questions about company 10-K annual reports using ONLY the numbered sources provided.
Cite every claim with its source number, like [2]. If the sources don't contain the answer,
say exactly: "The filings provided don't answer this." Never use outside knowledge for facts or figures."""


def retrieve(question: str, k: int = 5, ticker: str | None = None, collection: str = "filings_300"):
    col = chromadb.PersistentClient(path="data/chroma").get_collection(collection)
    q = EMBEDDER.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=k, where={"ticker": ticker} if ticker else None)
    return [{"id": i, "text": d, **m}
            for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]


def answer(question: str, **kw) -> dict:
    hits = retrieve(question, **kw)
    sources = "\n\n".join(
        f"[{n}] ({h['ticker']} 10-K filed {h['filing_date']})\n{h['text']}"
        for n, h in enumerate(hits, 1)
    )
    msg = LLM.messages.create(
        model=MODEL, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}],
    )
    return {"answer": "".join(b.text for b in msg.content if b.type == "text"), "sources": hits, "usage": msg.usage, "stop_reason": msg.stop_reason}


if __name__ == "__main__":
    question = sys.argv[1]
    ticker = sys.argv[2] if len(sys.argv) > 2 else None
    result = answer(question, ticker=ticker)
    print(result["answer"], "\n")
    if result["stop_reason"] == "max_tokens":
        print("WARNING: answer was cut off by max_tokens\n")
    for n, h in enumerate(result["sources"], 1):
        print(f"[{n}] {h['ticker']} 10-K filed {h['filing_date']} (chunk {h['chunk']})")
    u = result["usage"]
    cost = u.input_tokens * 0.10 / 1e6 + u.output_tokens * 0.50 / 1e6
    print(f"\ntokens: {u.input_tokens} in, {u.output_tokens} out | cost ≈ ${cost:.5f}")
