import sys

import pandas as pd

from src.agent import ask
from src.rag import LLM, MODEL, answer

JUDGE = """You grade answers to questions about company filings.
Compare the ANSWER to the EXPECTED answer and reply with exactly one word:
CORRECT (all key facts match), PARTIAL (some key facts right, some missing or wrong), or WRONG."""


def norm(s) -> str:
    return " ".join(str(s).split()).lower().replace("\u2019", "'")


def judge(question, expected, got) -> str:
    msg = LLM.messages.create(
        model=MODEL, max_tokens=1000, system=JUDGE,
        messages=[{"role": "user", "content": f"QUESTION: {question}\nEXPECTED: {expected}\nANSWER: {got}"}])
    text = "".join(b.text for b in msg.content if b.type == "text").strip().upper()
    word = text.split()[0].strip(".*:") if text else ""
    return word if word in {"CORRECT", "PARTIAL", "WRONG"} else "UNKNOWN"


def run(name="baseline", collection="filings_300", k=5, mode="vector"):
    gold = pd.read_csv("eval/gold.csv", dtype=str).fillna("")
    rows = []
    for r in gold.itertuples():
        res = ask(r.question, ticker=r.ticker or None) if mode == "agent" else answer(r.question, ticker=r.ticker or None, k=k, collection=collection, mode=mode)
        retrieved = " ".join(norm(h["text"]) for h in res["sources"])
        if r.type == "unanswerable":
            hit = None
            grade = "CORRECT" if "don't answer this" in norm(res["answer"]) else "WRONG"
        else:
            hit = all(norm(p) in retrieved for p in r.evidence.split("|"))
            grade = judge(r.question, r.expected_answer, res["answer"])
        rows.append({"id": r.id, "type": r.type, "question": r.question, "retrieval_hit": hit,
                     "grade": grade, "answer": res["answer"]})
        print(f"[{r.id}] {r.type:12s} hit={hit!s:5s} grade={grade}")

    out = pd.DataFrame(rows)
    out.to_csv(f"eval/results_{name}.csv", index=False)
    answerable = out[out["type"] != "unanswerable"]
    print(f"\n=== {name}: {collection}, k={k}, {mode} ===")
    print(f"Retrieval hit rate: {answerable['retrieval_hit'].mean():.0%}")
    print(answerable["grade"].value_counts(normalize=True).mul(100).round(0).to_string())
    refusals = out[out["type"] == "unanswerable"]
    if len(refusals):
        print(f"Correct refusals: {(refusals['grade'] == 'CORRECT').sum()} of {len(refusals)}")
    print("\nBy type:")
    print(answerable.groupby("type")["grade"].apply(lambda g: (g == "CORRECT").mean()).round(2).to_string())


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    collection = sys.argv[2] if len(sys.argv) > 2 else "filings_300"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    mode = sys.argv[4] if len(sys.argv) > 4 else "vector"
    run(name, collection, k, mode)
