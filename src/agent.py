import ast
import json
import operator as op
import os
import re
import sys

import httpx
from dotenv import load_dotenv
from edgar import Company, set_identity

from src.rag import LLM, MODEL, retrieve

load_dotenv()
IDENTITY = os.environ["EDGAR_IDENTITY"]
set_identity(IDENTITY)

SYSTEM = """You are a financial filings analyst. Answer using your tools, never from memory.
- search_filings: for anything described in 10-K text. Call it more than once when the question needs
  several facts (for example, the same figure in two different years).
- get_financial_fact: for exact reported financial figures (revenue, net income, assets, ...).
- calculate: for any arithmetic.
Cite filing passages by their tag, like [S2], and figures from get_financial_fact as [SEC data].
If your tools don't provide the answer, say exactly: "The filings provided don't answer this."
Answer concisely."""

TOOLS = [
    {"name": "search_filings",
     "description": "Search 10-K filing text. Returns tagged passages with ticker and filing date.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}, "ticker": {"type": "string"}},
                      "required": ["query"]}},
    {"name": "get_financial_fact",
     "description": ("Exact annual values from SEC XBRL financial data. `concept` is a us-gaap tag, e.g. "
                     "Revenues, RevenueFromContractWithCustomerExcludingAssessedTax, NetIncomeLoss, "
                     "OperatingIncomeLoss, Assets, CashAndCashEquivalentsAtCarryingValue, "
                     "ResearchAndDevelopmentExpense. If one tag returns an error, try another."),
     "input_schema": {"type": "object",
                      "properties": {"ticker": {"type": "string"}, "concept": {"type": "string"}},
                      "required": ["ticker", "concept"]}},
    {"name": "calculate",
     "description": "Evaluate arithmetic with numbers and + - * /, e.g. (190820-183323)/183323",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
]


def get_financial_fact(ticker: str, concept: str):
    cik = str(Company(ticker).cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
    r = httpx.get(url, headers={"User-Agent": IDENTITY}, timeout=30)
    if r.status_code != 200:
        return {"error": f"No us-gaap '{concept}' data for {ticker}. Try another concept name."}
    values = {}
    for unit, facts in r.json()["units"].items():
        for f in facts:
            # "frame" marks SEC's de-duplicated value for a calendar year (or year-end for balances)
            if re.fullmatch(r"CY\d{4}(Q4I)?", f.get("frame", "")):
                values[f["frame"]] = {"period_end": f["end"], "value": f["val"], "unit": unit}
    rows = [{"period": k, **v} for k, v in sorted(values.items())][-5:]
    return rows or {"error": "No annual values found for that concept."}


OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}


def calculate(expression: str) -> float:
    """Safe arithmetic: numbers and + - * / only (never Python's eval)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("Only numbers and + - * / are allowed")
    return ev(ast.parse(expression, mode="eval").body)


def run_tool(name, args, state) -> str:
    try:
        if name == "search_filings":
            lines = []
            for h in retrieve(args["query"], k=5, ticker=args.get("ticker")):
                state["hits"].append(h)
                lines.append(f"[S{len(state['hits'])}] ({h['ticker']} 10-K filed {h['filing_date']})\n{h['text']}")
            return "\n\n".join(lines)
        if name == "get_financial_fact":
            return json.dumps(get_financial_fact(args["ticker"], args["concept"]))
        if name == "calculate":
            return str(calculate(args["expression"]))
        return f"Unknown tool {name}"
    except Exception as e:
        return f"Tool error: {e}"


def ask(question: str, ticker: str | None = None, max_steps: int = 6) -> dict:
    prompt = question + (f"\n(Company ticker: {ticker})" if ticker else "")
    messages = [{"role": "user", "content": prompt}]
    state = {"hits": [], "trace": []}
    for _ in range(max_steps):
        resp = LLM.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM,
                                   tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "sources": state["hits"], "trace": state["trace"]}
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                state["trace"].append(f"{block.name}({json.dumps(block.input)})")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": run_tool(block.name, block.input, state)})
        messages.append({"role": "user", "content": results})
    return {"answer": "Stopped: too many tool calls.", "sources": state["hits"], "trace": state["trace"]}


if __name__ == "__main__":
    result = ask(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(result["answer"], "\n\nTool calls:")
    for step in result["trace"]:
        print("  ", step)
