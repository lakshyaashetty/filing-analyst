import gradio as gr

from src.agent import ask

COMPANIES = ["Any company", "BLK - BlackRock", "IVZ - Invesco", "GOOGL - Alphabet", "IBM - IBM",
             "M - Macy's", "W - Wayfair", "MS - Morgan Stanley", "GS - Goldman Sachs"]

EXAMPLES = [
    ["How many employees did Alphabet have at the end of 2025?", "GOOGL - Alphabet"],
    ["What was IBM's net income in 2024?", "IBM - IBM"],
    ["How did Wayfair's employee count change from 2024 to 2025?", "W - Wayfair"],
    ["What does BlackRock say about the risks of using AI?", "BLK - BlackRock"],
    ["What are Morgan Stanley's business segments?", "MS - Morgan Stanley"],
]

INTRO = """# 10-K Filing Analyst
Ask questions about the latest two annual reports (10-Ks) of 8 public companies. An AI agent searches
the filings, pulls exact figures from SEC financial data, does the math, and cites every claim.

**Evaluated on 28 hand-written questions:** 88% correct overall; **73% on 11 held-out questions**
written after the system was built. Weak spots: facts that only appear inside tables, and long lists.
It's a portfolio project, not financial advice."""


def respond(question, company):
    question = (question or "").strip()[:500]  # cap length to protect the API budget
    if not question:
        return "Please type a question.", ""
    ticker = None if company == "Any company" else company.split(" - ")[0]
    result = ask(question, ticker)

    steps = "\n".join(f"{i}. `{s}`" for i, s in enumerate(result["trace"], 1)) or "No tools used."
    passages = "\n\n".join(
        f"**[S{i}] {h['ticker']} 10-K filed {h['filing_date']}**\n\n> {h['text'][:700]}..."
        for i, h in enumerate(result["sources"], 1)) or "No filing passages retrieved."
    return result["answer"], f"### Tool calls\n{steps}\n\n### Passages the agent read\n{passages}"


with gr.Blocks(title="10-K Filing Analyst") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        question = gr.Textbox(label="Your question", placeholder="e.g. What drives Invesco's revenue?", scale=3)
        company = gr.Dropdown(COMPANIES, value="Any company", label="Company", scale=1)
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    with gr.Accordion("How the agent got this answer", open=False):
        details = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[question, company])
    ask_btn.click(respond, [question, company], [answer, details])
    question.submit(respond, [question, company], [answer, details])

if __name__ == "__main__":
    demo.launch()
