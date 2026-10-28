"""Grade a sample of answers yourself, then compare with the AI judge."""
import textwrap

import pandas as pd

gold = pd.read_csv("eval/gold.csv", dtype=str).set_index("id")
res = pd.read_csv("eval/results_final.csv", dtype={"id": str})
res = res[res["type"] != "unanswerable"]

flagged = res[res["grade"] != "CORRECT"]
others = res[res["grade"] == "CORRECT"].sample(5, random_state=1)
sample = pd.concat([flagged, others]).sample(frac=1, random_state=2)  # shuffle so you can't guess

labels = {"c": "CORRECT", "p": "PARTIAL", "w": "WRONG"}
rows = []
for n, r in enumerate(sample.itertuples(), 1):
    print("\n" + "=" * 70)
    print(f"{n}/{len(sample)}  QUESTION: {r.question}")
    print(f"\nEXPECTED: {gold.loc[r.id, 'expected_answer']}")
    print("\nANSWER:\n" + textwrap.fill(str(r.answer).replace("\\n", " "), 100)[:1500])
    mine = ""
    while mine not in labels:
        mine = input("\nYour grade [c]orrect / [p]artial / [w]rong: ").strip().lower()
    rows.append({"id": r.id, "judge": r.grade, "me": labels[mine]})

out = pd.DataFrame(rows)
out.to_csv("eval/judge_check.csv", index=False)
agree = (out["judge"] == out["me"]).mean()
print("\n" + "=" * 70)
print(f"You and the judge agreed on {agree:.0%} ({(out.judge == out.me).sum()} of {len(out)})")
print(out[out["judge"] != out["me"]].to_string(index=False) or "No disagreements")
