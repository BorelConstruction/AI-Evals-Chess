from itertools import combinations

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

rows = []
for info in list_eval_logs("logs/compare-v1"):
    log = read_eval_log(info)
    if log.status != "success":
        continue
    for s in log.samples:
        score = next(iter(s.scores.values()))
        rows.append({"model": log.eval.model, "id": s.id, "score": score.value})

df = pd.DataFrame(rows).pivot_table(index="id", columns="model", values="score")

print(df.agg(["mean", "sem"]).T, "\n")

for a, b in combinations(df.columns, 2):
    d = (df[a] - df[b]).dropna()
    print(f"{a} - {b}: {d.mean():+.3f} ± {d.sem():.3f}  (n={len(d)})")

import matplotlib.pyplot as plt

summary = df.agg(["mean", "sem"]).T

ax = summary["mean"].plot.bar(yerr=summary["sem"], capsize=4, rot=15)
ax.set_ylabel("score")
ax.set_title("Chess eval by model")
plt.tight_layout()
plt.savefig("compare.png", dpi=150)

summary.to_html("summary.html")
df.to_csv("per_position.csv")  