"""Print a sample of surviving mutants with source context, for manual
classification as REAL DEFECT vs EQUIVALENT/TRIVIAL.

Needed because sec:pool admits a pair only if the injecting oracle catches the
unsafe variant. Equivalent mutants survive CI but would never be admitted, so
raw survival rate is an UPPER BOUND on the mutation half of U_CI.
"""
import ast, json, random, sys
from pathlib import Path

SRC = Path(sys.argv[1]) / "src/flask"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
mut = json.load(open("/tmp/mutants.json"))
surv = [m for m in mut if m["survived"]]
rng = random.Random(7)
rng.shuffle(surv)

print(f"{len(surv)} survivors total; showing {min(K,len(surv))}\n")
for i, m in enumerate(surv[:K], 1):
    f = next(SRC.rglob(m["file"]), None)
    if f is None: continue
    lines = f.read_text().splitlines()
    ln = m["line"]
    ctx = "\n".join(f"    {n+1:5d}| {lines[n]}"
                    for n in range(max(0, ln-3), min(len(lines), ln+2)))
    print(f"--- [{i}] {m['file']}:{ln}  op={m['op']}  covered={m['covered']}")
    print(ctx); print()
