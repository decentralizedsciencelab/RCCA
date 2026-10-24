"""U_CI probe: what fraction of mutation-operator defects does a repo's own CI miss?

Survival rate here == the CI miss rate for mutation-seeded defects, which is
one half of RCCA's U_CI (the other half being CWE-family defects).

Controls:
  - per-file "null mutant": ast.unparse with no change must keep tests green,
    so any later failure is attributable to the mutation and not to unparsing.
  - coverage map: separates "survived because the line is untested" from
    "survived despite being executed" -- only the latter is a test-quality gap.
"""
import ast, json, random, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(sys.argv[1]); N = int(sys.argv[2]); SEED = int(sys.argv[3])
PY_BIN = REPO / ".venv/bin/python"
DESELECT = ["--deselect", "tests/test_reqctx.py::test_bad_environ_raises_bad_request"]
PYTEST = [str(PY_BIN), "-m", "pytest", "tests", "-x", "-q", "--no-header",
          "-p", "no:cacheprovider", *DESELECT]

rng = random.Random(SEED)

# ---- mutation operators -------------------------------------------------
AOR = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.Div: ast.Mult}
ROR = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
       ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
COR = {ast.And: ast.Or, ast.Or: ast.And}

class Collect(ast.NodeVisitor):
    """Record every mutatable site as (op_class, node_id, lineno)."""
    def __init__(self): self.sites = []
    def visit_BinOp(self, n):
        if type(n.op) in AOR: self.sites.append(("AOR", id(n), n.lineno))
        self.generic_visit(n)
    def visit_Compare(self, n):
        if len(n.ops) == 1 and type(n.ops[0]) in ROR:
            self.sites.append(("ROR", id(n), n.lineno))
        self.generic_visit(n)
    def visit_BoolOp(self, n):
        if type(n.op) in COR: self.sites.append(("COR", id(n), n.lineno))
        self.generic_visit(n)
    def visit_Constant(self, n):
        if isinstance(n.value, bool): self.sites.append(("CRP_bool", id(n), n.lineno))
        elif isinstance(n.value, int) and not isinstance(n.value, bool):
            self.sites.append(("CRP_int", id(n), n.lineno))
        self.generic_visit(n)
    def visit_If(self, n):
        self.sites.append(("NEG_if", id(n), n.lineno)); self.generic_visit(n)

class Apply(ast.NodeTransformer):
    """Mutate exactly the one site whose id matches."""
    def __init__(self, target, op): self.target, self.op, self.done = target, op, False
    def visit_BinOp(self, n):
        self.generic_visit(n)
        if id(n) == self.target and self.op == "AOR":
            n.op = AOR[type(n.op)](); self.done = True
        return n
    def visit_Compare(self, n):
        self.generic_visit(n)
        if id(n) == self.target and self.op == "ROR":
            n.ops = [ROR[type(n.ops[0])]()]; self.done = True
        return n
    def visit_BoolOp(self, n):
        self.generic_visit(n)
        if id(n) == self.target and self.op == "COR":
            n.op = COR[type(n.op)](); self.done = True
        return n
    def visit_Constant(self, n):
        if id(n) == self.target and self.op.startswith("CRP"):
            if isinstance(n.value, bool): n.value = not n.value
            else: n.value = n.value + 1
            self.done = True
        return n
    def visit_If(self, n):
        self.generic_visit(n)
        if id(n) == self.target and self.op == "NEG_if":
            n.test = ast.UnaryOp(op=ast.Not(), operand=n.test); self.done = True
        return n

def run_tests(timeout=120):
    try:
        r = subprocess.run(PYTEST, cwd=REPO, capture_output=True, timeout=timeout)
        return r.returncode == 0          # True == suite green == mutant SURVIVED
    except subprocess.TimeoutExpired:
        return False                       # hang == CI catches it (job times out)

# ---- coverage map -------------------------------------------------------
cov = json.load(open("/tmp/cov.json"))["files"]
covmap = {}
for k, v in cov.items():
    covmap[Path(k).name] = set(v.get("executed_lines", []))

targets = sorted((REPO / "src/flask").rglob("*.py"))
targets = [t for t in targets if t.name != "__init__.py" and t.stat().st_size > 800]

# ---- null-mutant control ------------------------------------------------
print("== null-mutant control ==", flush=True)
usable = []
for t in targets:
    orig = t.read_text()
    try:
        rt = ast.unparse(ast.parse(orig))
    except Exception as e:
        print(f"  SKIP {t.name}: unparse error {e}", flush=True); continue
    t.write_text(rt)
    ok = run_tests()
    t.write_text(orig)
    if ok: usable.append(t)
    else:  print(f"  SKIP {t.name}: unparse alone breaks the suite", flush=True)
print(f"  {len(usable)}/{len(targets)} files pass the control", flush=True)

# ---- enumerate sites ----------------------------------------------------
pool = []
for t in usable:
    tree = ast.parse(t.read_text())
    c = Collect(); c.visit(tree)
    for op, nid, line in c.sites:
        pool.append((t, op, line))
print(f"== {len(pool)} mutatable sites across {len(usable)} files ==", flush=True)

rng.shuffle(pool)
results, seen, t0 = [], set(), time.time()
for t, op, line in pool:
    if len(results) >= N: break
    key = (t.name, op, line)
    if key in seen: continue
    seen.add(key)
    orig = t.read_text()
    tree = ast.parse(orig)
    c = Collect(); c.visit(tree)
    match = [nid for o, nid, l in c.sites if o == op and l == line]
    if not match: continue
    ap = Apply(match[0], op); new = ap.visit(tree)
    if not ap.done: continue
    try: src = ast.unparse(ast.fix_missing_locations(new))
    except Exception: continue
    if src == ast.unparse(ast.parse(orig)): continue      # no-op mutant
    t.write_text(src)
    survived = run_tests()
    t.write_text(orig)
    results.append({"file": t.name, "op": op, "line": line,
                    "covered": line in covmap.get(t.name, set()),
                    "survived": survived})
    if len(results) % 25 == 0:
        s = sum(r["survived"] for r in results)
        print(f"  {len(results)}/{N}  survived={s} ({s/len(results):.1%})  "
              f"{time.time()-t0:.0f}s", flush=True)

json.dump(results, open("/tmp/mutants.json", "w"), indent=1)
print(f"\n== DONE {len(results)} mutants in {time.time()-t0:.0f}s ==", flush=True)
