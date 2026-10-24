"""Combine both halves of the U_CI probe and report the implied mixture."""
import json
from scipy.stats import beta

def wilson_exact(k, n, a=0.05):
    """Clopper-Pearson two-sided, same family as sec:qual's bound."""
    lo = 0.0 if k == 0 else beta.ppf(a/2, k, n-k+1)
    hi = 1.0 if k == n else beta.ppf(1-a/2, k+1, n-k)
    return lo, hi

mut = json.load(open("/tmp/mutants.json"))
n = len(mut); s = sum(m["survived"] for m in mut)
lo, hi = wilson_exact(s, n)
print("="*66)
print("HALF 1 -- mutation-operator defects (flask, own test suite)")
print("="*66)
print(f"  mutants run      : {n}")
print(f"  survived (missed): {s}")
print(f"  CI miss rate     : {s/n:.1%}  [95% CI {lo:.1%}, {hi:.1%}]")

cov = [m for m in mut if m["covered"]]
unc = [m for m in mut if not m["covered"]]
if cov:
    sc = sum(m["survived"] for m in cov)
    print(f"  on COVERED lines : {sc}/{len(cov)} = {sc/len(cov):.1%} missed"
          f"   <- genuine test-quality gap")
if unc:
    su = sum(m["survived"] for m in unc)
    print(f"  on UNCOVERED     : {su}/{len(unc)} = {su/len(unc):.1%} missed"
          f"   <- unreachable by construction")

print("\n  by operator:")
ops = {}
for m in mut: ops.setdefault(m["op"], []).append(m["survived"])
for op, v in sorted(ops.items(), key=lambda kv: -len(kv[1])):
    print(f"    {op:10s} {sum(v):3d}/{len(v):3d} survived = {sum(v)/len(v):5.1%}")

try:
    cwe = json.load(open("/tmp/cwe_defects.json"))
except FileNotFoundError:
    print("\n(CWE half not yet run)"); raise SystemExit

app = [c for c in cwe if c["applied"]]
ns = len(app); ss = sum(1 for c in app if c["survived"])
slo, shi = wilson_exact(ss, ns)
print("\n" + "="*66)
print("HALF 2 -- CWE-family defects")
print("="*66)
print(f"  defects applied  : {ns}")
print(f"  survived (missed): {ss}")
print(f"  CI miss rate     : {ss/ns:.0%}  [95% CI {slo:.0%}, {shi:.0%}]")
for c in app:
    print(f"    {c['id']} {c['cwe']:9s} {'MISSED' if c['survived'] else 'caught':7s} {c['desc']}")

print("\n" + "="*66)
print("IMPLIED U_CI AS A MIXTURE")
print("="*66)
pm, pc = s/n, ss/ns
print(f"  mutation-operator miss rate p_m = {pm:.3f}")
print(f"  CWE-family        miss rate p_c = {pc:.3f}")
print(f"\n  U_CI(w) = w*p_c + (1-w)*p_m,  w = share of seeded defects from CWE families\n")
print(f"    {'w':>6} {'U_CI':>8}")
for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
    print(f"    {w:6.2f} {w*pc + (1-w)*pm:8.3f}")
target = 0.40
if abs(pc - pm) > 1e-9:
    w_star = (target - pm) / (pc - pm)
    if 0.0 <= w_star <= 1.0:
        print(f"\n  w for U_CI = 0.40 (the power-analysis assumption): {w_star:.2f}")
    else:
        print(f"\n  U_CI = 0.40 NOT reachable by mixing: implied w = {w_star:.2f}, outside [0,1]")
        print(f"  achievable range is [{min(pm,pc):.3f}, {max(pm,pc):.3f}]")
