"""F2 security: static analysis with semgrep and bandit, restricted to changed lines.

Outcome rules
  pass               no finding on a changed line
  fail_unvalidated   one tool reports a finding on a changed line
  fail_validated     semgrep and bandit both report a finding at the same
                     changed line with the same CWE. Criteria: (i) the rule
                     set is a declared invariant, (ii) a single finding is the
                     minimal witness, (iii) static analysis is deterministic
                     and is re-run at k sub-seeds to confirm, (iv) the second
                     tool is the confirmer
  inconclusive       a tool crashed, timed out, or could not parse the tree

Rules are local files (`--config <path>`), never the registry: no network at
run time (spec §9). The rule files are the *detector* side of the circularity
firewall and must stay disjoint from the seeding catalog (spec §11).

`semgrep` is the executable F2 runs: with the repository config's
env.host_tools, the venv `corpus.envs.ensure_tool` builds at its pin (semgrep's
own requirements would replace the subject's click), otherwise semgrep from
PATH. Its version is part of every record's provenance, because the version
changes which findings exist and so whether the two tools agree. It runs with
its own directory first on PATH (`semgrep_env`): semgrep's launcher appends that
directory only at the end, and its semgrep-core then runs `pysemgrep` through
PATH, so a pysemgrep earlier on PATH would otherwise do the scanning.
"""
from __future__ import annotations
import functools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..core import Action, Context, Meter, Outcome, Registry, Validation, sub_seed
from ..diffutil import changed_lines, in_change

_CWE = re.compile(r"CWE-(\d+)")


def _bandit(m: Meter, ctx: Context, python: str, target: str, timeout: int) -> Optional[List[dict]]:
    r = m.tool([python, "-m", "bandit", "-r", target, "-f", "json", "-q", "-ll"], ctx.repo, timeout)
    if r.returncode not in (0, 1):
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    out = []
    for f in data.get("results", []):
        cwe = f.get("issue_cwe", {}).get("id")
        out.append({"tool": "bandit", "file": f["filename"], "line": int(f["line_number"]),
                    "cwe": f"CWE-{cwe}" if cwe else "", "rule": f.get("test_id", ""), "severity": f.get("issue_severity", "")})
    return out


def semgrep_env(semgrep: str) -> Dict[str, str]:
    """The environment `semgrep` runs in: the current one with the executable's own directory first on
    PATH. semgrep's launcher appends its directory only at the END of PATH and execs semgrep-core,
    which runs `pysemgrep` found through PATH; without this a pysemgrep earlier on PATH (the
    orchestrator's, from requirements-dev.txt) would answer for the pinned venv, version and scan."""
    exe = semgrep if os.path.dirname(semgrep) else shutil.which(semgrep)
    env = dict(os.environ)
    if exe:
        env["PATH"] = os.pathsep.join(p for p in (os.path.dirname(os.path.abspath(exe)), env.get("PATH", "")) if p)
    return env


@functools.lru_cache(maxsize=None)
def semgrep_version_of(semgrep: str) -> Optional[str]:
    """`semgrep --version` of the executable F2 runs, in the environment F2 runs it in, or None when
    it does not run."""
    try:
        r = subprocess.run([semgrep, "--version"], capture_output=True, text=True, timeout=120,
                           env=dict(semgrep_env(semgrep), SEMGREP_ENABLE_VERSION_CHECK="0"))
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = re.search(r"\d+(?:\.\d+)+", r.stdout)
    return found.group(0) if r.returncode == 0 and found else None


def _semgrep(m: Meter, ctx: Context, semgrep: str, rules: Sequence[str], target: str,
             timeout: int) -> Optional[List[dict]]:
    cmd = [semgrep, "scan", "--json", "--quiet", "--metrics=off", "--disable-version-check", "--no-git-ignore"]
    for rp in rules:
        cmd += ["--config", rp]
    cmd.append(target)
    r = m.tool(cmd, ctx.repo, timeout, env=semgrep_env(semgrep))
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    if data.get("errors") and not data.get("results"):
        fatal = [e for e in data["errors"] if e.get("level") == "error"]
        if fatal:
            return None
    out = []
    for f in data.get("results", []):
        meta = f.get("extra", {}).get("metadata", {})
        cwes = meta.get("cwe", [])
        cwes = [cwes] if isinstance(cwes, str) else cwes
        cwe = ""
        for c in cwes:
            mm = _CWE.search(str(c))
            if mm:
                cwe = f"CWE-{mm.group(1)}"; break
        out.append({"tool": "semgrep", "file": f["path"], "line": int(f["start"]["line"]), "cwe": cwe,
                    "rule": f.get("check_id", ""), "severity": f.get("extra", {}).get("severity", "")})
    return out


def make_static_security(python: str, semgrep_rules: Sequence[str], target: str = "src", k: int = 2,
                         timeout: int = 900, use_semgrep: bool = True, semgrep: str = "semgrep",
                         semgrep_version: Optional[str] = None) -> Action:
    """`semgrep` is the executable to run. Its version, as evidence.run checked it or as found here when
    the action is built (never inside a record's meter), goes into every record's provenance."""
    version = (semgrep_version or semgrep_version_of(semgrep) or "missing") if use_semgrep else None

    def run(ctx: Context, m: Meter) -> Outcome:
        cl = changed_lines(ctx.repo, ctx.repo_state_sha)
        tgt = target if target else ctx.package_root
        prov = {"tools": "semgrep+bandit" if use_semgrep else "bandit", "rules": ";".join(semgrep_rules), "target": tgt}
        if version:
            prov["semgrep_version"] = version
        blind = ["pattern-based: misses logic flaws without a syntactic signature", "changed lines only"]
        b = _bandit(m, ctx, python, tgt, timeout)
        if b is None:
            return Outcome("inconclusive", provenance=prov, blind_spots=blind, status="error")
        s = _semgrep(m, ctx, semgrep, semgrep_rules, tgt, timeout) if use_semgrep else []
        if s is None:
            return Outcome("inconclusive", provenance=prov, blind_spots=blind, status="error")
        hits = [f for f in b + s if in_change(cl, f["file"], f["line"])]
        prov["findings_total"] = str(len(b) + len(s)); prov["findings_on_change"] = str(len(hits))
        if not hits:
            return Outcome("pass", provenance=prov, blind_spots=blind)
        # agreement: same file (by suffix), same line, same CWE, different tools
        def key(f): return (Path(f["file"]).name, f["line"], f["cwe"])
        by = {}
        for f in hits:
            by.setdefault(key(f), set()).add(f["tool"])
        agreed = [kk for kk, tools in by.items() if len(tools) >= 2 and kk[2]]
        prov["agreed"] = str(len(agreed))
        if not agreed:
            return Outcome("fail_unvalidated", Validation(contradicts_reference=True, minimized=True, repeats_across_seeds=True),
                           prov, blind_spots=blind)
        # (iii) repeat at k sub-seeds: both tools again, same agreed keys
        seeds = [sub_seed(ctx.seed, i) for i in range(k)]
        repeats = True
        for _ in seeds:
            b2 = _bandit(m, ctx, python, tgt, timeout) or []
            s2 = _semgrep(m, ctx, semgrep, semgrep_rules, tgt, timeout) or []
            by2 = {}
            for f in [x for x in b2 + s2 if in_change(cl, x["file"], x["line"])]:
                by2.setdefault(key(f), set()).add(f["tool"])
            if any(len(by2.get(kk, set())) < 2 for kk in agreed):
                repeats = False; break
        v = Validation(True, True, repeats, repeats, "bandit+semgrep-agreement" if repeats else "",
                       ";".join(f"{a}:{l}:{c}" for a, l, c in agreed), seeds)
        return Outcome("fail_validated" if v.all_four else "fail_unvalidated", v, prov, blind_spots=blind)

    return Action("f2.static_security", "F2", "author_side", "semgrep+bandit" if use_semgrep else "bandit", run, priority=20)


def register(reg: Registry, python: str, semgrep_rules: Sequence[str], target: str, **kw) -> Action:
    return reg.register(make_static_security(python, semgrep_rules, target, **kw))
