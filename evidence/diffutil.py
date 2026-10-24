"""Changed-line map for a candidate: which (file, line) pairs the candidate added
or modified relative to the pinned state. Producers restrict findings to these
lines so a finding is attributable to the change, not to pre-existing code."""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_NOISE = re.compile(r"(^|/)(\.venv|venv|__pycache__|\.pytest_cache|\.hypothesis|\.coverage[^/]*|\.mypy_cache|"
                    r"node_modules|[^/]*\.egg-info)(/|$)")


def untracked_files(repo: Path) -> List[str]:
    """Files `git apply` created but did not stage: added modules and added tests
    are untracked, so a plain `git diff` would miss them."""
    r = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip() and not _NOISE.search(l)]


def changed_lines(repo: Path, base_sha: str) -> Dict[str, Set[int]]:
    """{rel_path: {added/modified line numbers in the working tree}} vs base_sha.
    Every line of an untracked (added) file counts as changed."""
    r = subprocess.run(["git", "diff", "--unified=0", "--no-color", base_sha, "--"], cwd=repo,
                       capture_output=True, text=True)
    out: Dict[str, Set[int]] = {}
    for f in untracked_files(repo):
        p = repo / f
        if p.is_file():
            try:
                n = sum(1 for _ in p.open("rb"))
            except OSError:
                continue
            out[f] = set(range(1, n + 1))
    cur = None
    for line in r.stdout.splitlines():
        if line.startswith("+++ "):
            p = line[4:]
            cur = p[2:] if p.startswith("b/") else (None if p == "/dev/null" else p)
            if cur is not None:
                out.setdefault(cur, set())
        elif line.startswith("@@") and cur is not None:
            m = _HUNK.match(line)
            if m:
                start = int(m.group(1)); n = int(m.group(2) or 1)
                out[cur].update(range(start, start + n))
    return out


def changed_files(repo: Path, base_sha: str) -> List[str]:
    """Tracked files that differ from base_sha plus untracked (added) files."""
    r = subprocess.run(["git", "diff", "--name-only", base_sha, "--"], cwd=repo, capture_output=True, text=True)
    tracked = [l for l in r.stdout.splitlines() if l.strip()]
    seen = set(tracked)
    return tracked + [f for f in untracked_files(repo) if f not in seen]


def in_change(cl: Dict[str, Set[int]], path: str, line: int) -> bool:
    p = path.replace("\\", "/")
    for k, v in cl.items():
        if p == k or p.endswith("/" + k) or k.endswith("/" + p):
            return line in v
    return False
