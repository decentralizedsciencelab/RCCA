"""Challenger, spec-guided review (`ch.spec_guided`, family F3, an LLM reading the change).

Policy 3's LLM-review stage (rcca.decision.order_fixed_funnel puts a challenger whose producer starts
with "llm" there). A language model reads the candidate's diff against the pinned state together with
what the pinned state stated about itself -- the signatures and docstrings of every function and class
the change touches -- and names the lines where the change breaks that statement or introduces a defect
(docs/design/2026-10-23-redesign.md §4.4). It never sees the task text, the seed record, the oracle,
labels, or another producer's record: only the checkout and its pinned commit, like every producer.

The model is served behind an OpenAI-compatible chat endpoint on the loopback interface: Ollama on the
operator's machine, or vLLM on the cluster through an SSH tunnel (cluster/serve_qwen.sbatch). A
non-loopback endpoint is refused, and the request ignores proxy variables, so the producer cannot reach
the network beyond the port the operator opened.

Outcome rules
  pass               k replies parsed, and no changed hunk is named by at least `min_agree` of them
  fail_unvalidated   some changed hunk is named by at least `min_agree` of the k replies (sub-seeded
                     samples). (i) the pinned state's stated specification is the reference, (iii) the
                     finding repeats across sub-seeds; there is no executable witness and no second
                     tool, so a reading is never a validated violation. That is enough for the decision:
                     an unvalidated failure moves rho and turns an accept into an escalate
  inconclusive       the endpoint is unreachable or serves another model, fewer than `min_agree` replies
                     parse, or the change does not fit the declared context window

Only hunks under the context's package_root count: the candidate's tests are shown as the change's stated
intent, but a reading of a test is not a finding about the library. A finding is attributed to the change
by hunk: its line falls inside a hunk's span in the changed file
(within `line_slack`), or its quote is one of the hunk's added or removed lines. A pure deletion has no
line of its own in the changed file, which is why the span and the quote both count.

Circularity firewall: the prompt (`configs/review_prompt.md`) is a generic review instruction. It names
no CWE and no catalog anchor (tests/test_spec_guided.py checks both), and it carries no checklist shaped
like the seeder's operators. Any class-targeted wording belongs to the construction half of a split
catalog (redesign §4.3), never to the half detection is reported on.

Cost: tokens are the server's reported usage summed over the k calls; tool calls count the HTTP
requests. The model runs outside this process tree, so cpu_s, and with it ci_equiv, stays near zero;
the token ceiling is the one that bounds this producer.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..core import Action, Context, Meter, Outcome, Registry, Validation, sub_seed
from ..diffutil import untracked_files

LOOPBACK = ("127.0.0.1", "localhost", "::1")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_CHARS_PER_TOKEN = 3.0          # conservative for code: a packet estimated to fit does fit


class ReviewConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewConfig:
    endpoint: str                    # http://127.0.0.1:<port>/v1
    model: str                       # the served model id the records name
    prompt: str                      # the frozen review instruction (file contents)
    prompt_sha: str
    context_tokens: int              # the server's context window; a larger packet is inconclusive
    k: int = 3
    min_agree: int = 2
    temperature: float = 0.7
    max_tokens: int = 1024
    context_lines: int = 6
    line_slack: int = 2
    timeout: int = 600
    json_mode: bool = True
    api_key_env: Optional[str] = None


def review_config(c: Dict[str, Any], base_dir: Path = Path(".")) -> ReviewConfig:
    """The evidence config's `spec_guided` block. The endpoint must be on the loopback interface."""
    url = urllib.parse.urlparse(c["endpoint"])
    if url.scheme != "http" or (url.hostname or "") not in LOOPBACK:
        raise ReviewConfigError(f"spec_guided.endpoint {c['endpoint']!r} is not an http endpoint on the loopback "
                                f"interface; reach a remote server through an SSH tunnel")
    p = Path(c.get("prompt", "configs/review_prompt.md"))
    p = p if p.is_absolute() else base_dir / p
    text = p.read_text()
    k = int(c.get("k", 3))
    min_agree = int(c.get("min_agree", math.ceil((k + 1) / 2)))
    if not 1 <= min_agree <= k:
        raise ReviewConfigError(f"spec_guided.min_agree {min_agree} is not in 1..k={k}")
    return ReviewConfig(c["endpoint"].rstrip("/"), c["model"], text, hashlib.sha256(text.encode()).hexdigest(),
                        int(c["context_tokens"]), k, min_agree, float(c.get("temperature", 0.7)),
                        int(c.get("max_tokens", 1024)), int(c.get("context_lines", 6)),
                        int(c.get("line_slack", 2)), int(c.get("timeout", 600)), bool(c.get("json_mode", True)),
                        c.get("api_key_env"))


def check_independent(model: str, authors: Any) -> None:
    """A change is never reviewed by the model that authored it (the rule corpus.judge applies to judges)."""
    names = {str(a.get("model", "")).lower() for a in (authors or {}).values()} if isinstance(authors, dict) else set()
    if model.lower() in names:
        raise ReviewConfigError(f"spec_guided.model {model} is an authoring model of the record")


# ---------------------------------------------------------------- the packet

@dataclass
class Hunk:
    file: str
    lo: int                          # span in the changed file's numbering, inclusive
    hi: int
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)


def _git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout


def _numbered_diff(diff: str) -> List[str]:
    """The diff of one file with each line prefixed by its number in the changed file. Context lines
    keep their number; removed lines get none."""
    out: List[str] = []
    new, inside = 0, False
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            new, inside = int(m.group(3)), True
            out.append(line)
            continue
        if not inside or line.startswith("\\"):           # file headers precede the first hunk
            continue
        tag, body = line[:1], line[1:]
        if tag == "+":
            out.append(f"{new:>6} + {body}"); new += 1
        elif tag == "-":
            out.append(f"{'':>6} - {body}")
        else:
            out.append(f"{new:>6}   {body}"); new += 1
    return out


def _hunks(path: str, diff0: str) -> List[Hunk]:
    """The edits of a zero-context diff, each with its span in the changed file. A pure deletion has no
    line of its own there; its span is the two lines around the place it was."""
    out: List[Hunk] = []
    h: Optional[Hunk] = None
    for line in diff0.splitlines():
        m = _HUNK.match(line)
        if m:
            c, d = int(m.group(3)), int(m.group(4) or 1)
            h = Hunk(path, c, c + d - 1) if d else Hunk(path, c, c + 1)
            out.append(h)
        elif h is not None and line.startswith("+"):        # file headers precede the first hunk
            h.added.append(line[1:])
        elif h is not None and line.startswith("-"):
            h.removed.append(line[1:])
    return out


def _edit_spans(diff0: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Old-side and new-side line spans of each hunk of a zero-context diff. An empty side marks its
    insertion point, so a pure deletion still has a new-side position and a pure addition an old one."""
    old, new = [], []
    for line in diff0.splitlines():
        m = _HUNK.match(line)
        if m:
            a, b = int(m.group(1)), int(m.group(2) or 1)
            c, d = int(m.group(3)), int(m.group(4) or 1)
            old.append((a, a + b - 1) if b else (a, a + 1))
            new.append((c, c + d - 1) if d else (c, c + 1))
    return old, new


def _defs(src: str) -> List[Tuple[str, int, int, ast.AST]]:
    """(qualified name, first line, last line, node) of every function and class in `src`."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []

    def walk(node: ast.AST, prefix: str) -> None:
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}{ch.name}"
                first = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append((q, first, ch.end_lineno or ch.lineno, ch))
                walk(ch, q + ".")
            else:
                walk(ch, prefix)
    walk(tree, "")
    return out


def _signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{kw} {node.name}({ast.unparse(node.args)}){ret}"


def _touched(defs, spans: Sequence[Tuple[int, int]]) -> Dict[str, ast.AST]:
    """The innermost-first defs whose lines overlap a span, by qualified name."""
    hit = {}
    for q, lo, hi, node in defs:
        if any(lo <= b and a <= hi for a, b in spans):
            hit[q] = node
    return hit


def _spec(path: str, before: Optional[str], after: Optional[str], old_spans, new_spans) -> List[str]:
    """What the pinned state stated about every def the change touches: its signature and docstring,
    and the changed signature and docstring too when the change edits them. A def the change adds is
    stated by itself."""
    b_defs = _defs(before) if before is not None else []
    a_defs = _defs(after) if after is not None else []
    b_all = {q: n for q, _, _, n in b_defs}
    a_all = {q: n for q, _, _, n in a_defs}
    lines: List[str] = []
    for q in sorted(set(_touched(b_defs, old_spans)) | set(_touched(a_defs, new_spans))):
        nb, na = b_all.get(q), a_all.get(q)
        lines.append(f"{path} :: {q}" + ("  (added by the change)" if nb is None else
                                         "  (removed by the change)" if na is None else ""))
        lines.append(f"    {_signature(nb or na)}")
        if nb is not None and na is not None and _signature(na) != _signature(nb):
            lines.append(f"    after the change: {_signature(na)}")
        db = ast.get_docstring(nb) if nb is not None else None
        da = ast.get_docstring(na) if na is not None else None
        if db:
            lines.extend("    | " + l for l in db.splitlines())
        if da and da != db:
            lines.append("    docstring after the change:" if nb is not None else "    docstring:")
            lines.extend("    | " + l for l in da.splitlines())
        lines.append("")
    return lines


@dataclass
class Packet:
    text: str
    hunks: List[Hunk]
    files: List[str]


def build_packet(repo: Path, base: str, context_lines: int = 6) -> Packet:
    """SPECIFICATION and CHANGE for the candidate in `repo` against `base` (redesign §4.4)."""
    tracked = [l for l in _git(repo, "diff", "--name-only", base, "--").splitlines() if l.strip()]
    added = [f for f in untracked_files(repo) if f not in tracked]
    spec: List[str] = []
    change: List[str] = []
    hunks: List[Hunk] = []
    for f in tracked + added:
        p = repo / f
        after = p.read_text(errors="replace") if p.is_file() else None
        if after is not None and "\x00" in after:            # a binary file: named, not shown
            change.append(f"--- {f}  (binary, not shown)\n")
            continue
        if f in added:
            before = None
            n = len(after.splitlines()) if after is not None else 0
            diff = f"@@ -0,0 +1,{n} @@\n" + "".join("+" + l + "\n" for l in (after or "").splitlines())
            diff0 = diff
        else:
            r = subprocess.run(["git", "show", f"{base}:{f}"], cwd=repo, capture_output=True, text=True)
            before = r.stdout if r.returncode == 0 else None
            diff = _git(repo, "diff", f"-U{context_lines}", "--no-color", base, "--", f)
            diff0 = _git(repo, "diff", "-U0", "--no-color", base, "--", f)
        old_spans, new_spans = _edit_spans(diff0)
        hunks.extend(_hunks(f, diff0))
        numbered = _numbered_diff(diff)
        change.append(f"--- {f}" + ("  (new file)" if f in added else "  (deleted)" if after is None else ""))
        change.extend(numbered)
        change.append("")
        if f.endswith(".py"):
            spec.extend(_spec(f, before, after, old_spans, new_spans))
    text = "SPECIFICATION\n\n" + ("\n".join(spec) if spec else "(no Python definition is touched)\n") + \
           "\nCHANGE\n\n" + "\n".join(change)
    return Packet(text, hunks, tracked + added)


# ---------------------------------------------------------------- the model

class EndpointError(RuntimeError):
    pass


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))   # proxies never apply


def _request(cfg: ReviewConfig, path: str, body: Optional[dict]) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(cfg.endpoint + path, data=data, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        with _opener().open(req, timeout=cfg.timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise EndpointError(f"{path}: {e}") from e


def served_models(cfg: ReviewConfig) -> Dict[str, dict]:
    return {m.get("id", ""): m for m in _request(cfg, "/models", None).get("data", [])}


def served_window(cfg: ReviewConfig, entry: dict) -> Optional[int]:
    """The context window the server confirms for the model: vLLM lists `max_model_len` with the model;
    Ollama states `num_ctx` in the model's parameters (/api/show) only when its Modelfile sets it, and
    otherwise cuts a long prompt at a runtime default without saying so. None if nothing confirms one."""
    if isinstance(entry.get("max_model_len"), int):
        return entry["max_model_len"]
    root = urllib.parse.urlparse(cfg.endpoint)
    show = dict(cfg.__dict__, endpoint=f"{root.scheme}://{root.netloc}")
    try:
        r = _request(ReviewConfig(**show), "/api/show", {"model": cfg.model})
    except EndpointError:
        return None
    m = re.search(r"^num_ctx\s+(\d+)", str(r.get("parameters", "")), re.M)
    return int(m.group(1)) if m else None


def complete(cfg: ReviewConfig, packet: str, seed: int) -> Tuple[str, Dict[str, int]]:
    body = {"model": cfg.model, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
            "seed": seed % (2 ** 31),
            "messages": [{"role": "system", "content": cfg.prompt}, {"role": "user", "content": packet}]}
    if cfg.json_mode:
        body["response_format"] = {"type": "json_object"}
    r = _request(cfg, "/chat/completions", body)
    try:
        text = r["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise EndpointError(f"malformed completion: {str(r)[:200]}") from e
    return text, {k: int(v) for k, v in (r.get("usage") or {}).items() if isinstance(v, (int, float))}


def parse_findings(text: str) -> Optional[List[dict]]:
    """The reply's findings, or None if it is not the requested JSON object."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|```\s*$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None
    f = obj.get("findings") if isinstance(obj, dict) else None
    if not isinstance(f, list):
        return None
    out = []
    for x in f:
        if not isinstance(x, dict) or not isinstance(x.get("file"), str):
            continue
        line = x.get("line")
        try:
            line = int(line) if line is not None else None
        except (TypeError, ValueError):
            line = None
        quote = x.get("quote") if isinstance(x.get("quote"), str) else ""
        if line is None and not quote.strip():
            continue
        out.append({"file": x["file"], "line": line, "quote": quote, "concern": str(x.get("concern", ""))[:300]})
    return out


def _same_file(a: str, b: str) -> bool:
    a, b = a.replace("\\", "/").lstrip("./"), b.replace("\\", "/").lstrip("./")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def attribute(f: dict, hunks: Sequence[Hunk], slack: int) -> Optional[int]:
    """The index of the changed hunk a finding names, by line (within `slack` of the hunk's edits) or by
    quote (one of its added or removed lines); None if it names none."""
    q = f["quote"].strip()
    for i, h in enumerate(hunks):
        if not _same_file(f["file"], h.file):
            continue
        if q and any(q == l.strip() for l in h.added + h.removed):
            return i
    for i, h in enumerate(hunks):
        if _same_file(f["file"], h.file) and f["line"] is not None and h.lo - slack <= f["line"] <= h.hi + slack:
            return i
    return None


# ---------------------------------------------------------------- the producer

def make_spec_guided(cfg: ReviewConfig) -> Action:
    def run(ctx: Context, m: Meter) -> Outcome:
        host = urllib.parse.urlparse(cfg.endpoint)
        prov: Dict[str, str] = {"tool": "openai-chat", "endpoint": f"{host.hostname}:{host.port}",
                                "model": cfg.model, "prompt_sha": cfg.prompt_sha[:16], "k": str(cfg.k),
                                "min_agree": str(cfg.min_agree), "temperature": str(cfg.temperature),
                                "context_tokens": str(cfg.context_tokens)}
        blind = ["no executable witness: a finding is a reading of the change, never a validated violation",
                 "sampled replies; a server that batches requests can change a reply at a fixed seed",
                 "sees the change and the pinned state's signatures and docstrings, not the task text",
                 f"a change larger than the {cfg.context_tokens}-token context window is inconclusive"]
        m.tool_calls += 1
        try:
            served = served_models(cfg)
        except EndpointError as e:
            return Outcome("inconclusive", provenance=dict(prov, error=str(e)[:300]), blind_spots=blind,
                           status="error")
        if cfg.model not in served:
            return Outcome("inconclusive", provenance=dict(prov, error=f"model not served; served: {list(served)}"[:300]),
                           blind_spots=blind, status="error")
        window = served_window(cfg, served[cfg.model])
        prov["served_window"] = str(window)
        if window is None or window < cfg.context_tokens:
            return Outcome("inconclusive", provenance=dict(prov, error=f"the server confirms a context window of "
                           f"{window}, not {cfg.context_tokens}"), blind_spots=blind, status="error")
        pk = build_packet(ctx.repo, ctx.repo_state_sha, cfg.context_lines)
        prov.update(packet_sha=hashlib.sha256(pk.text.encode()).hexdigest()[:16], packet_chars=str(len(pk.text)),
                    hunks=str(len(pk.hunks)), files=str(len(pk.files)))
        # findings count on the package's own hunks; the candidate's tests are shown, as the change's stated
        # intent, but a reading of a test is not a finding about the library (F2's target is the package too)
        root = ctx.package_root.rstrip("/") + "/"
        scope = {i for i, h in enumerate(pk.hunks) if h.file.startswith(root)}
        prov["hunks_in_package"] = str(len(scope))
        if not scope:
            return Outcome("inconclusive", provenance=dict(prov, note=f"no change under {ctx.package_root}"),
                           blind_spots=blind)
        budget = cfg.context_tokens - cfg.max_tokens
        if (len(pk.text) + len(cfg.prompt)) / _CHARS_PER_TOKEN > budget:
            return Outcome("inconclusive", provenance=dict(prov, error="the packet exceeds the context window"),
                           blind_spots=blind)
        seeds = [sub_seed(ctx.seed, i) % (2 ** 31) for i in range(cfg.k)]
        votes: Dict[int, List[str]] = {}
        parsed, per_sample, unattributed, outside = 0, [], 0, 0
        for s in seeds:
            m.tool_calls += 1
            try:
                text, usage = complete(cfg, pk.text, s)
            except EndpointError as e:
                per_sample.append("error"); prov["error"] = str(e)[:300]
                continue
            m.add_tokens(usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)))
            if usage.get("prompt_tokens", 0) >= budget:     # the server cut the packet: the reply is not about it
                per_sample.append("truncated")
                continue
            fs = parse_findings(text)
            if fs is None:
                per_sample.append("unparsed")
                continue
            parsed += 1
            named: Dict[int, str] = {}
            for f in fs:
                i = attribute(f, pk.hunks, cfg.line_slack)
                if i is None:
                    unattributed += 1
                elif i not in scope:
                    outside += 1
                else:
                    named.setdefault(i, f["concern"])
            per_sample.append(str(len(named)))
            for i, concern in named.items():
                votes.setdefault(i, []).append(concern)
        flagged = sorted((i for i, v in votes.items() if len(v) >= cfg.min_agree), key=lambda i: -len(votes[i]))
        prov.update(seeds=",".join(map(str, seeds)), replies_parsed=str(parsed),
                    hunks_named_per_reply=",".join(per_sample), unattributed_findings=str(unattributed),
                    findings_outside_package=str(outside),
                    flagged=json.dumps([{"file": pk.hunks[i].file, "lines": f"{pk.hunks[i].lo}-{pk.hunks[i].hi}",
                                         "votes": len(votes[i]), "concern": votes[i][0]} for i in flagged])[:1500])
        if flagged:
            v = Validation(contradicts_reference=True, minimized=False, repeats_across_seeds=True,
                           second_tool_confirmed=False, witness_ref=f"{pk.hunks[flagged[0]].file}:"
                                                                    f"{pk.hunks[flagged[0]].lo}", sub_seeds=seeds)
            return Outcome("fail_unvalidated", v, prov, blind_spots=blind)
        if parsed < cfg.min_agree:
            return Outcome("inconclusive", provenance=prov, blind_spots=blind,
                           status="error" if "error" in prov else "ok")
        return Outcome("pass", Validation(sub_seeds=seeds), prov, blind_spots=blind)

    return Action("ch.spec_guided", "F3", "challenger", f"llm-review:{cfg.model}", run, priority=63)


def register(reg: Registry, c: Dict[str, Any], base_dir: Path = Path(".")) -> Action:
    return reg.register(make_spec_guided(review_config(c, base_dir)))
