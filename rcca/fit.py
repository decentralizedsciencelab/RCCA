"""The seam between the evidence layer and the decision layer.

Two functions the real study needs that the synthetic generator hid:

* `load_matrices` reads producer output (one JSON record per (candidate, rep,
  action)) plus a candidate manifest and, separately, a labels file. Policies
  never see labels; `load_matrices` attaches them only so the evaluator can
  score decisions afterwards, and `strip_labels` removes them for a policy run.
* `fit_likelihood` estimates p(o | Y, action) on the training partition with
  additive smoothing, giving the `EvidenceModel.likelihood` tables.

Record layout on disk (spec §10), one JSON object per line in `records.jsonl`:

    {"candidate_id": "...", "rep": 0, "action_id": "f1.repo_ci",
     "outcome": "pass|fail_validated|fail_unvalidated|inconclusive",
     "validated": false, "status": "ok|error|timeout",
     "cost": {"tokens": 0, "tool_calls": 1, "ci_equiv": 1.0, "wall_s": 61.2},
     "provenance": {...}}

Manifest `manifest.jsonl`, one object per candidate:

    {"candidate_id": "t17:u:m0", "task": "t17", "repo": "flask", "mode": "m0",
     "safe_twin": "t17:s:m0", "partition": "test", "commit": "...",
     "risk_features": {"risk_logit": -1.3, ...}, "ood": false}

Labels `labels.jsonl`, one object per candidate, written by the hidden oracle:

    {"candidate_id": "t17:u:m0", "unsafe": true, "severity": 0.7}
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

from .matrix import Cost, EvidenceMatrix, EvidenceRecord, Outcome, OUTCOMES


def _read_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_matrices(records_path: Path, manifest_path: Path,
                  labels_path: Optional[Path] = None,
                  partition: Optional[str] = None) -> List[EvidenceMatrix]:
    """Assemble one EvidenceMatrix per (candidate, rep).

    `partition` filters the manifest (train / tune / test). Labels are
    attached when `labels_path` is given; pass None for a policy run.
    """
    manifest = {m["candidate_id"]: m for m in _read_jsonl(Path(manifest_path))}
    if partition is not None:
        manifest = {k: v for k, v in manifest.items() if v.get("partition") == partition}
    labels = {}
    if labels_path is not None:
        labels = {l["candidate_id"]: l for l in _read_jsonl(Path(labels_path))}
    cells: Dict[Tuple[str, int], Dict[str, EvidenceRecord]] = defaultdict(dict)
    for r in _read_jsonl(Path(records_path)):
        cid = r["candidate_id"]
        if cid not in manifest:
            continue
        c = r["cost"]
        cells[(cid, int(r["rep"]))][r["action_id"]] = EvidenceRecord(
            action_id=r["action_id"], outcome=Outcome(r["outcome"]),
            cost=Cost(float(c.get("tokens", 0)), float(c.get("tool_calls", 0)),
                      float(c.get("ci_equiv", 0)), float(c.get("wall_s", 0))),
            validated=bool(r.get("validated", r["outcome"] == "fail_validated")),
            status=r.get("status", "ok"), provenance=r.get("provenance", {}))
    out: List[EvidenceMatrix] = []
    for (cid, rep), recs in sorted(cells.items()):
        m = manifest[cid]
        lab = labels.get(cid)
        out.append(EvidenceMatrix(
            candidate_id=cid, rep=rep, repo=m["repo"], records=recs,
            risk_features=dict(m.get("risk_features", {})), ood=bool(m.get("ood", False)),
            truth_unsafe=(None if lab is None else bool(lab["unsafe"])),
            truth_severity=(None if lab is None else float(lab.get("severity", 0.0))),
            safe_twin=m.get("safe_twin"), task=m.get("task")))
    return out


def strip_labels(matrices: Iterable[EvidenceMatrix]) -> List[EvidenceMatrix]:
    """Return copies with hidden labels removed, for any code path a policy touches."""
    out = []
    for m in matrices:
        out.append(EvidenceMatrix(m.candidate_id, m.rep, m.repo, m.records, m.risk_features,
                                  m.ood, None, None, m.safe_twin, m.task))
    return out


def fit_likelihood(matrices: Iterable[EvidenceMatrix], alpha: float = 0.5,
                   action_ids: Optional[Iterable[str]] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Estimate p(o | Y=1, action) and p(o | Y=0, action) by smoothed counts.

    Additive (Dirichlet) smoothing with pseudo-count `alpha` per outcome keeps
    every likelihood strictly positive, so no single record can drive the
    posterior to 0 or 1. Only labeled matrices contribute. Actions that never
    appear get the uniform table.
    """
    counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros((2, 4)))
    for m in matrices:
        if m.truth_unsafe is None:
            raise ValueError(f"unlabeled matrix {m.candidate_id}; fit needs training labels")
        y = 1 if m.truth_unsafe else 0
        for aid, rec in m.records.items():
            if rec.status != "ok" and rec.outcome != Outcome.INCONCLUSIVE:
                continue
            counts[aid][y, OUTCOMES.index(rec.outcome)] += 1
    ids = set(counts) | set(action_ids or [])
    lik: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for aid in sorted(ids):
        c = counts[aid] + alpha
        p1 = c[1] / c[1].sum(); p0 = c[0] / c[0].sum()
        lik[aid] = (p1, p0)
    return lik


def fit_prior_logit(matrices: Iterable[EvidenceMatrix], feature: str = "risk_logit") -> Tuple[float, float]:
    """One-feature logistic calibration of the risk logit on labeled matrices.

    Returns (a, b) such that logit p(Y=1) = a + b * r. A full risk model is
    out of scope here; this is the minimal calibration that turns a raw risk
    score into the prior the decision layer expects. Fitted by Newton steps.
    """
    xs, ys = [], []
    for m in matrices:
        if m.truth_unsafe is None:
            continue
        xs.append(float(m.risk_features.get(feature, 0.0))); ys.append(1.0 if m.truth_unsafe else 0.0)
    x = np.asarray(xs); y = np.asarray(ys)
    if x.size == 0:
        return 0.0, 1.0
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + 1e-3 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-3 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def apply_prior(matrices: Iterable[EvidenceMatrix], a: float, b: float,
                feature: str = "risk_logit") -> None:
    """Write the calibrated prior into each matrix as `rho0` (in place)."""
    for m in matrices:
        r = float(m.risk_features.get(feature, 0.0))
        z = a + b * r
        m.risk_features["rho0"] = float(np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6))


def write_records(path: Path, records: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
