"""End-to-end analysis: evidence records -> fitted evidence model -> nine policies
at three ceilings -> the headline quantities and the fixed-sequence primary test.

    python -m rcca.analyze --out corpus/flask --freeze configs/freeze.json \
        [--fit-partition train] [--eval-partition test] [--quantiles 0.3 0.6 0.9]
    python -m rcca.analyze --out corpus/flask-dev --dev      # development: fit and evaluate on all

Inputs under --out: records.jsonl, manifest.jsonl, labels.jsonl.
Outputs under --out/analysis: fit.json, ceilings.json, nominal_costs.json,
table_rq1.json, table_rq1.md, decisions.jsonl (one row per policy x candidate x
rep x ceiling, with its realized cost, stop reason, overrun flag and overhead_s,
the policy's own wall-clock), and traces.jsonl (one row per decision: the
acquisition steps, each with its nominal and realized cost, the posterior before
and after, the gain and the outcome). overhead_s is the one value that differs
between two runs; traces.jsonl is byte for byte reproducible.

Ceilings are per-dimension quantiles of the full-evidence cost per cell on
the fit partition: the 60th percentile is the primary
setting, 30th and 90th the secondary frontier points. They bound tokens, tool
calls and CI-equivalents; wall-clock time is an outcome, reported but never a
ceiling (results.md E2). Once the freeze record holds ceilings (stage frozen
pastes ceilings.json into it), a run replays the record's instead of computing
its own, fit.json says which, and at stage unblinded the record's are required.

Policies select on nominal costs and are charged realized costs after acquiring
(Global Constraints). An action's nominal cost is the per-dimension median of
its realized cost over the fit partition's records, never another partition's;
nominal_costs.json and the record's nominal_costs follow the ceilings' rule
above. The weights w_j of C_w are the record's cost_weights; without a record
they are equal.

Under a freeze record (`rcca.freeze`) the run needs stage frozen, fits on
'train' only, evaluates on a partition the stage has opened (test once
unblinded), and takes the loss constants and m_S from the record; at stage
frozen a tuning run may pass tau_qual and eta with --loss. Without a record,
--dev and an --out ending in -dev are required, and the loss constants come
from --loss or the dev defaults, which the output says.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np

from .decision import EvidenceModel, LossParams, POLICIES
from .fit import apply_prior, fit_likelihood, fit_prior_logit, load_matrices, strip_labels
from .freeze import (Freeze, FreezeError, check_manifest, frozen_ceilings, frozen_cost_weights, frozen_loss,
                     frozen_margin, frozen_nominal_costs, gate, is_dev, require_fit_partitions, require_open)
from .matrix import CEILING_DIMS, COST_DIMS, Action, Ceilings, Cost, EvidenceMatrix, Source
from .replay import Row, policy_intervals, primary_analysis, replay, summarize

DEV_LOSS = {"kappa": 100.0, "lambda_R": 10.0, "lambda_E": 2.0, "tau_qual": 0.02, "eta": 1.0, "frozen": False}
PRIORITY = {"f1.repo_ci": 10, "f2.static_security": 20, "f2.dependency": 21, "f3.api_contract": 30,
            "f3.resource_budget": 31, "f1.generated_tests": 40, "f4.impact_revert": 50,
            "ch.property": 60, "ch.fuzz": 61, "ch.differential": 62, "ch.spec_guided": 63, "ch.metamorphic": 64}


def _read_jsonl(p: Path) -> List[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def actions_from_records(records: List[dict], fit_candidate_ids: Set[str],
                         frozen: Optional[Dict[str, Dict[str, float]]] = None) -> List[Action]:
    """The actions the records name, each priced at its nominal cost (Global Constraints: acquisition
    decisions use nominal costs estimated on the training partition): the per-dimension median of its
    realized costs over the records of `fit_candidate_ids` only, so no record from another partition prices
    an action, or, once the freeze record holds them, the frozen nominal costs (`frozen`). A policy selects
    on the nominal cost and is charged a record's realized cost after acquiring it (rcca.decision). An
    action without a fit-partition record is refused (ValueError), and so is one `frozen` does not price
    (FreezeError)."""
    seen: Dict[str, dict] = {}
    costs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in records:
        seen.setdefault(r["action_id"], r)
        if r["candidate_id"] in fit_candidate_ids:
            c = r["cost"]
            costs[r["action_id"]].append(np.array([c.get(d, 0) for d in COST_DIMS], float))
    if frozen is not None:
        lacking = sorted(a for a in seen if a not in frozen)
        if lacking:
            raise FreezeError(f"the freeze record's nominal_costs do not price {', '.join(lacking)}, which the "
                              f"records hold; nominal costs are frozen with the action set (Task B5)")
    else:
        unpriced = sorted(a for a in seen if not costs[a])
        if unpriced:
            raise ValueError(f"no fit-partition record of {', '.join(unpriced)}, so no nominal cost: a cost "
                             f"from another partition never prices an action")
    out = []
    for aid, r in seen.items():
        nominal = Cost(**frozen[aid]) if frozen is not None else Cost(*np.median(np.stack(costs[aid]), axis=0))
        src = Source.CHALLENGER if r.get("source") == "challenger" else Source.AUTHOR_SIDE
        out.append(Action(aid, r.get("family", "F1"), src, r.get("producer", aid), nominal,
                          PRIORITY.get(aid, 100), is_repo_ci=(aid == "f1.repo_ci")))
    return sorted(out, key=lambda a: (a.priority, a.id))


def _label(q: float) -> str:
    return f"q{int(round(q * 100))}"


def ceilings_from_cost(matrices: Sequence[EvidenceMatrix], quantiles: Sequence[float]) -> Dict[str, Ceilings]:
    """Per-dimension quantiles of the full-evidence cost per cell, over the ceiling dimensions only:
    wall-clock seconds would make the ceilings, and so the decisions, depend on the machine."""
    totals = np.stack([sum((r.cost.ceiling_vec() for r in m.records.values()), np.zeros(len(CEILING_DIMS)))
                       for m in matrices])
    out = {}
    for q in quantiles:
        v = np.quantile(totals, q, axis=0)
        v = np.maximum(v, [1.0, 1.0, 0.01])                 # a zero ceiling would make C_w undefined
        out[_label(q)] = Ceilings(*map(float, v))
    return out


def ceilings_from_record(frozen: Dict[str, Dict[str, float]], quantiles: Sequence[float]) -> Dict[str, Ceilings]:
    """The freeze record's ceilings at the requested quantiles; a quantile the record does not fix is refused."""
    missing = [_label(q) for q in quantiles if _label(q) not in frozen]
    if missing:
        raise FreezeError(f"the freeze record fixes ceilings {', '.join(frozen)}, not {', '.join(missing)}")
    return {_label(q): Ceilings(**frozen[_label(q)]) for q in quantiles}


def run(out: Union[Path, Sequence[Path]], loss: dict, fit_partition: Optional[str], eval_partition: Optional[str],
        quantiles: Sequence[float], B_boot: int = 2000, m_S: float = 0.05, seed: int = 0,
        freeze: Optional[Freeze] = None, analysis_dir: Optional[Path] = None) -> dict:
    """`out` is one corpus root or several: the confirmatory study is one analysis over flask and click,
    so their matrices are pooled and the task-level bootstrap stratifies by each manifest's repo. With
    one root the analysis is written under it (`analysis/`); with several, `analysis_dir` names where.

    `freeze` past stage dev applies the partition firewall before anything is read or written:
    fit on 'train', evaluate on an open partition, no excluded task in the manifest, no matrix from
    a closed partition on disk, and no fallback to all matrices when a partition is empty. A record
    that holds ceilings or nominal costs, at any stage, supplies them; otherwise they are computed from
    the fit matrices. Under a record the cost weights are the record's; without one they are equal.

    Every output is a function of the inputs alone except `overhead_s` in decisions.jsonl, the policy's
    own wall-clock (rcca.decision.run_policy), which varies run to run; traces.jsonl holds no timing, so
    two runs write it byte for byte."""
    require_fit_partitions(freeze, fit_partition, eval_partition)
    frozen = frozen_ceilings(freeze)                           # malformed, or unset at unblinded: refused
    fixed = ceilings_from_record(frozen, quantiles) if frozen is not None else None
    nominal = frozen_nominal_costs(freeze)                     # likewise
    weights = frozen_cost_weights(freeze)                      # malformed: refused; None without a record
    outs = [Path(out)] if isinstance(out, (str, Path)) else [Path(o) for o in out]
    if analysis_dir is None:
        if len(outs) > 1:
            raise ValueError("several corpus roots: name the analysis directory (analysis_dir / --analysis-dir)")
        analysis_dir = outs[0] / "analysis"
    rows = [r for o in outs for r in _read_jsonl(o / "manifest.jsonl")]
    check_manifest(freeze, rows, ", ".join(str(o / "manifest.jsonl") for o in outs))
    notes: List[str] = []
    all_m = [m for o in outs for m in load_matrices(o / "records.jsonl", o / "manifest.jsonl", o / "labels.jsonl")]
    manifest = {m["candidate_id"]: m for m in rows}
    part_of = lambda m: manifest[m.candidate_id].get("partition", "")
    require_open(freeze, {part_of(m) for m in all_m})          # test matrices refused before unblinding
    fit_m = [m for m in all_m if fit_partition is None or part_of(m) == fit_partition]
    eval_m = [m for m in all_m if eval_partition is None or part_of(m) == eval_partition]
    if not is_dev(freeze) and not (fit_m and eval_m):
        raise FreezeError(f"no matrices in partition {fit_partition if not fit_m else eval_partition!r}; "
                          f"a run under a freeze record never falls back to all matrices")
    if not fit_m or fit_partition is None:
        fit_m = all_m; notes.append("fit on ALL matrices (no training partition): leaky, dev only")
    if not eval_m or eval_partition is None:
        eval_m = all_m; notes.append("evaluated on ALL matrices (no test partition): dev only")
    if not loss.get("frozen", False):
        notes.append("loss constants, tau_qual and eta are dev defaults, not frozen")
    # nominal costs from the fit partition's records only, or the record's: an action the record does
    # not price is refused here, before anything is written
    actions = actions_from_records([r for o in outs for r in _read_jsonl(o / "records.jsonl")],
                                   {m.candidate_id for m in fit_m}, nominal)
    adir = Path(analysis_dir); adir.mkdir(parents=True, exist_ok=True)

    lik = fit_likelihood(fit_m, alpha=0.5)
    a, b = fit_prior_logit(fit_m)
    apply_prior(all_m, a, b)
    model = EvidenceModel(lik, s_tilde=float(loss.get("s_tilde", 0.6)))
    lp = LossParams(loss["kappa"], loss["lambda_R"], loss["lambda_E"], loss["tau_qual"], loss["eta"])
    ceilings = fixed if fixed is not None else ceilings_from_cost(fit_m, quantiles)
    if weights is not None:
        ceilings = {k: replace(c, weights=weights) for k, c in ceilings.items()}
    source = "freeze-record" if fixed is not None else "computed"
    (adir / "fit.json").write_text(json.dumps({"prior": {"a": a, "b": b}, "likelihood": {k: [v[0].tolist(), v[1].tolist()] for k, v in lik.items()},
                                               "n_fit": len(fit_m), "n_eval": len(eval_m), "ceilings_source": source,
                                               "nominal_costs_source": "freeze-record" if nominal is not None else "computed",
                                               "cost_weights_source": "freeze-record" if weights is not None else "default",
                                               "notes": notes}, indent=1))
    # the shapes of freeze.ceilings and freeze.nominal_costs, so stage frozen can paste these files into
    # the record (Task B5)
    (adir / "ceilings.json").write_text(json.dumps({k: {d: getattr(c, d) for d in CEILING_DIMS}
                                                    for k, c in ceilings.items()}, indent=1))
    (adir / "nominal_costs.json").write_text(json.dumps({act.id: act.nominal_cost.as_dict() for act in actions},
                                                        indent=1))

    # policies see no labels; the evaluator joins them back by candidate id afterwards
    hidden = {m.candidate_id: (m.truth_unsafe, m.truth_severity) for m in eval_m}
    blind = strip_labels(eval_m)
    table: Dict[str, Dict[str, dict]] = {}
    dec_rows: List[dict] = []
    trace_rows: List[dict] = []
    tests: Dict[str, dict] = {}
    no_intervals: set = set()
    for cname, B in ceilings.items():
        rows = replay(blind, actions, lp, model, B, seed=seed)
        for r in rows:                                         # evaluator attaches truth
            r.unsafe, r.severity = hidden[r.candidate_id][0], hidden[r.candidate_id][1] or 0.0
        table[cname] = {p: summarize(rows, p) for p in POLICIES}
        for r in rows:
            # decisions.jsonl: the decision and its realized cost, without the steps; traces.jsonl: the
            # steps, keyed to it, with no label and no timing
            dec_rows.append({"ceiling": cname, **{k: v for k, v in r.__dict__.items() if k != "trace"}})
            trace_rows.append({"policy": r.policy, "candidate_id": r.candidate_id, "rep": r.rep, "ceiling": cname,
                               "stop_reason": r.stop_reason, "overrun": r.overrun,
                               "steps": [st.as_dict() for st in r.trace]})
        n_tasks = len({r.task for r in rows})
        if n_tasks >= 10:
            gate, prim = primary_analysis(rows, m_S=m_S, B=B_boot, seed=seed)
            tests[cname] = {"gate": gate.__dict__, "primary": prim.__dict__, "n_tasks": n_tasks}
        else:
            tests[cname] = {"skipped": f"{n_tasks} tasks; the task-level bootstrap needs at least 10"}
        # per-policy task-level means and 95 % intervals
        for p in POLICIES:
            ci = policy_intervals(rows, p, B=B_boot, seed=seed)
            if ci:
                table[cname][p].update(ci)
            else:
                no_intervals.add(p)
    if no_intervals:
        notes.append("per-policy 95 % intervals omitted: the task-level cluster bootstrap "
                     "needs at least 10 tasks and no singleton repository stratum")
    for name, lines in (("decisions.jsonl", dec_rows), ("traces.jsonl", trace_rows)):
        with open(adir / name, "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
    # outcome mix per action, safe vs unsafe (the raw material of H2)
    mix: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: {"safe": defaultdict(int), "unsafe": defaultdict(int)})
    for m in eval_m:
        for aid, rec in m.records.items():
            mix[aid]["unsafe" if m.truth_unsafe else "safe"][rec.outcome.value] += 1
    per_repo: Dict[str, Set[str]] = defaultdict(set)
    for m in eval_m:
        per_repo[str(manifest[m.candidate_id].get("repo", ""))].add(m.task)
    result = {"notes": notes, "n_matrices": len(eval_m), "n_tasks": len({m.task for m in eval_m}),
              "repos": {k: len(v) for k, v in sorted(per_repo.items())}, "corpus_roots": [str(o) for o in outs],
              "actions": [a.id for a in actions], "ceilings": {k: v.__dict__ for k, v in ceilings.items()},
              "ceilings_source": source,
              "table": table, "tests": tests, "outcome_mix": {k: {s: dict(d) for s, d in v.items()} for k, v in mix.items()}}
    (adir / "table_rq1.json").write_text(json.dumps(result, indent=1, default=float))
    (adir / "table_rq1.md").write_text(render_md(result))
    return result


def render_md(res: dict) -> str:
    lines = [f"# Table rq1 draft ({res['n_tasks']} tasks, {res['n_matrices']} matrices)", ""]
    for n in res["notes"]:
        lines.append(f"> {n}")
    lines.append("")
    for cname, pol in res["table"].items():
        c = res["ceilings"][cname]
        lines.append(f"## Ceiling {cname}: tokens {c['tokens']:.0f}, tool calls {c['tool_calls']:.1f}, CI-equiv {c['ci_equiv']:.2f}")
        lines.append("")
        lines.append("| Policy | L | U | S | C | Esc S | Esc U | Rej S | Rej U | tool calls (med) | CI-equiv (med) | wall s (med) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in POLICIES:
            s = pol[p]
            lines.append(f"| {p} | {s['L']:.3f} | {s['U']:.3f} | {s['S']:.3f} | {s['C']:.3f} | {s['esc_S']:.2f} | {s['esc_U']:.2f} | "
                         f"{s['rej_S']:.2f} | {s['rej_U']:.2f} | {s['tool_calls_median']:.1f} | {s['ci_equiv_median']:.2f} | {s['wall_s_median']:.0f} |")
        t = res["tests"][cname]
        if "skipped" in t:
            lines.append(f"\nPrimary analysis skipped: {t['skipped']}.")
        else:
            g, pr = t["gate"], t["primary"]
            lines.append(f"\nGate: Delta_S = {g['delta_S_hat']:.3f}, upper = {g['upper']:.3f} vs m_S = {g['margin']}: "
                         f"{'passed' if g['passed'] else 'failed'}. Primary RCCA-CI L: {pr['estimate']:.3f} "
                         f"[{pr['lower']:.3f}, {pr['upper']:.3f}], one-sided p = {pr['p_one_sided_less']:.3f}, {t['n_tasks']} tasks.")
        lines.append("")
    lines.append("## Outcome mix per action (safe | unsafe)")
    lines.append("")
    lines.append("| Action | safe: pass / fail_v / fail_u / inconcl | unsafe: pass / fail_v / fail_u / inconcl |")
    lines.append("|---|---|---|")
    for aid, mx in res["outcome_mix"].items():
        f = lambda d: " / ".join(str(d.get(k, 0)) for k in ("pass", "fail_validated", "fail_unvalidated", "inconclusive"))
        lines.append(f"| {aid} | {f(mx['safe'])} | {f(mx['unsafe'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", action="append", required=True, help="corpus root; repeat for a study over several")
    ap.add_argument("--analysis-dir", default=None, help="where the analysis is written (required for several --out)")
    ap.add_argument("--loss", default=None)
    ap.add_argument("--fit-partition", default="train")
    ap.add_argument("--eval-partition", default="test")
    ap.add_argument("--quantiles", type=float, nargs="*", default=[0.3, 0.6, 0.9])
    ap.add_argument("--m-S", type=float, default=None, help="default: the freeze record's m_S, else 0.05")
    ap.add_argument("--dev", action="store_true",
                    help="development run: fit and evaluate on all matrices, no freeze record, --out ends in -dev")
    ap.add_argument("--freeze", default=None, help="freeze record (rcca.freeze); needed for any --out not ending in -dev")
    a = ap.parse_args(argv)
    fit, ev = (None, None) if a.dev else (a.fit_partition, a.eval_partition)
    given = json.load(open(a.loss)) if a.loss else None
    try:
        for o in a.out:
            fz = gate(a.freeze, dev=a.dev, out=Path(o), min_stage="frozen", partitions=[fit, ev])
        adir = Path(a.analysis_dir) if a.analysis_dir else None
        if a.dev and adir is not None and not any(part.endswith("-dev") for part in adir.parts):
            raise FreezeError(f"a development run writes its analysis under a directory named *-dev, not {adir}")
        if is_dev(fz):
            loss, m_S = (given or dict(DEV_LOSS)), (0.05 if a.m_S is None else a.m_S)
        else:
            loss, m_S = frozen_loss(fz, given), frozen_margin(fz, a.m_S)
        res = run([Path(o) for o in a.out], loss, fit, ev, a.quantiles, m_S=m_S, freeze=fz, analysis_dir=adir)
    except ValueError as e:
        print(f"rcca.analyze: refused: {e}", file=sys.stderr)
        return 2
    except FreezeError as e:
        print(f"rcca.analyze: refused: {e}", file=sys.stderr)
        return 2
    print(render_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
