"""Tuning (plan Task B5): eta and tau_qual from a fixed grid, on the tuning partition.

The loss constants kappa, lambda_R and lambda_E are the record's (D7); eta and
tau_qual are selected here, before the test partition is opened, by RCCA's own
loss on the tuning partition among the grid points that keep its safe-change
acceptance within the noninferiority margin: S_RCCA >= 1 - m_S, which, since
admission makes S_CI = 1 by construction, is exactly what the gate will ask of
it. When no point is feasible the point with the highest S is reported and
flagged; the record is then not filled and the design is revisited. m_S is the
D1 default unless the operator passes another value, and the automatic-
acceptance coverage C of the selected point is reported against the qualification
qualification floor (D8).

Each grid point is a full `rcca.analyze.run` (fit on train, evaluate on tune)
written under `<analysis_dir>/eta<eta>_tau<tau>/`; `tuning.json` beside them
holds the grid, the selection, and `freeze_fields` in the shape
`python -m rcca.freeze fill --from` pastes into the record. The ceilings and
nominal costs come from the training fit and do not depend on the grid point.

    python -m rcca.tune --out corpus/flask --out corpus/click --analysis-dir analysis/tune --freeze configs/freeze.json
    python -m rcca.tune --out corpus/flask-api-dev --dev
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import analyze
from .freeze import Freeze, FreezeError, frozen_loss, frozen_margin, gate, is_dev

GRID_ETA = (0.5, 1.0, 2.0, 4.0)
GRID_TAU = (0.1, 0.2)            # the plan's {0.01, 0.02, 0.05} caps safe acceptance below the gate (docs/findings/2026-10-21-tuning-feasibility.md)
RQ2_COVERAGE_FLOOR = 0.42        # the qualification stream needs this much automatic acceptance
_EPS = 1e-12


def run(outs: Sequence[Path], loss: dict, *, etas: Sequence[float] = GRID_ETA, taus: Sequence[float] = GRID_TAU,
        m_S: float = 0.05, quantiles: Sequence[float] = (0.3, 0.6, 0.9), primary: str = "q60", B_boot: int = 2000,
        seed: int = 0, freeze: Optional[Freeze] = None, fit_partition: str = "train", eval_partition: str = "tune",
        analysis_dir: Optional[Path] = None) -> dict:
    if eval_partition != "tune" or fit_partition != "train":
        raise ValueError("tuning fits on the train partition and evaluates on the tune partition, never on test")
    outs = [Path(o) for o in outs]
    if analysis_dir is None:
        if len(outs) > 1:
            raise ValueError("several corpus roots: name the analysis directory (analysis_dir / --analysis-dir)")
        analysis_dir = outs[0] / "analysis" / "tune"
    analysis_dir = Path(analysis_dir)
    grid, ceilings, nominal = [], None, None
    for eta in etas:
        for tau in taus:
            sub = analysis_dir / f"eta{eta:g}_tau{tau:g}"
            res = analyze.run(outs, {**loss, "eta": float(eta), "tau_qual": float(tau)}, fit_partition, eval_partition,
                              list(quantiles), B_boot=B_boot, m_S=m_S, seed=seed, freeze=freeze, analysis_dir=sub)
            if primary not in res["table"]:
                raise ValueError(f"primary ceiling {primary!r} is not among {sorted(res['table'])}")
            s = res["table"][primary]["rcca"]
            grid.append({"eta": float(eta), "tau_qual": float(tau), "S": s["S"], "U": s["U"], "L": s["L"], "C": s["C"],
                         "esc_S": s["esc_S"], "esc_U": s["esc_U"], "ci_equiv_median": s["ci_equiv_median"],
                         "tokens_median": s["tokens_median"], "feasible": bool(s["S"] >= 1 - m_S - _EPS),
                         "n_tasks_tune": res["n_tasks"], "n_matrices_tune": res["n_matrices"], "dir": str(sub)})
            if ceilings is None:
                ceilings = json.loads((sub / "ceilings.json").read_text())
                nominal = json.loads((sub / "nominal_costs.json").read_text())
            print(f"eta {eta:g} tau_qual {tau:g}: S {s['S']:.3f} L {s['L']:.4f} C {s['C']:.3f} "
                  f"ci-equiv {s['ci_equiv_median']:.2f}{'' if grid[-1]['feasible'] else '  (infeasible)'}", flush=True)
    feasible = [g for g in grid if g["feasible"]]
    if feasible:
        sel = min(feasible, key=lambda g: (g["L"], g["ci_equiv_median"], g["eta"], g["tau_qual"]))
    else:
        sel = max(grid, key=lambda g: (g["S"], -g["L"]))
    notes = [] if feasible else [f"no grid point keeps S_RCCA >= {1 - m_S:.3f}; the highest-S point is reported and "
                                 f"the record must not be filled from it"]
    out = {"grid": grid, "selected": sel, "feasible_points": len(feasible), "m_S": m_S, "primary_ceiling": primary,
           "fit_partition": fit_partition, "eval_partition": eval_partition, "corpus_roots": [str(o) for o in outs],
           "rq2_coverage_floor": RQ2_COVERAGE_FLOOR, "rq2_qualifiable": bool(sel["C"] >= RQ2_COVERAGE_FLOOR),
           "freeze_fields": {"loss": {"tau_qual": sel["tau_qual"], "eta": sel["eta"]}, "m_S": m_S,
                             "ceilings": ceilings, "nominal_costs": nominal},
           "notes": notes}
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "tuning.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", action="append", required=True, help="corpus root; repeat for a study over several")
    ap.add_argument("--analysis-dir", default=None)
    ap.add_argument("--eta", type=float, nargs="*", default=list(GRID_ETA))
    ap.add_argument("--tau", type=float, nargs="*", default=list(GRID_TAU))
    ap.add_argument("--m-S", type=float, default=None, help="default: the record's m_S, else 0.05 (D1)")
    ap.add_argument("--quantiles", type=float, nargs="*", default=[0.3, 0.6, 0.9])
    ap.add_argument("--primary", default="q60")
    ap.add_argument("--loss", default=None, help="dev only: loss constants JSON (kappa, lambda_R, lambda_E, s_tilde)")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--freeze", default=None)
    a = ap.parse_args(argv)
    given = json.load(open(a.loss)) if a.loss else None
    try:
        for o in a.out:
            fz = gate(a.freeze, dev=a.dev, out=Path(o), min_stage="frozen", partitions=["train", "tune"])
        adir = Path(a.analysis_dir) if a.analysis_dir else None
        if a.dev and adir is not None and not any(part.endswith("-dev") for part in adir.parts):
            raise FreezeError(f"a development run writes its analysis under a directory named *-dev, not {adir}")
        if is_dev(fz):
            loss, m_S = (given or dict(analyze.DEV_LOSS)), (0.05 if a.m_S is None else a.m_S)
        else:
            # the record fixes the constants; the tuned pair is supplied per grid point
            loss = frozen_loss(fz, {"tau_qual": a.tau[0], "eta": a.eta[0]})
            m_S = frozen_margin(fz, a.m_S)
        res = run([Path(o) for o in a.out], loss, etas=a.eta, taus=a.tau, m_S=m_S, quantiles=a.quantiles,
                  primary=a.primary, freeze=fz, analysis_dir=adir)
    except (FreezeError, ValueError) as e:
        print(f"rcca.tune: refused: {e}", file=sys.stderr)
        return 2
    sel = res["selected"]
    print(f"selected eta {sel['eta']:g}, tau_qual {sel['tau_qual']:g}: S {sel['S']:.3f} L {sel['L']:.4f} C {sel['C']:.3f}"
          + ("" if sel["feasible"] else "  -- INFEASIBLE: the record must not be filled"))
    for n in res["notes"]:
        print(f"note: {n}")
    return 0 if sel["feasible"] else 3


if __name__ == "__main__":
    sys.exit(main())
