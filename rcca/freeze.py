"""Freeze record and stage firewall.

The freeze record (`configs/freeze.json`) says what the confirmatory study is:
the salt every seeded choice derives from, the repository and evidence configs
it governs, the seeder knobs, the loss constants, the cost weights w_j of C_w,
the producers and their tool digests, the authors and the oracle, the tasks
excluded for development exposure, the task-frame thresholds (min_tasks over the
repositories, min_tasks_per_repo each; corpus.build select and census check
them), and, once fitted and tuned, m_S, the ceilings, the nominal costs,
tau_qual and eta. Every command that writes an
output the paper may use runs under it, and the record's stage decides what the
command may touch:

    dev         no salt yet; any partition, but every output directory ends in -dev
    pre-freeze  salt drawn; task selection, partitioning, environments, baselines;
                the training partition only
    frozen      training and tuning evidence, fits and tuning; test matrices
                refused; `excluded_tasks` enforced on every manifest read
    unblinded   test opened once; every fitted field present, the record signed

A command declares the earliest stage it runs at and the partitions it touches;
`gate` refuses it, with a one-line reason, when the stage is too early, a
partition is still closed, one of its configs is not listed in the record, or
`validate` has a finding that binds at the stage. The fields stage frozen fills
itself (m_S, ceilings, nominal_costs, loss.tau_qual, loss.eta) bind from
unblinded on. The ceilings take the shape of analysis/ceilings.json, q30, q60
and q90 of tokens, tool calls and CI-equivalents (wall-clock is an outcome, not
a ceiling), and the nominal costs that of analysis/nominal_costs.json, each
action's four metered dimensions as non-negative numbers; a malformed object is
a finding that binds at once. cost_weights, three positive numbers summing to 1
over the ceiling dimensions, are fixed when the record is written. Without
a record a command is a development run: it needs --dev and an output directory
whose name ends in -dev. From pre-freeze on, the record's salt replaces each
config's `freeze_salt` for partitions, seeded defects, and evidence cell seeds.

    python -m rcca.freeze validate --freeze configs/freeze.json   # findings; exit 1 if any
    python -m rcca.freeze init     --freeze configs/freeze.json   # draw the salt once; dev -> pre-freeze
    python -m rcca.freeze seal     --freeze configs/freeze.json   # pin freeze_id to the record and its configs
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .matrix import CEILING_DIMS, COST_DIMS

STAGES = ("dev", "pre-freeze", "frozen", "unblinded")
PARTITIONS = ("train", "tune", "test")
OPEN = {"dev": PARTITIONS, "pre-freeze": ("train",), "frozen": ("train", "tune"), "unblinded": PARTITIONS}
FIELDS = ("freeze_id", "stage", "signed", "created", "salt", "repos", "evidence_configs", "knobs", "loss",
          "cost_weights", "m_S", "ceilings", "nominal_costs", "producers", "authors", "oracle", "excluded_tasks",
          "min_tasks", "min_tasks_per_repo", "tool_digests")
LOSS_CONSTANTS = ("kappa", "lambda_R", "lambda_E", "s_tilde")   # fixed when the record is written
TUNED_LOSS = ("tau_qual", "eta")                                 # selected on the tuning partition
CEILING_QUANTILES = ("q30", "q60", "q90")                        # q60 is the primary ceiling
_TYPES = {"stage": str, "signed": bool, "repos": list, "evidence_configs": list, "knobs": dict, "loss": dict,
          "cost_weights": list, "producers": list, "excluded_tasks": list, "min_tasks": int,
          "min_tasks_per_repo": int, "tool_digests": dict}
_NULLABLE = {"freeze_id": str, "created": str, "salt": str, "m_S": (int, float), "ceilings": dict,
             "nominal_costs": dict}
_SALT = re.compile(r"[0-9a-f]{32}")                               # secrets.token_hex(16)
_TBD = re.compile(r"\bTBD\b", re.I)


class FreezeError(RuntimeError):
    """The record is malformed, or a command may not run under it."""


@dataclass
class Freeze:
    salt: Optional[str]
    repos: List[str]
    knobs: Dict[str, float]
    loss: Dict[str, Optional[float]]
    cost_weights: List[float]
    m_S: Optional[float]
    ceilings: Optional[dict]
    nominal_costs: Optional[dict]
    producers: List[str]
    authors: Any
    oracle: Any
    stage: str
    signed: bool
    evidence_configs: List[str]
    excluded_tasks: List[str]
    min_tasks: int                 # eligible tasks across the repositories (180 analyzed plus 30 % attrition)
    min_tasks_per_repo: int        # eligible tasks per repository; corpus.build select and census check both
    tool_digests: Dict[str, str]
    created: Optional[str]
    freeze_id: Optional[str]
    path: Optional[Path] = field(default=None, compare=False)

    def config_paths(self) -> List[Path]:
        """The repository and evidence configs the record governs (relative to the working directory)."""
        return [Path(p) for p in (*self.repos, *self.evidence_configs)]


def load_freeze(path: Path) -> Freeze:
    """Read a record. Every field must be present with its type; keys starting with "_" are comments."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise FreezeError(f"freeze record {path} not found") from None
    except json.JSONDecodeError as e:
        raise FreezeError(f"{path} is not valid JSON ({e.msg}, line {e.lineno})") from None
    if not isinstance(raw, dict):
        raise FreezeError(f"{path}: a freeze record is a JSON object")
    missing = [k for k in FIELDS if k not in raw]
    unknown = sorted(k for k in raw if not k.startswith("_") and k not in FIELDS)
    problems = ([f"missing fields {', '.join(missing)}"] if missing else []) + \
               ([f"unknown fields {', '.join(unknown)}"] if unknown else [])
    if problems:
        raise FreezeError(f"{path}: " + "; ".join(problems))
    for k, t in _TYPES.items():
        if not isinstance(raw[k], t) or (t is int and isinstance(raw[k], bool)):
            raise FreezeError(f"{path}: {k} must be a JSON {t.__name__}, not {raw[k]!r}")
    for k, t in _NULLABLE.items():
        if raw[k] is not None and (not isinstance(raw[k], t) or isinstance(raw[k], bool)):
            raise FreezeError(f"{path}: {k} must be null or a {getattr(t, '__name__', 'number')}, not {raw[k]!r}")
    if raw["stage"] not in STAGES:
        raise FreezeError(f"{path}: unknown stage {raw['stage']!r} (stages: {', '.join(STAGES)})")
    return Freeze(**{k: raw[k] for k in FIELDS}, path=path)


# ---------------------------------------------------------------- stages and partitions

def _rank(stage: str) -> int:
    if stage not in STAGES:
        raise FreezeError(f"unknown stage {stage!r} (stages: {', '.join(STAGES)})")
    return STAGES.index(stage)


def _name(freeze: Freeze) -> str:
    return str(freeze.path or "the freeze record")


def is_dev(freeze: Optional[Freeze]) -> bool:
    """No record, or a record still at stage dev: a development run."""
    return freeze is None or freeze.stage == "dev"


def require_stage(freeze: Freeze, stage: str) -> None:
    """Raise FreezeError unless the record has reached `stage`."""
    if _rank(freeze.stage) < _rank(stage):
        raise FreezeError(f"{_name(freeze)} is at stage {freeze.stage}; this needs stage {stage} or later")


def require_open(freeze: Optional[Freeze], partitions: Iterable[Optional[str]]) -> None:
    """Refuse every partition the stage has not opened; None stands for all of them."""
    if is_dev(freeze):
        return
    opened = OPEN[freeze.stage]
    for p in partitions:
        for q in (PARTITIONS if p is None else (p,)):
            if q not in opened:
                every = " (no partition named means all of them)" if p is None else ""
                raise FreezeError(f"partition {q!r} is closed at stage {freeze.stage}{every}; "
                                  f"open: {', '.join(opened)}")


def require_dev_output(out: Path) -> None:
    """A development run writes only into a directory whose name ends in -dev."""
    if not Path(out).expanduser().resolve().name.endswith("-dev"):
        raise FreezeError(f"a development run writes only into a directory whose name ends in -dev, not {out}")


def require_listed(freeze: Freeze, configs: Iterable[str]) -> None:
    """A command under the record reads only configs the record lists, which `validate` checks."""
    listed = {Path(p).expanduser().resolve() for p in freeze.config_paths()}
    for c in configs:
        if Path(c).expanduser().resolve() not in listed:
            raise FreezeError(f"{c} is not listed in {_name(freeze)} (repos, evidence_configs)")


def require_fit_partitions(freeze: Optional[Freeze], fit_partition: Optional[str],
                           eval_partition: Optional[str]) -> None:
    """Fit on training only; evaluate on a named partition the stage has opened, so
    the test partition only once unblinded. Development runs may fit and evaluate anywhere."""
    if is_dev(freeze):
        return
    require_stage(freeze, "frozen")
    if fit_partition != "train":
        raise FreezeError(f"fit on {fit_partition!r} refused: under a freeze record the fit uses 'train' only")
    if eval_partition is None:
        raise FreezeError("evaluation on all matrices is a development run; name the evaluation partition")
    require_open(freeze, [eval_partition])


def check_manifest(freeze: Optional[Freeze], rows: Iterable[dict], where: object = "the manifest") -> None:
    """From stage frozen on, no manifest a command reads may hold a task the record excludes."""
    if freeze is None or _rank(freeze.stage) < _rank("frozen"):
        return
    hit = sorted({str(r.get("task")) for r in rows} & set(freeze.excluded_tasks))
    if hit:
        raise FreezeError(f"{where} holds tasks the freeze record excludes: {', '.join(hit)}")


def resolve_salt(freeze: Optional[Freeze], cfg: dict) -> Tuple[str, str]:
    """The salt behind every seeded choice (partition, seeded defect, evidence cell seed) and its
    source. A development run falls back to the config's freeze_salt; from pre-freeze on only the
    record's salt is used, and a record without one is refused."""
    if is_dev(freeze):
        if not cfg.get("freeze_salt"):
            raise FreezeError("development run: the config has no freeze_salt to fall back to")
        return str(cfg["freeze_salt"]), "repo-config (dev)"
    if not freeze.salt:
        raise FreezeError(f"{_name(freeze)} has no salt at stage {freeze.stage}; draw it with "
                          f"python -m rcca.freeze init")
    return freeze.salt, "freeze-record"


def frozen_loss(freeze: Freeze, given: Optional[dict] = None) -> dict:
    """Loss constants for a run under the record (Eq. loss). The record fixes kappa, lambda_R,
    lambda_E and s_tilde; tau_qual and eta are the record's once tuned, and at stage frozen a
    tuning run may supply them (`--loss`). Nothing else may differ from the record."""
    loss = dict(freeze.loss)
    for k, v in (given or {}).items():
        if k == "frozen" or k.startswith("_"):
            continue
        if k in TUNED_LOSS and freeze.stage == "frozen":
            loss[k] = v
        elif loss.get(k) != v:
            raise FreezeError(f"--loss sets {k} = {v!r}, but {_name(freeze)} fixes {loss.get(k)!r} "
                              f"at stage {freeze.stage}")
    untuned = [k for k in TUNED_LOSS if loss.get(k) is None]
    if untuned:
        raise FreezeError(f"{' and '.join('loss.' + k for k in untuned)} not tuned yet; a tuning run at "
                          f"stage frozen passes them with --loss")
    loss["frozen"] = all(loss[k] == freeze.loss.get(k) for k in TUNED_LOSS)
    return loss


def frozen_margin(freeze: Freeze, given: Optional[float], default: float = 0.05) -> float:
    """The noninferiority margin m_S: the record's once set. Before unblinding a run may use another
    value; once unblinded it may not."""
    if given is not None and freeze.m_S is not None and given != freeze.m_S and freeze.stage == "unblinded":
        raise FreezeError(f"--m-S {given} differs from the frozen m_S {freeze.m_S}")
    if given is not None:
        return given
    return default if freeze.m_S is None else freeze.m_S


def frozen_ceilings(freeze: Optional[Freeze]) -> Optional[Dict[str, Dict[str, float]]]:
    """The record's ceilings, which stage frozen fills from the training partition
    by pasting analysis/ceilings.json (Task B5); None while unset, and a run then computes its own
    from the fit partition. A malformed object is refused, and so is stage unblinded without them:
    the confirmatory run replays the frozen ceilings, never ones recomputed from data."""
    if freeze is None or freeze.ceilings is None:
        if freeze is not None and freeze.stage == "unblinded":
            raise FreezeError(f"{_name(freeze)} has no ceilings at stage unblinded; the confirmatory run "
                              f"replays the frozen ceilings, never recomputed ones")
        return None
    problems = ceilings_problems(freeze.ceilings)
    if problems:
        raise FreezeError(f"{_name(freeze)}: ceilings are malformed ({'; '.join(problems)})")
    return {q: {d: float(freeze.ceilings[q][d]) for d in CEILING_DIMS} for q in CEILING_QUANTILES}


def ceilings_problems(ceilings: Any) -> List[str]:
    """Why `ceilings` is not the frozen shape, the one rcca.analyze writes to analysis/ceilings.json:
    exactly q30, q60 and q90, each with exactly tokens, tool_calls and ci_equiv as positive
    numbers. Wall-clock time is an outcome, never a ceiling (results.md E2). [] means well formed."""
    if not isinstance(ceilings, dict):
        return [f"not an object of {', '.join(CEILING_QUANTILES)}"]
    out = []
    missing = [q for q in CEILING_QUANTILES if q not in ceilings]
    extra = sorted(str(q) for q in ceilings if q not in CEILING_QUANTILES)
    if missing:
        out.append(f"no {', '.join(missing)}")
    if extra:
        out.append(f"unexpected {', '.join(extra)}")
    for q in (q for q in CEILING_QUANTILES if q in ceilings):
        c = ceilings[q]
        if not isinstance(c, dict):
            out.append(f"{q} is not an object of {', '.join(CEILING_DIMS)}")
            continue
        lacks = [d for d in CEILING_DIMS if d not in c]
        more = sorted(str(d) for d in c if d not in CEILING_DIMS)
        if lacks:
            out.append(f"{q} lacks {', '.join(lacks)}")
        if more:
            out.append(f"{q} has {', '.join(more)}"
                       + (" (wall-clock is an outcome, not a ceiling)" if "wall_s" in more else ""))
        out += [f"{q}.{d} is {c[d]!r}, not a positive number" for d in CEILING_DIMS
                if d in c and not _positive(c[d])]
    return out


def frozen_nominal_costs(freeze: Optional[Freeze]) -> Optional[Dict[str, Dict[str, float]]]:
    """The record's nominal costs (Global Constraints: acquisition decisions use nominal costs estimated on
    the training partition), which stage frozen fills by pasting analysis/nominal_costs.json (Task B5);
    None while unset, and a run then computes them from the fit partition. A malformed table is refused,
    and so is stage unblinded without one: the confirmatory run prices actions as frozen."""
    if freeze is None or freeze.nominal_costs is None:
        if freeze is not None and freeze.stage == "unblinded":
            raise FreezeError(f"{_name(freeze)} has no nominal_costs at stage unblinded; the confirmatory run "
                              f"prices every action at its frozen nominal cost, never a recomputed one")
        return None
    problems = nominal_costs_problems(freeze.nominal_costs)
    if problems:
        raise FreezeError(f"{_name(freeze)}: nominal_costs are malformed ({'; '.join(problems)})")
    return {a: {d: float(c[d]) for d in COST_DIMS} for a, c in freeze.nominal_costs.items()}


def nominal_costs_problems(table: Any) -> List[str]:
    """Why `table` is not the frozen shape, the one rcca.analyze writes to analysis/nominal_costs.json: at
    least one action, each an object of exactly tokens, tool_calls, ci_equiv and wall_s as finite numbers
    of at least zero (a producer that calls no model costs zero tokens). [] means well formed."""
    if not isinstance(table, dict):
        return ["not an object of action ids"]
    if not table:
        return ["no actions"]
    out = []
    for aid, c in table.items():
        if not isinstance(c, dict):
            out.append(f"{aid} is not an object of {', '.join(COST_DIMS)}")
            continue
        lacks = [d for d in COST_DIMS if d not in c]
        more = sorted(str(d) for d in c if d not in COST_DIMS)
        if lacks:
            out.append(f"{aid} lacks {', '.join(lacks)}")
        if more:
            out.append(f"{aid} has {', '.join(more)}")
        out += [f"{aid}.{d} is {c[d]!r}, not a number of at least zero" for d in COST_DIMS
                if d in c and not _non_negative(c[d])]
    return out


def frozen_cost_weights(freeze: Optional[Freeze]) -> Optional[Tuple[float, ...]]:
    """The weights w_j of C_w(e) = sum_j w_j c_j(e) / B_j over the ceiling dimensions,
    which price every acquisition for RCCA and cheapest-first: a protocol constant the record fixes when
    it is written. None without a record, and a development run keeps equal weights. Malformed weights
    are refused."""
    if freeze is None:
        return None
    problems = cost_weights_problems(freeze.cost_weights)
    if problems:
        raise FreezeError(f"{_name(freeze)}: cost_weights are malformed ({'; '.join(problems)})")
    return tuple(float(w) for w in freeze.cost_weights)


def cost_weights_problems(weights: Any) -> List[str]:
    """Why `weights` are not w_j: one positive finite number per ceiling dimension (tokens, tool calls,
    CI-equivalents, in that order), summing to 1 within 1e-9. [] means well formed."""
    if not isinstance(weights, (list, tuple)):
        return [f"not a list of {len(CEILING_DIMS)} numbers"]
    out = []
    if len(weights) != len(CEILING_DIMS):
        out.append(f"{len(weights)} weights for the {len(CEILING_DIMS)} ceiling dimensions ({', '.join(CEILING_DIMS)})")
    bad = [w for w in weights if not _positive(w)]
    if bad:
        out.append(f"not positive numbers: {', '.join(map(repr, bad))}")
    elif weights and not abs(sum(weights) - 1.0) <= 1e-9:
        out.append(f"they sum to {sum(weights)!r}, not 1")
    return out


def _positive(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _non_negative(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


# ---------------------------------------------------------------- validation

def validate(freeze: Freeze, configs: Sequence[Path]) -> List[str]:
    """Every reason the record and `configs` (repository or evidence configs) are not ready for the
    record's stage, one line each; an empty list means valid. At stage dev the findings are reported
    and nothing refuses to run."""
    return _record_findings(freeze) + _pending(freeze) + _config_findings(freeze, configs)


def binding_findings(freeze: Freeze, configs: Sequence[Path]) -> List[str]:
    """The findings that refuse a command at the record's stage: none at dev; at frozen all but the
    fields that stage fills itself; all of them at pre-freeze and unblinded."""
    if is_dev(freeze):
        return []
    found = validate(freeze, configs)
    if freeze.stage == "frozen":
        pending = set(_pending(freeze))
        found = [f for f in found if f not in pending]
    return found


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _record_findings(fz: Freeze) -> List[str]:
    rec, at, out = _name(fz), _rank(fz.stage), []
    if at >= _rank("pre-freeze"):
        if not fz.salt:
            out.append(f"{rec}: salt is null at stage {fz.stage}; draw it once with python -m rcca.freeze init")
        elif not _SALT.fullmatch(fz.salt):
            out.append(f"{rec}: salt {fz.salt!r} was not drawn by init (32 hex digits, secrets.token_hex(16))")
        if not fz.repos or not fz.evidence_configs:
            out.append(f"{rec}: repos or evidence_configs is empty, so nothing is validated")
    tbd = sorted({s for s in _strings(fz.authors) if _TBD.search(s)})
    if tbd:
        out.append(f"{rec}: authors carries unfrozen values {', '.join(map(repr, tbd))}")
    malformed = ceilings_problems(fz.ceilings) if fz.ceilings is not None else []
    if malformed:                                  # unlike unset ceilings, never pending: it binds
        out.append(f"{rec}: ceilings are malformed ({'; '.join(malformed)})")
    malformed = nominal_costs_problems(fz.nominal_costs) if fz.nominal_costs is not None else []
    if malformed:                                  # likewise for nominal costs
        out.append(f"{rec}: nominal_costs are malformed ({'; '.join(malformed)})")
    malformed = cost_weights_problems(fz.cost_weights)
    if malformed:                                  # fixed when the record is written: never pending
        out.append(f"{rec}: cost_weights are malformed ({'; '.join(malformed)})")
    if at >= _rank("frozen"):
        if not fz.producers:
            out.append(f"{rec}: producers is empty")
        missing = [p for p in fz.producers if not fz.tool_digests.get(p)]
        if missing:
            out.append(f"{rec}: tool_digests missing for {', '.join(missing)}")
        out += [f"{rec}: loss.{k} is not set" for k in LOSS_CONSTANTS if fz.loss.get(k) is None]
        if not fz.authors:
            out.append(f"{rec}: authors are not recorded")
        if not fz.oracle:
            out.append(f"{rec}: oracle is not recorded")
    if fz.stage == "unblinded" and not fz.signed:
        out.append(f"{rec}: signed is false at stage unblinded")
    now = fingerprint(fz) if fz.freeze_id else None
    if now is not None and now != fz.freeze_id:
        out.append(f"{rec}: freeze_id {fz.freeze_id} no longer matches the record and its configs "
                   f"({now}); something sealed changed")
    return out


def _pending(fz: Freeze) -> List[str]:
    """Findings on the fields stage frozen fills itself (plan Task B5): the ceilings and nominal costs
    from the training partition, tau_qual, eta and m_S from the tuning partition. Ceilings or nominal
    costs that are set but malformed are a record finding instead, binding from pre-freeze on."""
    if _rank(fz.stage) < _rank("frozen"):
        return []
    rec, out = _name(fz), []
    if fz.m_S is None:
        out.append(f"{rec}: m_S is not set")
    if fz.ceilings is None:
        out.append(f"{rec}: ceilings are not set")
    if fz.nominal_costs is None:
        out.append(f"{rec}: nominal_costs are not set")
    out += [f"{rec}: loss.{k} is not set" for k in TUNED_LOSS if fz.loss.get(k) is None]
    return out


def _config_findings(fz: Freeze, configs: Sequence[Path]) -> List[str]:
    repos, evidence, out = [], [], []
    for p in configs:
        try:
            cfg = json.loads(Path(p).expanduser().read_text())
        except FileNotFoundError:
            out.append(f"{p}: config not found")
            continue
        except json.JSONDecodeError as e:
            out.append(f"{p}: not valid JSON ({e.msg})")
            continue
        if isinstance(cfg, dict) and "checkout" in cfg:
            repos.append((p, cfg))
        elif isinstance(cfg, dict) and ("python" in cfg or "f1" in cfg):
            evidence.append((p, cfg))
        else:
            out.append(f"{p}: neither a repository config nor an evidence config")
    f1 = [(p, _f1_selection(e)) for p, e in evidence if e.get("f1")]
    for p, cfg in repos:
        out += _repo_findings(fz, p, cfg, f1)
    return out


def _repo_findings(fz: Freeze, p: Path, cfg: dict, f1: List[Tuple[Path, Any]]) -> List[str]:
    out = []
    authors = cfg.get("authors") or []
    if not authors:
        out.append(f"{p}: lists no authors")
    for a in authors:
        bad = [f"{k} {a.get(k)!r}" for k in ("model", "version") if not a.get(k) or _TBD.search(str(a.get(k)))]
        if bad:
            out.append(f"{p}: author {a.get('mode', '?')} has an unfrozen {' and '.join(bad)}")
    if not cfg.get("oracle_cmd"):
        out.append(f"{p}: oracle_cmd is null; admission needs the hidden oracle, and standard CI is not one")
    if cfg.get("gold_patch_dev"):
        out.append(f"{p}: gold_patch_dev is true; gold-patch authoring is plumbing, not an agent-authored change")
    if cfg.get("history_fallback"):
        out.append(f"{p}: history_fallback is true; tasks from git history have unverified issue linkage")
    catalog = cfg.get("cwe_catalog")
    if not catalog:
        out.append(f"{p}: cwe_catalog is null; seeding would fall back to mutation only")
    elif not Path(catalog).expanduser().exists():
        out.append(f"{p}: cwe_catalog {catalog} not found")
    prompt = cfg.get("author_prompt")
    if not prompt:
        out.append(f"{p}: author_prompt is null; both authoring modes read one frozen prompt file")
    elif not Path(prompt).expanduser().exists():
        out.append(f"{p}: author_prompt {prompt} not found")
    for k, v in fz.knobs.items():
        if cfg.get(k) != v:
            out.append(f"{p}: {k} is {cfg.get(k)!r}, but the freeze record fixes {v!r}")
    hidden = hidden_oracle(cfg.get("oracle_cmd"))
    battery = pytest_selection(cfg.get("battery_cmd") or [])
    same = [str(ep) for ep, selection in f1 if battery is not None and battery == selection]
    if same and not hidden:
        out.append(f"{p}: battery_cmd runs the F1 suite of {same[0]}, which is standard CI, not a hidden battery")
    if hidden:
        battery_dir = _battery_dir(cfg)
        if not battery_dir.is_dir() or not any(battery_dir.glob("test_battery_*.py")):
            out.append(f"{p}: oracle_battery {battery_dir} holds no test_battery_*.py; the hidden oracle has no battery")
    return out


def hidden_oracle(cmd: Optional[Sequence[str]]) -> bool:
    """Whether an oracle_cmd is the hidden oracle's interface (corpus.oracle: it names {candidate} and
    {mode}), which runs the hidden battery as admission (a)'s second half; battery_cmd is then CI's
    half and may be F1's suite. Spelled out here so rcca imports nothing from corpus."""
    args = [str(a) for a in cmd or ()]
    return any("{candidate}" in a for a in args) and any("{mode}" in a for a in args)


def _battery_dir(cfg: dict) -> Path:
    return Path(cfg.get("oracle_battery") or f"oracle/{cfg.get('repo', '')}").expanduser()


def _battery_digest(battery_dir: Path) -> str:
    """sha256 over a battery's files and the shared conftest above it (corpus.oracle.battery_digest)."""
    files = [q for q in battery_dir.rglob("*") if q.is_file() and "__pycache__" not in q.parts] \
        if battery_dir.is_dir() else []
    shared = battery_dir.parent / "conftest.py"
    if not files:
        return "missing"
    h = hashlib.sha256()
    for q in sorted(files + ([shared] if shared.exists() else [])):
        h.update(str(q.relative_to(battery_dir.parent)).encode() + b"\0" + q.read_bytes() + b"\0")
    return h.hexdigest()


# pytest options whose value is the next argument, so it is not a test target
_VALUED = {"-p", "-k", "-m", "-c", "-o", "-W", "-n", "--deselect", "--ignore", "--ignore-glob", "--rootdir",
           "--confcutdir", "--basetemp", "--tb", "--durations", "--maxfail", "--junitxml", "--junit-xml",
           "--import-mode", "--log-level", "--capture", "--color", "--override-ini", "--config-file",
           "--cov", "--cov-report", "--timeout"}
# the options that change which tests run; the rest (-q, -x, -rfE, --no-header, -p ...) change only how
_SELECTING = {"--deselect": "deselect", "--ignore": "ignore", "--ignore-glob": "ignore", "-k": "keyword",
              "-m": "marker"}


def pytest_selection(argv: Sequence[str]) -> Optional[Tuple[Tuple[str, ...], ...]]:
    """What a pytest command line runs: test targets (normalized as paths, as pytest resolves
    them), deselected node-id prefixes, ignored paths, -k and -m expressions (literal, as pytest
    matches them), each sorted; None when `argv` does not run pytest. The interpreter, a coverage
    wrapper, and reporting flags do not count, so one suite under two flag sets is equal."""
    argv = [str(a) for a in argv]
    start = next((i + 2 for i in range(len(argv) - 1) if argv[i] == "-m" and argv[i + 1] == "pytest"), None)
    if start is None and argv and Path(argv[0]).name in ("pytest", "py.test"):
        start = 1
    if start is None:
        return None
    targets: List[str] = []
    chosen: Dict[str, List[str]] = {"deselect": [], "ignore": [], "keyword": [], "marker": []}
    i = start
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-"):
            name, eq, value = tok.partition("=")
            if not eq and name in _VALUED and i + 1 < len(argv):
                i += 1
                value = argv[i]
            if name in _SELECTING:
                chosen[_SELECTING[name]].append(value)
        else:
            targets.append(os.path.normpath(tok))
        i += 1
    return (tuple(sorted(targets)),) + tuple(tuple(sorted(v)) for v in chosen.values())


def _f1_selection(ecfg: dict):
    """The selection of the command F1 runs, built from the evidence config by F1's own builder."""
    from evidence.producers.f1_tests import ci_command     # here, so rcca imports nothing from evidence
    f1 = ecfg["f1"]
    return pytest_selection(ci_command(str(ecfg.get("python", "python")), f1.get("test_args", []),
                                       f1.get("source", ""), f1.get("extra_deselect", ())))


# ---------------------------------------------------------------- seal: the pre-registration snapshot

_UNSEALED = ("freeze_id", "stage", "signed", "m_S", "ceilings", "nominal_costs")


def _digest(p: Path) -> str:
    try:
        return hashlib.sha256(Path(p).expanduser().read_bytes()).hexdigest()
    except OSError:
        return "missing"


def fingerprint(freeze: Freeze) -> str:
    """sha256 over what the record pre-registers: its fields except those that change after sealing
    by design (stage, signed, freeze_id, and the m_S, ceilings, nominal_costs, loss.tau_qual and loss.eta
    that stage frozen fills), plus the bytes of every listed config and of the files they name whose
    content is protocol: the CWE catalogs, the authoring prompt and, for a config with the hidden
    oracle, its battery (oracle_battery and the shared oracle/conftest.py), which decides admission. A
    sealed record whose freeze_id no longer matches has drifted; `validate` reports it."""
    record = {k: getattr(freeze, k) for k in FIELDS if k not in _UNSEALED}
    record["loss"] = {k: v for k, v in freeze.loss.items() if k not in TUNED_LOSS}
    files = {}
    for p in freeze.config_paths():
        files[str(p)] = _digest(p)
        try:
            cfg = json.loads(Path(p).expanduser().read_text())
        except (OSError, ValueError):
            cfg = None
        for key in ("cwe_catalog", "author_prompt"):
            named = cfg.get(key) if isinstance(cfg, dict) else None
            if named:
                files[str(named)] = _digest(Path(named))
        if isinstance(cfg, dict) and hidden_oracle(cfg.get("oracle_cmd")):     # the battery decides admission
            files[str(_battery_dir(cfg))] = _battery_digest(_battery_dir(cfg))
    blob = json.dumps({"record": record, "files": files}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- the firewall a command passes

def gate(freeze_path: Optional[str], *, dev: bool, out: Path, min_stage: str,
         partitions: Iterable[Optional[str]] = (), configs: Sequence[str] = ()) -> Optional[Freeze]:
    """Run before a command writes. `freeze_path` is its --freeze, `dev` its --dev, `out` the directory
    it writes into, `min_stage` the earliest stage it may run at, `partitions` those it touches (None
    for all), `configs` the configs it reads. Returns the record, or None for a development run
    without one; raises FreezeError with a one-line reason when the command may not run."""
    if freeze_path is None:
        if not dev:
            raise FreezeError("no freeze record: pass --freeze <record>, or --dev for a development run "
                              "into a directory ending in -dev")
        require_dev_output(out)
        return None
    fz = load_freeze(Path(freeze_path))
    if fz.stage == "dev":
        require_dev_output(out)
        return fz
    if dev:
        raise FreezeError(f"--dev with {fz.path} at stage {fz.stage}: a development run takes no freeze record")
    require_stage(fz, min_stage)
    require_open(fz, partitions)
    require_listed(fz, configs)
    found = binding_findings(fz, fz.config_paths())
    if found:
        raise FreezeError(f"{fz.path} does not validate at stage {fz.stage}: " + "; ".join(found))
    return fz


# ---------------------------------------------------------------- record maintenance

def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def init_record(path: Path) -> str:
    """Draw the salt once (secrets.token_hex(16)), move the record from dev to pre-freeze, stamp
    `created`. Refuses a record that already has a salt: the salt never changes."""
    path = Path(path)
    fz = load_freeze(path)
    if fz.salt:
        raise FreezeError(f"{path} already has its salt; it is drawn once and never changes")
    if fz.stage != "dev":
        raise FreezeError(f"{path} is at stage {fz.stage}; init starts from a record at stage dev")
    raw = json.loads(path.read_text())
    raw.update(salt=secrets.token_hex(16), stage="pre-freeze",
               created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _write_json(path, raw)
    return raw["salt"]


FILLED = ("loss.tau_qual", "loss.eta", "m_S", "ceilings", "nominal_costs")   # what stage frozen fills


def fill_record(path: Path, fields: dict) -> List[str]:
    """Paste the tuned fields (rcca.tune's `freeze_fields`) into a record at stage frozen: loss.tau_qual,
    loss.eta, m_S, ceilings, nominal_costs. Each is written once; a value already there is refused, so a
    second tuning run cannot quietly replace the first. The stage does not move."""
    path = Path(path)
    fz = load_freeze(path)
    if fz.stage != "frozen":
        raise FreezeError(f"{path} is at stage {fz.stage}; the tuned fields are filled at stage frozen only")
    raw = json.loads(path.read_text())
    loss_in = fields.get("loss") or {}
    incoming = {"loss.tau_qual": loss_in.get("tau_qual"), "loss.eta": loss_in.get("eta"), "m_S": fields.get("m_S"),
                "ceilings": fields.get("ceilings"), "nominal_costs": fields.get("nominal_costs")}
    missing = [k for k, v in incoming.items() if v is None]
    if missing:
        raise FreezeError(f"the tuning result lacks {', '.join(missing)}")
    problems = ceilings_problems(incoming["ceilings"]) + nominal_costs_problems(incoming["nominal_costs"])
    if problems:
        raise FreezeError("; ".join(problems))
    current = {"loss.tau_qual": (raw.get("loss") or {}).get("tau_qual"), "loss.eta": (raw.get("loss") or {}).get("eta"),
               "m_S": raw.get("m_S"), "ceilings": raw.get("ceilings"), "nominal_costs": raw.get("nominal_costs")}
    taken = [k for k, v in current.items() if v is not None]
    if taken:
        raise FreezeError(f"{path} already holds {', '.join(taken)}; a filled field is never replaced")
    raw.setdefault("loss", {})
    raw["loss"]["tau_qual"], raw["loss"]["eta"] = incoming["loss.tau_qual"], incoming["loss.eta"]
    raw["m_S"], raw["ceilings"], raw["nominal_costs"] = incoming["m_S"], incoming["ceilings"], incoming["nominal_costs"]
    _write_json(path, raw)
    return list(FILLED)


def seal_record(path: Path) -> str:
    """Write freeze_id, the fingerprint of the record and its configs, so later drift is a finding."""
    path = Path(path)
    fz = load_freeze(path)
    if is_dev(fz):
        raise FreezeError(f"{path} is at stage dev: nothing is frozen to seal")
    raw = json.loads(path.read_text())
    raw["freeze_id"] = fingerprint(fz)
    _write_json(path, raw)
    return raw["freeze_id"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rcca.freeze", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (("validate", "report every finding; exit 1 if there is any"),
                       ("init", "draw the salt once and move the record from dev to pre-freeze"),
                       ("fill", "paste the tuned fields (rcca.tune's freeze_fields) into a record at stage frozen"),
                       ("seal", "pin freeze_id to the record and the configs it lists")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--freeze", default="configs/freeze.json")
        if name == "validate":
            p.add_argument("--config", action="append", default=[],
                           help="a further config to validate with the record's own")
        if name == "fill":
            p.add_argument("--from", dest="source", required=True, help="tuning.json written by rcca.tune")
    a = ap.parse_args(argv)
    try:
        if a.command == "init":
            init_record(Path(a.freeze))
            print(f"{a.freeze}: salt drawn, stage pre-freeze")
            return 0
        if a.command == "fill":
            src = json.loads(Path(a.source).read_text())
            filled = fill_record(Path(a.freeze), src.get("freeze_fields", src))
            print(f"{a.freeze}: filled {', '.join(filled)} from {a.source}")
            return 0
        if a.command == "seal":
            print(f"{a.freeze}: sealed, freeze_id {seal_record(Path(a.freeze))}")
            return 0
        fz = load_freeze(Path(a.freeze))
    except FreezeError as e:
        print(f"rcca.freeze: {e}", file=sys.stderr)
        return 2
    found = validate(fz, fz.config_paths() + [Path(c) for c in a.config])
    print(f"{a.freeze}: stage {fz.stage}, {len(found)} finding{'' if len(found) == 1 else 's'}")
    for f in found:
        print(f"  {f}")
    if fz.stage == "dev":
        print("stage dev reports findings; nothing refuses to run, and every output ends in -dev")
    elif fz.stage == "frozen" and _pending(fz):
        print("m_S, ceilings, nominal_costs, loss.tau_qual and loss.eta are filled at this stage; "
              "they bind from unblinded on")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
