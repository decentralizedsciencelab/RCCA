"""RCCA decision layer: evidence-matrix schema, decision loop, policies, statistics.

The evidence layer (docs/superpowers/specs/2026-09-22-evidence-layer-design.md)
produces one EvidenceMatrix per (candidate, repetition). This package replays a
matrix to the nine policies and computes the endpoints of
the noninferiority gate and the accepted-set bound. Nothing here calls a tool or a
model; everything is deterministic given the matrix and a seed.
"""
from .matrix import (Outcome, Cost, Action, Source, EvidenceRecord,
                     EvidenceMatrix, Ceilings, cell_seed)
from .decision import LossParams, EvidenceModel, State, Decision, run_policy, POLICIES
from .stats import (clopper_pearson_upper, cluster_bootstrap, noninferiority_gate,
                    holm, min_accepts_for_bound)

__all__ = [n for n in dir() if not n.startswith("_")]
