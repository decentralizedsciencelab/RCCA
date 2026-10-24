"""Challenger, property-based (`ch.property`, family F1, hypothesis).

Inputs come from hypothesis strategies derived from the changed function's
annotations. The property is behavioural equivalence with the pre-change
version: `hypothesis.find` searches for an input on which base and candidate
diverge and shrinks it, so criterion (ii) is met by the technique itself.

Generation and shrinking run here, in the orchestrator, with its hypothesis;
only the calls run in the candidate's environment (`Pair`). That hypothesis is
pinned in the repository config's env.host_tools and in requirements-dev.txt,
evidence.run holds it to the pin under a freeze record, and every record
carries its version.
"""
from __future__ import annotations
import random
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import hypothesis
from hypothesis import HealthCheck, Phase, find, settings
from hypothesis.errors import NoSuchExample

from ..core import Action, Context, Meter, Outcome, Registry
from .common import ChallengerConfig, run_technique
from .harness import Divergence, Pair
from .strategies import for_target, runnable
from .targets import Target


class _Budget(Exception):
    pass


def make_property(cfg: ChallengerConfig, max_examples: int = 300) -> Action:
    def run(ctx: Context, m: Meter) -> Outcome:
        def search(targets: List[Target], pair: Pair, base_tree: Path, deadline: float) -> Iterator[Tuple[Divergence, bool]]:
            for i, t in enumerate(targets):
                if time.monotonic() > deadline:
                    return

                def diverges(rak, t=t) -> bool:
                    if time.monotonic() > deadline:
                        raise _Budget()
                    receiver, args, kwargs = rak
                    return pair.diverges(t, args, kwargs, receiver) is not None

                s = settings(max_examples=max_examples, database=None, deadline=None,
                             suppress_health_check=list(HealthCheck), phases=(Phase.generate, Phase.shrink))
                try:
                    rak = find(for_target(t), diverges, settings=s, random=random.Random(ctx.seed + i))
                except NoSuchExample:
                    continue
                except _Budget:
                    return
                except Exception:            # noqa: BLE001  strategy or target errors are not evidence
                    continue
                receiver, args, kwargs = rak
                d = pair.diverges(t, args, kwargs, receiver)
                if d is not None:
                    yield d, True

        return run_technique(ctx, m, cfg, "property-based", "pytest-witness", search,
                             {"tool": "hypothesis", "hypothesis_version": hypothesis.__version__,
                              "max_examples": str(max_examples)}, target_filter=runnable)

    return Action("ch.property", "F1", "challenger", "hypothesis", run, priority=60)


def register(reg: Registry, cfg: ChallengerConfig, **kw) -> Action:
    return reg.register(make_property(cfg, **kw))
