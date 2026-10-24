"""Challenger producers. Independently configured from the
authoring agent; they produce evidence within F1 and are not a fifth family.

    targets        changed public functions and public methods (instance, static,
                   class) with their signatures and constructor parameters
    harness        base worktree, paired executors, divergence oracle, ddmin,
                   repetition at sub-seeds, pytest witness (the second tool),
                   per-repository factory table for non-JSON parameter types
    strategies     hypothesis strategies from annotations, receivers for methods
    declared       the declared-change rule (candidate's own tests exercise the
                   changed callable -> divergence is fail_unvalidated, not a violation)
    common         the shared producer skeleton
    property       ch.property      hypothesis-generated inputs, shrunk by hypothesis
    differential   ch.differential  inputs recorded from the pinned state's test run
    fuzz           ch.fuzz          atheris, coverage-guided, replayed through the oracle

All three share one oracle: divergence from pre-change behaviour. A null
result is `pass` when the search ran to its budget and `inconclusive` when no
harness could be built, never a sign of safety.
"""
from .common import ChallengerConfig
