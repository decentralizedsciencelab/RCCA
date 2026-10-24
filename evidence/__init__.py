"""Evidence layer (spec: docs/superpowers/specs/2026-09-22-evidence-layer-design.md).

Produces, per (candidate, repetition, action), one record with an outcome in
{pass, fail_validated, fail_unvalidated, inconclusive}, a validation block,
a metered cost vector, and provenance. Never decides anything.

    core        Producer protocol, Meter, Record, Registry, produce()
    baseline    standard-CI baseline per repository (ci_equiv denominator)
    producers/  one module per producer; f1_tests is the repository's own suite
    run         matrix generation CLI over a corpus manifest
"""
