# RCCA — Risk-Calibrated Change Assurance

Can a gate that acquires evidence only as needed accept the same safe changes as plain CI while
letting through fewer unsafe ones, at a bounded cost? The study builds a pool of matched safe and
unsafe agent-authored changes over two repositories, runs eight evidence producers on each, and
replays nine acceptance policies under a pre-registered loss model.


## Layout

    corpus/     construction code, and the data it produces under corpus/<repo>/
    evidence/   the eight producers and the challenger
    rcca/       decision rule, replay, fit, analysis, the freeze record
    configs/    repository, evidence and catalog configs; three freeze records

Three sealed records, all validating clean: `freeze.json` `1be24755677d90ef` (confirmatory),
`freeze-tune.json` `ade4f7c444fb605c` and `freeze-qwen.json` `4e5cd2dc74b482f3` (exploratory, the
tune partition authored by claude, codex and a local Qwen).

## Running

    python -m corpus.build {select,author,pairs,labels} --config configs/flask.json --freeze configs/freeze.json
    python -m evidence.run --config configs/flask.json --evidence configs/evidence_flask.json --freeze configs/freeze.json

    python -m pytest -q --ignore=tests/test_tune.py      # 286 pass, 8 skip


