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

`tests/test_tune.py` does not collect: its line 10 imports `tests.test_analyze` and `tests/` has no
`__init__.py`. Secrets live in `.env` (gitignored).

## Sharing the corpus

`corpus/` is ~10 GB, of which 9.6 GB is regenerable and should not be copied: `envs/` (8.5 GB, built
by `corpus/envs.py` from pinned specs), `tools/` (a venv) and `repos/` (public clones at the base
commits in `tasks.jsonl`). What cannot be regenerated is the agent output, since the agents are
nondeterministic: about 59 MB, near 10 MB compressed. Note that publishing `oracle/` ends the
blinding of the hidden oracle for any later use of this corpus.
