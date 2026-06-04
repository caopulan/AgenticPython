# AgenticPython

AgenticPython v1 is a prototype for an Agent-owned editable Python execution tape.

It runs supported Python scripts one editable instruction at a time. The local
controller advances automatically until an error or configured human trigger
fires. At that point it sends a compact context snapshot to Codex SDK, applies
the returned tape patch, and resumes without rerunning earlier instructions.

## Setup

Use Python 3.10 or newer. On this machine, the bundled Codex runtime Python
works:

```bash
/Users/caopu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test,codex]'
```

Codex SDK uses local Codex auth. If auth is missing, run `codex login` first.

## Commands

```bash
agentpython llm-smoke
agentpython run examples/error_rollback.py --out-dir .agentpython-runs/error
agentpython run examples/training_loop.py --triggers examples/training_trigger.json --out-dir .agentpython-runs/training
```

Each run writes:

- `journal.jsonl`
- `final_tape.json`
- `replay.py`

## v1 Scope

v1 supports module-level instructions and stepwise top-level `for` loops. It
rejects unsupported editable constructs instead of silently pretending to be a
complete Python VM.
