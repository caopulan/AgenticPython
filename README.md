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
agentpython tui examples/training_loop.py --out-dir .agentpython-runs/tui-demo
```

Each run writes:

- `journal.jsonl`
- `final_tape.json`
- `replay.py`

## v1 Scope

v1 supports module-level instructions and stepwise top-level `for` loops. It
rejects unsupported editable constructs instead of silently pretending to be a
complete Python VM.

## Interactive TUI

The TUI keeps program logs in the upper pane and an input prompt at the bottom.

```bash
agentpython tui examples/training_loop.py --out-dir .agentpython-runs/tui-demo
```

Commands:

- `/pause` pauses automatic execution.
- `/resume` or `/start` resumes.
- `/quit` exits and writes artifacts.
- Any other text is sent to Codex as an instruction. The runtime applies the
  returned patch and stays paused until `/resume`.
- Default TUI logging is `INFO`, which shows normal program output. Use
  `--log-level DEBUG` to also show each executed tape instruction.
- The TUI execution loop runs in a background thread, so the input prompt stays
  usable while the current instruction is running. Python and PyTorch execution
  is still cooperative: pause or instruction requests take effect after the
  current instruction, such as the current minibatch, finishes.
- Log rows are labeled and colorized by source: user instructions, program
  output, Codex triggers, patch summaries, agent state, and errors are visually
  separated. Patch code is summarized in the TUI; the full patch is still stored
  in `journal.jsonl`.

For a real CPU MNIST training script, install the optional dependencies and run:

```bash
.venv/bin/python -m pip install -e '.[mnist]'
agentpython tui examples/cpu_mnist.py --out-dir .agentpython-runs/mnist
```

`examples/cpu_mnist.py` trains a MNIST-adapted VGG16 on the full MNIST training
split with `batch_size = 64` for `num_epochs = 3` and evaluates the full test
split every `eval_every_epochs = 1`. The top-level tape advances one minibatch
at a time, reports periodic batch progress, and reports evaluation accuracy
after each epoch.

Example live instruction:

```text
下个 step 开始把 optimizer 的 lr 调成 0.01，并且在当前 instruction 后额外 evaluate 一次，把结果 append 到 evals。
```
