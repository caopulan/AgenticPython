# Agentic CPython Runtime

AgenticPython keeps CPython source outside the repository. Fetch the pinned
source into `.agentpython-build/cpython`:

```bash
.venv/bin/python -m tools.cpython_backend.fetch_cpython
.venv/bin/python -m tools.cpython_backend.apply_patches
```

The first supported tag is `v3.12.13`.

Build and install the patched runtime:

```bash
cd .agentpython-build/cpython
./configure --prefix="$PWD/../install-agentic"
make -j4
make install
```

Run the package-internal frame smoke:

```bash
cd /Users/caopu/workspace/AgenticPython
rm -f /tmp/agentic-events.jsonl
PYTHON_AGENTIC=1 \
PYTHON_AGENTIC_RUN_ID=smoke \
PYTHON_AGENTIC_EVENTS=/tmp/agentic-events.jsonl \
.agentpython-build/install-agentic/bin/python3.12 examples/native_package_demo/run_demo.py
rg 'pkgdemo/inner.py.*compute' /tmp/agentic-events.jsonl
```

Expected stdout:

```text
result=8
```

Expected events include `call`, `line`, and `return` rows for
`examples/native_package_demo/pkgdemo/inner.py` function `compute`.

The patched runtime is enabled with `PYTHON_AGENTIC=1`.

Environment variables:

- `PYTHON_AGENTIC_RUN_ID` identifies the run in emitted events.
- `PYTHON_AGENTIC_EVENTS` is the JSONL frame-event output path.
- `PYTHON_AGENTIC_COMMANDS` is an append-only command file. The current command
  format is `exec_file\t/path/to/code.py`.
- `PYTHON_AGENTIC_PAUSE_FILE` pauses the interpreter at matching trace
  safepoints while the file exists.
- `PYTHON_AGENTIC_FILTER` limits tracing, command polling, and pause checks to
  frames whose filename contains the filter string. The native TUI sets this to
  the target script path so imported libraries such as PyTorch do not produce a
  frame event for every internal line.

Run the native MNIST TUI from the repository root:

```bash
.venv/bin/python -m pip install -e '.[mnist,codex]'
.venv/bin/agentpython native-tui examples/native_cpu_mnist.py --out-dir .agentpython-runs/native-mnist
```

Inside the TUI, `/exec <python code>` appends an `exec_file` command. CPython
executes that file in the current frame globals and locals at the next matching
trace safepoint. Natural-language input pauses the process, asks Codex SDK for
native Python code, queues it through the same command file, and waits for
`/resume`.

Each native TUI run writes:

- `journal.jsonl` for TUI-visible logs, user inputs, Codex request contexts,
  Codex returned code, stderr, and process lifecycle events.
- `frame_events.jsonl` for CPython frame events.
- `commands.jsonl` and `commands/*.py` for the command channel and exact
  injected code.
