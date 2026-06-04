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
trace safepoint. Natural-language runtime controls such as `我要在 optim 里停`
are parsed locally into trace/break/resume commands. Other natural-language
input pauses the process, asks Codex SDK for native Python code, queues it
through the same command file, and waits for `/resume`.

Each native TUI run writes:

- `journal.jsonl` for TUI-visible logs, user inputs, Codex request contexts,
  Codex returned code, stderr, and process lifecycle events.
- `frame_events.jsonl` for CPython frame events.
- `commands.jsonl` and `commands/*.py` for the command channel and exact
  injected code.

The command channel supports dynamic trace control:

- `set_filters\t<filter1>\t<filter2>` replaces the filename-substring filters.
- `add_filter\t<filter>` adds one filename-substring filter.
- `clear_filters` traces all Python frames.
- `set_break_mode\toff|line|call|return|exception|all` controls automatic
  stopping at matching events.
- `set_step_mode\tinto|over|out|none` enables one-shot debugger-style stepping.
- `resume` releases an internal CPython break wait.

The TUI exposes these as slash commands:

```text
/trace script
/trace package torch.optim
/trace all
/break line
/step
/next
/out
/continue
```

The same controls can be requested in Chinese natural language:

```text
我要在 optim 里停
在 optim 用梯度更新参数的时候停一下
我想在 optimizer 每一行停住
在任意 python 代码每一行都停
optim 里不要停了
```

These controls operate at Python frame trace-event level. They can enter Python
files under packages such as `torch.optim`, but they cannot break inside
C++/ATen/CUDA/native execution.
