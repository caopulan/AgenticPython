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

The first probe writes JSONL frame events when `PYTHON_AGENTIC=1` is set. It
does not call Codex or block on the controller from inside CPython.
