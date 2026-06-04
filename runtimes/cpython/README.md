# Agentic CPython Runtime

AgenticPython keeps CPython source outside the repository. Fetch the pinned
source into `.agentpython-build/cpython`:

```bash
.venv/bin/python -m tools.cpython_backend.fetch_cpython
.venv/bin/python -m tools.cpython_backend.apply_patches
```

The first supported tag is `v3.12.13`.
