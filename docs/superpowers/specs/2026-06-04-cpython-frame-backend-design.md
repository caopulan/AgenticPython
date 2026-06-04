# CPython Frame Backend Design

Date: 2026-06-04
Status: working design; user delegated design authority and requested execution

## Goal

AgenticPython should evolve from a pure Python editable tape prototype into a
general execution-control system that can observe and eventually intervene at
CPython frame safepoints. The final target is not "an LLM runs Python line by
line"; it is a deterministic runtime that advances Python execution through
observable safepoints, pauses on triggers, exposes enough state to an Agent, and
applies well-scoped mutations without putting LLM calls inside the interpreter
hot path.

The durable shape is:

- CPython fork or patchset: thin runtime probe and control hooks only.
- AgenticPython controller: scheduling, policy, TUI, Codex calls, patch
  validation, journaling, replay, and distributed coordination.
- Existing v1 tape runner: stays as the source-level editable backend for
  scripts that can be transformed safely.

## Core Conclusion

The desired system is achievable as a robust engineering project, but not as a
perfect universal "edit any arbitrary operation at any instant" machine.

CPython-level work can make Python frames, bytecode boundaries, exceptions,
future calls, selected locals, and package-internal Python code observable and
partly editable. It still cannot safely rewrite execution while the process is
inside native C/C++ extensions, CUDA kernels, blocking syscalls, or mismatched
distributed collectives. In those regions the correct operation is
observe/wait/timeout/checkpoint/kill/restart, not statement injection.

## Layered Architecture

### 1. Control Plane

The controller owns all non-hot-path behavior:

- receives runtime events from one or more Python processes;
- maintains a bounded context buffer that does not call Codex until a trigger;
- accepts TUI commands such as pause, resume, stop, and btw;
- calls Codex only for human instructions, runtime errors, or explicit policy
  triggers;
- validates every proposed mutation against a local schema and capability map;
- writes an append-only journal and enough metadata to reproduce decisions.

The controller is allowed to be complex. The CPython fork is not.

### 2. Runtime Probe

The CPython fork should begin as a minimal patch against CPython 3.12.x, chosen
to match the working local runtime family used by this project.

Probe responsibilities:

- emit frame events at gated safepoints: call, line, return, exception, and
  optionally opcode/basic-block later;
- support cooperative pause at safepoints;
- expose frame identifiers, code identity, filename, function, line number,
  selected locals/globals summaries, exception metadata, and rank/process
  metadata when available;
- accept control commands from the controller: resume, pause, stop, snapshot,
  mutate selected locals, replace future function code, and retry failed frame
  where semantics allow it.

Probe non-responsibilities:

- no Codex SDK;
- no long blocking network call from the eval loop;
- no policy decisions;
- no broad Python object pretty-printing in C;
- no attempt to interrupt native extensions in the middle of a C call.

### 3. Transport

Use a local Unix domain socket by default, with newline-delimited JSON messages.
The protocol is intentionally language-neutral so a patched CPython runtime,
pure Python helper process, or future Rust/C controller can all speak it.

Message properties:

- versioned: every message carries `protocol_version`;
- framed: one JSON object per line;
- bounded: runtime summaries have byte and item limits;
- directional: runtime emits events, controller emits commands;
- replayable: journal stores the exact event and command stream.

The first implementation can test this protocol in pure Python before the C
patch exists.

### 4. Backends

AgenticPython should have explicit backends instead of pretending there is one
universal mechanism.

- `tape` backend: current AST/source-level tape. Best for scripts and training
  loops that can be decomposed into editable statements.
- `frame` backend: CPython fork probe. Best for observing arbitrary Python code,
  package internals, and functions without rewriting source into a tape.
- `hybrid` backend: controller combines source-level patching for user scripts
  with frame-level observation for package internals.

This keeps v1 useful while the deeper runtime is built.

## Capability Map

### Strong Capabilities

- Pause and resume at Python safepoints.
- Observe current Python frame stack and bounded locals/globals summaries.
- Capture runtime errors with traceback, current frame, and recent event window.
- Patch future source-level tape units in the v1 backend.
- Wrap, replace, or monkey-patch future Python function calls when the function
  has not yet been entered.
- For package-internal pure Python code, observe call/line/exception events
  without modifying package source.

### Limited Capabilities

- Mutating locals in a currently executing optimized frame: possible only with
  narrow rules and validation; not all variable slots or closures are safe.
- Retrying a failed operation: robust when the failure is at a safepoint with a
  well-defined retry boundary; unsafe after irreversible side effects unless
  the journal marks the instruction as idempotent or checkpoint-backed.
- Replacing code for an already-entered frame: limited. Future calls are safer
  than rewriting the instruction stream of a live frame.
- Editing async/generators/with/finally/class bodies: possible only after
  backend-specific semantics are designed and tested.

### Hard Boundaries

- Cannot safely inject Python statements while execution is inside a native
  C/C++ extension call.
- Cannot preempt a CUDA kernel from CPython.
- Cannot make only one DDP/FSDP rank take a different collective path.
- Cannot guarantee rollback of external side effects such as file writes,
  socket sends, optimizer steps, or database commits unless explicit
  checkpoints or compensation hooks exist.

## Distributed And Training Runtime Model

For `torch.distributed`, the controller must treat a training job as a group of
ranked runtimes.

Rules:

- default pause is group pause: all ranks pause at the next compatible
  safepoint;
- mutations that affect collective topology must be broadcast and validated for
  every rank;
- read-only `/btw` can aggregate state from rank 0 plus sampled ranks;
- if one rank is inside native/CUDA/NCCL while another is at a Python safepoint,
  the controller waits, times out, or requests checkpoint-restart;
- rank-local debugging is allowed only for mutations that cannot alter
  distributed control flow.

The first native milestone does not implement DDP, but the protocol must carry
`process_id`, optional `rank`, and optional `world_size` from the beginning so
it does not need to be redesigned later.

## Repository Layout

The main repository should not vendor the CPython source tree.

Planned layout:

- `src/agenticpython/native/`: Python-side protocol, controller server, and
  frame backend integration.
- `tests/test_native_*.py`: deterministic tests for the protocol and controller.
- `runtimes/cpython/patches/`: small reviewable patch files against a pinned
  CPython tag.
- `tools/cpython_backend/`: fetch, patch, build, and smoke-test scripts.
- `.agentpython-build/`: ignored local build cache containing cloned CPython
  sources and build products.

This keeps the fork reproducible without polluting the repo with a full CPython
checkout.

## First Milestone

Build the narrowest native-backend slice that proves the architecture without
claiming hot-code replacement yet:

1. Add a versioned native protocol in pure Python.
2. Add tests for event decoding, command encoding, and bounded frame summaries.
3. Add build-cache ignore rules for `.agentpython-build/`.
4. Add a `frame-protocol-smoke` or equivalent diagnostic command that exercises
   protocol encoding locally.
5. Add CPython source-management tooling that can fetch a pinned CPython 3.12.x
   tag into `.agentpython-build/`.
6. Add the first CPython patch that emits call/line/return/exception events to
   the controller when `PYTHON_AGENTIC=1` is set.
7. Prove observation on a toy script and a toy imported package function.

Acceptance for this milestone:

- normal `agentpython run` and `agentpython tui` behavior is unchanged;
- protocol tests pass without Codex or CPython source checkout;
- build tooling does not vendor CPython into git;
- patched runtime can run a toy Python script and emit frame events;
- with the probe disabled, patched runtime behaves like ordinary Python for
  the smoke scripts.

## Design Guardrails

- The eval loop must never wait for the LLM.
- Every mutation must be schema-validated before reaching the runtime.
- Every event and command must be journaled.
- Every native/backend capability must be explicit; unsupported operations
  fail closed.
- The v1 tape backend remains available and testable while native work proceeds.
- CPython patches stay small enough to be reviewed line by line.
