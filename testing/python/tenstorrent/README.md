# Tenstorrent tests

Tenstorrent-specific tests are grouped by compiler and execution boundary:

- `frontend/`: DSL construction, TIRX capture, allocation metadata, topology,
  transfers, and collectives.
- `lower/`: Tenstorrent pass-pipeline, structured fallback, shared-grid
  formation, compute, accumulator, and composition tests.
- `device_ir/`: Device IR schema, verification, compaction, resource, and
  lifetime tests.
- `semantics/`: CPU/NumPy logical reference models. These are not hardware
  execution tests.
- `backend/`: target registration, capabilities, and lazy backend imports.
- `codegen/`: TTL emission and optional TT-Lang boundary tests.
- `legacy/`: explicitly skipped tests for superseded Device IR and PipeNet
  contracts. A skipped legacy test is migration debt, not active coverage.
- `_support/`: reusable non-test modules. It currently contains the shared
  Device IR traversal and NumPy reference models. Modules in this directory
  must not use the `test_` prefix.

Run the dedicated suite from the repository root:

```bash
python -m pytest -q testing/python/tenstorrent
```

Changes to native Tenstorrent passes must be rebuilt before running the suite.
Passing frontend, Lower, Device IR, or CPU reference tests does not establish
TTL compilation, TTNN execution, device numerical correctness, or performance.
