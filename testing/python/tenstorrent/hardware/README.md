# TileLang Tenstorrent hardware tests

These tests execute the public TileLang TTNN path on a Tenstorrent device:

```text
TileLang frontend -> Device Lower -> TTL Codegen -> TT-Lang -> ttnn.generic_op
```

They are intentionally separate from frontend, Lower, Device IR and compile-only
tests. A missing device or missing pinned TT-Lang/TTNN environment skips this
directory rather than turning a hardware test into compile-only coverage.

## Environment

Use the pinned local environment and ensure TT-Metal device nodes and hugepages
are available:

```bash
source /home/xiaoyao/.venvs/ttlang-4ca4b89/bin/activate
unset TTLANG_COMPILE_ONLY
```

Optional settings:

```bash
export TT_DEVICE_ID=0
export TILELANG_TT_ARCH=wormhole_b0  # otherwise detected through TTNN
```

Install `pytest-timeout` on a hardware host so the timeout markers can interrupt
device deadlocks.

## Run

Run the default single-Core correctness and repeated-launch suite:

```bash
python -m pytest -v testing/python/tenstorrent/hardware \
  --timeout-method=thread
```

The multi-Core SUMMA test runs in a subprocess because compiler or device faults
must not terminate the main pytest process. It is experimental and currently
disabled by default. With the current pinned TT-Lang, its complete EmitC path
can terminate on a native block-argument assertion; the subprocess converts
that termination into an ordinary test failure while preserving the main test
session:

```bash
TILELANG_TT_RUN_EXPERIMENTAL_SUMMA=1 \
python -m pytest -v \
  testing/python/tenstorrent/hardware/test_multicore_summa.py \
  --timeout-method=thread
```

The first hardware gate covers launch, numerical correctness and deadlock only.
It does not impose performance thresholds and does not cover required full-FP32
matmul or dynamic DFB family selection.
