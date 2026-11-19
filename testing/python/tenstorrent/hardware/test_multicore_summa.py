from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.timeout(360)
def test_experimental_multicore_summa(tt_arch):
    if os.environ.get("TILELANG_TT_RUN_EXPERIMENTAL_SUMMA") != "1":
        pytest.skip("set TILELANG_TT_RUN_EXPERIMENTAL_SUMMA=1 to run the experimental SUMMA test")
    completed = subprocess.run(
        [sys.executable, __file__, "--run", tt_arch],
        check=False,
        capture_output=True,
        text=True,
        timeout=330,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, f"SUMMA subprocess failed\nstdout:\n{completed.stdout[-8000:]}\nstderr:\n{completed.stderr[-8000:]}"


def _run(arch: str) -> None:
    import torch
    import ttnn

    from examples.tenstorrent.summa import make_summa
    from testing.python.tenstorrent.hardware.utils import (
        assert_close,
        compile_kernel,
        from_device,
        open_device,
        to_device,
    )

    program = make_summa(
        core_m=2,
        core_n=2,
        k_stages=2,
        block_m=32,
        block_n=32,
        block_k=32,
        input_dtype="bfloat16",
        accum_dtype="bfloat16",
        output_dtype="bfloat16",
    )
    torch.manual_seed(29)
    lhs_host = (torch.rand((64, 64), dtype=torch.float32) - 0.5).to(torch.bfloat16)
    rhs_host = (torch.rand((64, 64), dtype=torch.float32) - 0.5).to(torch.bfloat16)
    with open_device(ttnn) as device:
        lhs = to_device(ttnn, lhs_host, device)
        rhs = to_device(ttnn, rhs_host, device)
        kernel = compile_kernel(program, arch)
        result = kernel(lhs, rhs)
        expected = (lhs_host.float() @ rhs_host.float()).to(torch.bfloat16)
        assert_close(from_device(ttnn, result), expected, "bfloat16", reduction=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--run":
        raise SystemExit("usage: test_multicore_summa.py --run <arch>")
    _run(sys.argv[2])
