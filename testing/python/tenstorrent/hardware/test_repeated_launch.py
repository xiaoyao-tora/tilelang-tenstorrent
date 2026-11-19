from __future__ import annotations

import pytest
import torch

from testing.python.tenstorrent.hardware.kernels import make_add
from testing.python.tenstorrent.hardware.utils import assert_close, compile_kernel, from_device, to_device


@pytest.mark.timeout(180)
def test_repeated_launch_reuses_compiled_kernel_without_stale_results(ttnn_module, tt_device, tt_arch):
    kernel = compile_kernel(make_add("float32"), tt_arch)
    output = to_device(ttnn_module, torch.zeros((32, 32), dtype=torch.float32), tt_device)

    for iteration in range(4):
        lhs_host = torch.full((32, 32), float(iteration + 1), dtype=torch.float32)
        rhs_host = torch.full((32, 32), float(10 - iteration), dtype=torch.float32)
        lhs = to_device(ttnn_module, lhs_host, tt_device)
        rhs = to_device(ttnn_module, rhs_host, tt_device)
        result = kernel(lhs, rhs, output)
        assert result is output
        assert_close(
            from_device(ttnn_module, result),
            lhs_host + rhs_host,
            "float32",
        )
