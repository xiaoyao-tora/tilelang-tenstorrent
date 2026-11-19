from __future__ import annotations

import pytest
import torch

from testing.python.tenstorrent.hardware.kernels import make_add
from testing.python.tenstorrent.hardware.utils import (
    assert_close,
    compile_kernel,
    from_device,
    to_device,
    torch_dtype,
)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("preallocate_output", [False, True], ids=["auto-output", "preallocated-output"])
def test_tilelang_add(ttnn_module, tt_device, tt_arch, dtype, preallocate_output):
    torch.manual_seed(11)
    torch_type = torch_dtype(dtype)
    lhs_host = torch.randn((32, 32), dtype=torch_type)
    rhs_host = torch.randn((32, 32), dtype=torch_type)
    lhs = to_device(ttnn_module, lhs_host, tt_device)
    rhs = to_device(ttnn_module, rhs_host, tt_device)
    kernel = compile_kernel(make_add(dtype), tt_arch)

    if preallocate_output:
        output = to_device(ttnn_module, torch.zeros_like(lhs_host), tt_device)
        result = kernel(lhs, rhs, output)
        assert result is output
    else:
        result = kernel(lhs, rhs)

    expected = (lhs_host.float() + rhs_host.float()).to(torch_type)
    assert_close(from_device(ttnn_module, result), expected, dtype)
