from __future__ import annotations

import pytest
import torch

from testing.python.tenstorrent.hardware.kernels import make_elementwise_chain
from testing.python.tenstorrent.hardware.utils import assert_close, compile_kernel, from_device, to_device


@pytest.mark.timeout(120)
def test_tilelang_elementwise_chain(ttnn_module, tt_device, tt_arch):
    torch.manual_seed(23)
    lhs_host = torch.randn((32, 32), dtype=torch.float32)
    rhs_host = torch.rand((32, 32), dtype=torch.float32) + 0.25
    lhs = to_device(ttnn_module, lhs_host, tt_device)
    rhs = to_device(ttnn_module, rhs_host, tt_device)
    kernel = compile_kernel(make_elementwise_chain(), tt_arch)
    result = kernel(lhs, rhs)

    expected = torch.sqrt(lhs_host * lhs_host + rhs_host) / 2
    assert_close(from_device(ttnn_module, result), expected, "float32")
