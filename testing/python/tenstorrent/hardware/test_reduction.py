from __future__ import annotations

import pytest
import torch

from testing.python.tenstorrent.hardware.kernels import make_keepdims_reduction
from testing.python.tenstorrent.hardware.utils import (
    assert_close,
    compile_kernel,
    from_device,
    to_device,
    torch_dtype,
)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("reduction", ["sum", "max"])
def test_tilelang_keepdims_reduction(ttnn_module, tt_device, tt_arch, dtype, reduction):
    torch.manual_seed(17)
    torch_type = torch_dtype(dtype)
    source_host = (torch.rand((32, 32), dtype=torch.float32) - 0.5).to(torch_type)
    source = to_device(ttnn_module, source_host, tt_device)
    kernel = compile_kernel(make_keepdims_reduction(dtype, reduction), tt_arch)
    result = kernel(source)

    source_fp32 = source_host.float()
    if reduction == "sum":
        reduced = source_fp32.sum(dim=1, keepdim=True)
    else:
        reduced = source_fp32.max(dim=1, keepdim=True).values
    expected = reduced.expand(32, 32).to(torch_type)
    assert_close(
        from_device(ttnn_module, result),
        expected,
        dtype,
        reduction=True,
    )
