from __future__ import annotations

from contextlib import contextmanager
import os

import tilelang
import torch


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def torch_dtype(dtype: str):
    return _TORCH_DTYPES[dtype]


def ttnn_dtype(ttnn, dtype: str):
    return {
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
    }[dtype]


@contextmanager
def open_device(ttnn):
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    if os.environ.get("TT_METAL_SIMULATOR"):
        device = ttnn.open_device(device_id=device_id)
    else:
        dispatch = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
        device = ttnn.open_device(device_id=device_id, dispatch_core_config=dispatch)
    try:
        yield device
    finally:
        synchronize = getattr(ttnn, "synchronize_device", None)
        if callable(synchronize):
            synchronize(device)
        ttnn.close_device(device)


def to_device(ttnn, tensor: torch.Tensor, device, *, memory_config=None):
    dtype_name = "bfloat16" if tensor.dtype == torch.bfloat16 else "float32"
    return ttnn.from_torch(
        tensor,
        dtype=ttnn_dtype(ttnn, dtype_name),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def from_device(ttnn, tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor).cpu()


def compile_kernel(program, arch: str):
    return tilelang.compile(
        program,
        target={"kind": "tenstorrent", "arch": arch},
        target_host="c",
        execution_backend="ttnn",
    )


def assert_close(actual, expected, dtype: str, *, reduction: bool = False):
    if dtype == "bfloat16":
        rtol, atol = (0.03, 2.0) if reduction else (0.02, 0.08)
    else:
        rtol, atol = (0.005, 0.05) if reduction else (0.001, 0.005)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=rtol,
        atol=atol,
    )
