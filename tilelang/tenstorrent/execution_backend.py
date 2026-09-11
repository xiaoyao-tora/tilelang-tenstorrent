"""Tenstorrent execution-backend compatibility declaration."""

from __future__ import annotations

import importlib.util

from tilelang.backend.execution_backend import ExecutionBackendSpec

from .contracts import (
    EXECUTION_BACKEND_ORDER,
    TTNN_ENABLE_DEVICE_COMPILE,
    TTNN_ENABLE_HOST_CODEGEN,
)


def _is_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def is_ttnn_available() -> bool:
    """Return whether both optional TT-Lang and TTNN Python packages exist."""

    return _is_module_available("ttl") and _is_module_available("ttnn")


EXECUTION_BACKENDS = (
    ExecutionBackendSpec(
        EXECUTION_BACKEND_ORDER[0],
        is_available=is_ttnn_available,
        enable_host_codegen=TTNN_ENABLE_HOST_CODEGEN,
        enable_device_compile=TTNN_ENABLE_DEVICE_COMPILE,
    ),
)
