"""Tenstorrent execution-backend compatibility declaration."""

from __future__ import annotations

import importlib.util

from tilelang.backend.execution_backend import ExecutionBackendSpec


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
        "ttnn",
        is_available=is_ttnn_available,
        enable_host_codegen=False,
        enable_device_compile=False,
    ),
)
