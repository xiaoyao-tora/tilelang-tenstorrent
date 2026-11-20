from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import pytest

from testing.python.tenstorrent.hardware.utils import open_device


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _hardware_available() -> bool:
    return bool(
        os.environ.get("TT_METAL_SIMULATOR")
        or os.environ.get("TTLANG_HAS_DEVICE") == "1"
        or glob.glob("/dev/tenstorrent/*")
        or glob.glob("/dev/tenstorrent[0-9]*")
    )


_DEPENDENCIES_AVAILABLE = _module_available("ttl") and _module_available("ttnn")
_HARDWARE_AVAILABLE = _hardware_available()
_HARDWARE_TEST_DIR = Path(__file__).resolve().parent


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "tenstorrent_hardware: executes a TileLang kernel on a Tenstorrent device",
    )
    config.addinivalue_line(
        "markers",
        "timeout(seconds): bounds a hardware test when pytest-timeout is installed",
    )


def pytest_collection_modifyitems(items):
    reason = None
    if not _DEPENDENCIES_AVAILABLE:
        reason = "requires the pinned TT-Lang and TTNN packages"
    elif not _HARDWARE_AVAILABLE:
        reason = "requires a Tenstorrent device or TT_METAL_SIMULATOR"
    skip = pytest.mark.skip(reason=reason) if reason else None
    for item in items:
        if not Path(str(item.path)).resolve().is_relative_to(_HARDWARE_TEST_DIR):
            continue
        item.add_marker(pytest.mark.tenstorrent_hardware)
        if skip is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def ttnn_module():
    if os.environ.get("TTLANG_COMPILE_ONLY"):
        pytest.fail("hardware tests require TTLANG_COMPILE_ONLY to be unset")
    try:
        import ttnn
    except Exception as error:
        pytest.fail(f"failed to import TTNN in a hardware environment: {error}")
    return ttnn


@pytest.fixture(scope="session")
def tt_arch(ttnn_module):
    override = os.environ.get("TILELANG_TT_ARCH")
    raw_arch = override or str(ttnn_module.get_arch_name())
    normalized = raw_arch.lower()
    if "wormhole" in normalized:
        return "wormhole_b0"
    if "blackhole" in normalized:
        return "blackhole"
    pytest.fail(f"unsupported Tenstorrent architecture: {raw_arch!r}")


@pytest.fixture(scope="session")
def tt_device(ttnn_module):
    with open_device(ttnn_module) as device:
        yield device
