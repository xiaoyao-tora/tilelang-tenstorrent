from __future__ import annotations

import pytest
import torch

from testing.python.tenstorrent.hardware.utils import from_device, to_device


@pytest.mark.timeout(60)
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_ttnn_tensor_round_trip(ttnn_module, tt_device, memory):
    source = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) / 128
    memory_config = {
        "dram": ttnn_module.DRAM_MEMORY_CONFIG,
        "l1": ttnn_module.L1_MEMORY_CONFIG,
    }[memory]
    device_tensor = to_device(
        ttnn_module,
        source,
        tt_device,
        memory_config=memory_config,
    )
    torch.testing.assert_close(from_device(ttnn_module, device_tensor), source)
