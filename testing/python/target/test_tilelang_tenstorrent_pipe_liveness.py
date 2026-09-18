"""Transfer IDs must never stand in for payload IDs during dead-write removal."""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir, transform
from tvm import ir

from testing.python.target.test_tilelang_tenstorrent_compute_values import TARGET, _calls
from testing.python.target.test_tilelang_tenstorrent_lower_composition import interpret


def multicast_fragment_program(epochs):
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), T.comm.CoreRange(begin=(1, 0), end=(4, 1)))])

    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((96, 32), "float32")):
        with T.Kernel(4, 1, threads=1) as (x, y):
            panel = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            result = T.alloc_fragment((32, 32), "float32")
            for _epoch in T.serial(epochs):
                if T.comm.is_src(net):
                    T.copy(A, panel)
                for pipe in T.comm.foreach_src(net):
                    T.copy(panel, pipe)
                for pipe in T.comm.foreach_dst(net):
                    T.copy(pipe, panel)
                if T.comm.is_dst(net):
                    for i, j in T.Tiles(result):
                        result[i, j] = panel[i, j] + T.float32(1)
                    T.copy(result, C[(x - 1) * 32 : x * 32, :])

    return main


@pytest.mark.parametrize("epochs", [1, 2, 5])
def test_multicast_value_liveness_tracks_dfb_ids_not_transfer_ids(epochs):
    device = lower_tenstorrent_ir(tvm.IRModule({"main": multicast_fragment_program(epochs)}), TARGET)
    assert int(device.attrs["tt.device_ir_version"]) == 7
    records = list(device.attrs["tt.pipe_transfer_table"])
    assert len(records) == 3 * epochs
    operations = {name: [] for name in ("tl.tt.dfb_pipe_send", "tl.tt.dfb_pipe_recv", "tl.tt.dfb_pipe_wait")}
    for func in device.functions.values():
        for call in _calls(func.body):
            if call.op.name in operations:
                operations[call.op.name].append(tuple(map(int, call.args)))
    assert len(operations["tl.tt.dfb_pipe_send"]) == len(records)
    assert len(operations["tl.tt.dfb_pipe_recv"]) == len(records)
    assert len(operations["tl.tt.dfb_pipe_wait"]) == 2 * len(records)
    assert any(transfer != payload for transfer, payload, _ in operations["tl.tt.dfb_pipe_recv"])
    # Core 3 has two local DFB generations per epoch, while the last transfer
    # ID is 3 * epochs - 1. Filtering it as a local payload ID drops a live
    # receive even in the one-epoch case.
    last_core_dfbs = [dfb for dfb in device.attrs["tt.dfb_table"] if int(dfb.producer_domain.begin.x) == 3]
    assert len(last_core_dfbs) == 2 * epochs
    assert max(transfer for transfer, _, _ in operations["tl.tt.dfb_pipe_recv"]) >= len(last_core_dfbs)
    for transfer, payload, count in operations["tl.tt.dfb_pipe_recv"]:
        assert (transfer, payload, count) in operations["tl.tt.dfb_pipe_wait"]
    transform.VerifyTenstorrentDeviceIR()(device)
    ir.assert_structural_equal(ir.load_json(ir.save_json(device)), device)
    source = np.arange(32 * 32, dtype="float32").reshape(32, 32)
    outputs, _ = interpret(device, [source, np.full((96, 32), -1, dtype="float32")], seed=71)
    np.testing.assert_array_equal(outputs[1], np.tile(source + 1, (3, 1)))
