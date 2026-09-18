"""Frontend contract composition, evaluated as logical Device IR without hardware."""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tvm import ir

from testing.python.target.test_tilelang_tenstorrent_phase6_semantics import run_multicore_device


def communication_elementwise_program(collective):
    if collective:
        pipes = [T.comm.Pipe((0, 0), T.comm.CoreRange((1, 0), (3, 1)))]
    else:
        pipes = [T.comm.Pipe((0, 0), (1, 0)), T.comm.Pipe((0, 0), (2, 0))]
    net = T.comm.PipeNet(pipes)
    metadata = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}

    @T.prim_func
    def program(
        A: T.Tensor((64, 64), "bfloat16"),
        B: T.Tensor((32, 64), "bfloat16"),
        C: T.Tensor((128, 64), "bfloat16"),
    ):
        with T.Kernel(3, 1, threads=1) as (x, y):
            data = T.alloc_shared((64, 64), "bfloat16", annotations=metadata)
            bias = T.alloc_shared((32, 64), "bfloat16", annotations=metadata)
            result = T.alloc_shared((64, 64), "bfloat16", annotations=metadata)
            if T.comm.is_src(net):
                T.copy(A, data)
            for pipe in T.comm.foreach_src(net):
                T.copy(data, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, data)
                T.copy(B, bias)
                for i, j in T.Tiles(result):
                    result[i, j] = data[i, j] + bias[0, j]
                T.copy(result, C[(x - 1) * 64 : x * 64, :])

    return program


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("seed", [0, 19])
def test_communication_tiles_and_metadata_compose(arch, collective, seed):
    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    source = tvm.IRModule({"main": communication_elementwise_program(collective)})
    mod = TenstorrentPassPipelineBody(source, target)
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(mod, restored)
    ir.assert_structural_equal(mod, TenstorrentPassPipelineBody(mod, target))
    assert len(mod.attrs["tt.pipe_table"]) == (1 if collective else 2)
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 2
    assert all(int(d.block_count) == 2 for d in mod.attrs["tt.dfb_table"])
    assert all(str(d.element_dtype) == "bfloat16" for d in mod.attrs["tt.dfb_table"])

    # Small integers are exactly representable in BF16; nonuniform values
    # distinguish a true row broadcast from identity or scalar access.
    a = (np.arange(4096).reshape(64, 64) % 7).astype(np.float32)
    b = (np.arange(2048).reshape(32, 64) % 11).astype(np.float32)
    tensors, trace = run_multicore_device(restored, [a, b, np.zeros((128, 64), np.float32)], seed=seed)
    expected = np.concatenate([a + b[0], a + b[0]], axis=0)
    np.testing.assert_array_equal(tensors[-1], expected)
    assert len(trace["exports"]) == len(trace["deliveries"]) == 2
