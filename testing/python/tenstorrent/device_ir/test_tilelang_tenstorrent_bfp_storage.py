"""BFP physical storage verification without hardware or numerical conversion."""

import inspect
import math

import pytest

from tilelang.tenstorrent import device_ir as D, language as T, transform
from tvm import ir

from testing.python.tenstorrent.device_ir.test_tilelang_tenstorrent_phase5_resources import rejected
from testing.python.tenstorrent.semantics.test_tilelang_tenstorrent_phase4_semantics import lower


@T.prim_func
def program(A: T.Tensor((64, 64), "bfloat16"), C: T.Tensor((64, 64), "bfloat16")):
    with T.Kernel(1, 1, threads=1):
        a = T.alloc_shared(
            (64, 64), "bfloat16", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}
        )
        c = T.alloc_shared(
            (64, 64), "bfloat16", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}
        )
        T.copy(A, a)
        for i, j in T.Tiles(c):
            c[i, j] = a[i, j] + a[i, j]
        T.copy(c, C)


def replace(descriptor, **changes):
    cls = type(descriptor)
    fields = {
        name: getattr(descriptor, name)
        for name in inspect.signature(cls.__init__).parameters
        if name != "self"
    }
    return cls(**(fields | changes))


def formatted(format):
    mod = lower(program)
    for attr in ("tt.tensor_table", "tt.dfb_storage_groups", "tt.dfb_table"):
        mod = mod.with_attr(attr, [replace(item, storage_format=format) for item in mod.attrs[attr]])
    return mod


@pytest.mark.parametrize("format,tile_bytes", [("", 2048), ("bfp_bf8", 1088), ("bfp_bf4", 576)])
def test_exact_physical_capacity(format, tile_bytes):
    mod = formatted(format)
    payload = sum(
        int(group.block_count) * math.prod(int(n) for n in group.block_shape_in_tiles) * tile_bytes
        for group in mod.attrs["tt.dfb_storage_groups"]
    )
    mod = mod.with_attr("tt.l1_capacity_bytes", payload)
    before = ir.save_json(mod)
    transform.VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before
    restored = ir.load_json(before)
    transform.VerifyTenstorrentDeviceIR()(restored)
    rejected(mod.with_attr("tt.l1_capacity_bytes", payload - 1), "L1 logical payload lower bound.*exceeds")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"storage_format": "bfp_bad"}, "unsupported storage_format"),
        ({"dtype": "float32"}, "requires logical bfloat16"),
        ({"tile_shape": [16, 32]}, "requires a 32x32 tile"),
    ],
)
def test_invalid_tensor_format(changes, match):
    mod = formatted("bfp_bf8")
    tensors = list(mod.attrs["tt.tensor_table"])
    tensors[0] = replace(tensors[0], **changes)
    rejected(mod.with_attr("tt.tensor_table", tensors), match)


def test_raw_tensor_transfer_requires_same_format():
    mod = formatted("bfp_bf8")
    tensors = [replace(item, storage_format="bfp_bf4") for item in mod.attrs["tt.tensor_table"]]
    rejected(mod.with_attr("tt.tensor_table", tensors), "storage_format mismatch")


def test_compute_does_not_require_same_physical_format():
    mod = formatted("bfp_bf8")
    # The input load and output store must agree with their own Tensor, not
    # with one another: compute operates on logical bfloat16 values.
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    outputs = {int(group.storage_group_id) for group in groups if str(group.consumer_slot) == "ncrisc"}
    assert outputs
    mod = mod.with_attr(
        "tt.dfb_storage_groups",
        [replace(group, storage_format="bfp_bf4") if int(group.storage_group_id) in outputs else group for group in groups],
    )
    mod = mod.with_attr(
        "tt.dfb_table",
        [replace(dfb, storage_format="bfp_bf4") if int(dfb.dfb_id) in outputs else dfb for dfb in mod.attrs["tt.dfb_table"]],
    )
    tensors = list(mod.attrs["tt.tensor_table"])
    tensors[-1] = replace(tensors[-1], storage_format="bfp_bf4")
    mod = mod.with_attr("tt.tensor_table", tensors)
    transform.VerifyTenstorrentDeviceIR()(mod)


def test_expanded_dfb_format_must_match_group():
    mod = formatted("bfp_bf8")
    dfbs = list(mod.attrs["tt.dfb_table"])
    dfbs[0] = replace(dfbs[0], storage_format="bfp_bf4")
    rejected(mod.with_attr("tt.dfb_table", dfbs), "storage group and DFB storage_format mismatch")


@pytest.mark.parametrize("format,tile_bytes", [("bfp_bf8", 1088), ("bfp_bf4", 576)])
def test_complete_storage_backing_capacity(format, tile_bytes):
    mod = formatted(format)
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    group = groups[0]
    size = int(group.block_count) * math.prod(int(n) for n in group.block_shape_in_tiles) * tile_bytes
    tensor = mod.attrs["tt.tensor_table"][0]
    backing = D.TensorStorageBacking(int(tensor.global_arg_index), 0, size)
    groups[0] = replace(group, storage_backing=backing)
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.dfb_storage_groups", groups))
    groups[0] = replace(
        group, storage_backing=D.TensorStorageBacking(int(tensor.global_arg_index), 0, size + 1)
    )
    rejected(mod.with_attr("tt.dfb_storage_groups", groups), "must equal the complete DFB capacity")
    groups[0] = replace(group, storage_backing=backing)
    tensors = list(mod.attrs["tt.tensor_table"])
    tensors[0] = replace(tensor, storage_format="")
    rejected(
        mod.with_attr("tt.dfb_storage_groups", groups).with_attr("tt.tensor_table", tensors),
        "storage backing Tensor storage_format mismatch",
    )