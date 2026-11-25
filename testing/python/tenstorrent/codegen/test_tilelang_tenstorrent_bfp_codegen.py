"""BFP storage uses explicit casts around logical BF16 computation in real TTL."""

import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tilelang import tvm
from tvm.target import Target

from tilelang.tenstorrent import codegen, lower_tenstorrent_ir
from tilelang.tenstorrent.ttl_codegen.artifact import TenstorrentTTLModuleNode, build_module_metadata
from tilelang.tenstorrent.ttl_codegen.bindings import TTLANG_REVISION, load_bindings
from tilelang.tenstorrent.ttl_codegen.function import TTLFunctionEmitter
from testing.python.tenstorrent.frontend.test_tilelang_tenstorrent_storage_format import program


TARGET = Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
BF16_BLOCK = "tensor<2x2x!ttcore.tile<32x32, bf16>>"


@pytest.fixture(
    params=[
        ("bfp_bf8", "bfp_bf8"),
        ("bfp_bf4", "bfp_bf4"),
        ("bfp_bf8", "bfp_bf4"),
        ("bfp_bf4", "bfp_bf8"),
    ],
    ids=["bf8", "bf4", "bf8-to-bf4", "bf4-to-bf8"],
)
def formats(request):
    return request.param


@pytest.fixture
def device(formats):
    return lower_tenstorrent_ir(tvm.IRModule({"main": program(*formats)}), TARGET)


@pytest.fixture
def artifact(device):
    # Only an absent optional compiler may skip. Broken imports and revision
    # mismatches from an installed compiler must fail, not become skips.
    if importlib.util.find_spec("ttl") is None:
        pytest.skip(f"requires pinned TT-Lang compiler {TTLANG_REVISION}")
    load_bindings()
    result = codegen.build_ttl_without_compile(device, TARGET)
    assert isinstance(result, TenstorrentTTLModuleNode)
    assert ("tt-lang", TTLANG_REVISION) in result.metadata.dependency_revisions
    return result


def _storage_block(storage_format):
    return f"tensor<2x2x!ttcore.tile<32x32, {storage_format}>>"


@pytest.mark.parametrize("storage_format", ["bfp_bf8", "bfp_bf4"])
@pytest.mark.parametrize("assembled_lhs", [False, True], ids=["physical-lhs", "assembled-logical-lhs"])
def test_collective_reduce_bfp_operands_use_logical_bf16(storage_format, assembled_lhs):
    physical_lhs = SimpleNamespace(type=_storage_block(storage_format), name="physical-lhs")
    physical_rhs = SimpleNamespace(type=_storage_block(storage_format), name="physical-rhs")
    logical_lhs = SimpleNamespace(type=BF16_BLOCK, name="logical-lhs")
    logical_rhs = SimpleNamespace(type=BF16_BLOCK, name="logical-rhs")
    result = SimpleNamespace(type=BF16_BLOCK, name="sum")
    ttl = SimpleNamespace(
        typecast=Mock(side_effect=[logical_rhs] if assembled_lhs else [logical_lhs, logical_rhs]),
        add=Mock(return_value=result),
        max=Mock(),
        min=Mock(),
    )
    emitter = SimpleNamespace(
        dfb_id=lambda operand: operand,
        groups={0: SimpleNamespace(storage_format=storage_format), 1: SimpleNamespace(storage_format=storage_format)},
        values={0: physical_lhs, 1: physical_rhs},
        assembled={0: logical_lhs} if assembled_lhs else {},
        bindings=SimpleNamespace(ttl=ttl),
        type_converter=SimpleNamespace(logical_block_type=Mock(return_value=BF16_BLOCK)),
    )

    TTLFunctionEmitter.emit_collective_reduce(emitter, SimpleNamespace(args=[0, 1, 0, "sum"]))

    expected_casts = [] if assembled_lhs else [call(BF16_BLOCK, physical_lhs)]
    assert ttl.typecast.call_args_list == expected_casts + [call(BF16_BLOCK, physical_rhs)]
    ttl.add.assert_called_once_with(logical_lhs, logical_rhs)
    assert emitter.assembled[0] is result
    assert emitter.values[0] is physical_lhs
    assert emitter.values[1] is physical_rhs


@pytest.mark.parametrize(
    "storage_format,value_type,needs_cast",
    [
        ("bfp_bf8", BF16_BLOCK, True),
        ("bfp_bf4", BF16_BLOCK, True),
        ("bfp_bf8", _storage_block("bfp_bf8"), False),
        ("bfp_bf4", _storage_block("bfp_bf4"), False),
        ("", "tensor<2x2x!ttcore.tile<32x32, f32>>", False),
    ],
    ids=["pack-bf8", "pack-bf4", "already-bf8", "already-bf4", "non-bfp-type-mismatch"],
)
def test_store_and_publish_casts_only_mismatched_bfp(storage_format, value_type, needs_cast):
    storage_type = _storage_block(storage_format) if storage_format else BF16_BLOCK
    value = SimpleNamespace(type=value_type)
    packed = SimpleNamespace(type=storage_type)
    reservation, cb = object(), object()
    ttl = SimpleNamespace(typecast=Mock(return_value=packed), store=Mock(), cb_push=Mock())
    emitter = SimpleNamespace(
        groups={0: SimpleNamespace(storage_format=storage_format)},
        reserved={0: reservation},
        cbs={0: cb},
        assembled={0: value},
        bindings=SimpleNamespace(ttl=ttl),
        type_converter=SimpleNamespace(block_type=Mock(return_value=storage_type)),
    )

    TTLFunctionEmitter.store_and_publish(emitter, 0, value)

    if needs_cast:
        ttl.typecast.assert_called_once_with(storage_type, value)
    else:
        ttl.typecast.assert_not_called()
    ttl.store.assert_called_once_with(packed if needs_cast else value, reservation)
    ttl.cb_push.assert_called_once_with(cb)
    assert emitter.reserved == {}
    assert emitter.assembled == {}


def test_bfp_tensor_abi_preserves_storage_and_logical_dtype(device, formats):
    metadata = build_module_metadata(device)
    assert metadata == build_module_metadata(device)
    assert metadata.target_arch == "wormhole_b0"
    assert metadata.launch_grid == (1, 1)
    assert len(metadata.tensor_abi) == 2
    for index, (entry, storage_format, effect) in enumerate(zip(metadata.tensor_abi, formats, ("input", "output"))):
        assert entry.global_arg_index == index
        assert entry.shape == (64, 64)
        assert entry.tile_shape == (32, 32)
        assert entry.dtype == "bfloat16"
        assert entry.storage_format == storage_format
        assert entry.memory_space == "dram"
        assert entry.memory_layout == "interleaved"
        assert entry.effect == effect


def test_real_bfp_explicit_casts_and_mlir_verify(artifact, device, formats):
    input_format, output_format = formats
    assert artifact.metadata == build_module_metadata(device)
    with artifact.context:
        assert artifact.operation.verify()
        functions = {str(function.sym_name.value): function for function in artifact.body.operations}
        assert set(functions) == {"main_trisc", "main_ncrisc", "main_brisc"}
        compute = list(functions["main_trisc"].body.blocks[0].operations)
        adds = [op for op in compute if op.operation.name == "ttl.add"]
        assert len(adds) == 1
        add = adds[0]
        assert str(add.result.type) == BF16_BLOCK
        for operand in add.operands:
            assert str(operand.type) == BF16_BLOCK
            cast = operand.owner
            assert cast.name == "ttl.typecast"
            assert str(cast.operands[0].type) == _storage_block(input_format)
            assert cast.operands[0].owner.name == "ttl.attach_cb"

        stores = [op for op in compute if op.operation.name == "ttl.store"]
        assert stores
        for store in stores:
            value, destination = store.operands
            assert str(value.type) == str(destination.type) == _storage_block(output_format)
            cast = value.owner
            assert cast.name == "ttl.typecast"
            assert str(cast.operands[0].type) == BF16_BLOCK
        # The computed value itself, not just a later copy, is explicitly packed.
        assert any(store.operands[0].owner.operands[0] == add.result for store in stores)

        arguments = functions["main_ncrisc"].body.blocks[0].arguments
        assert len(arguments) == 2
        for argument, storage_format in zip(arguments, formats):
            tensor_type = artifact.bindings.ir.RankedTensorType(argument.type)
            tile_type = f"!ttcore.tile<32x32, {storage_format}>"
            assert list(tensor_type.shape) == [2, 2]
            assert str(tensor_type.element_type) == tile_type
            assert f"element_type = {tile_type}" in str(tensor_type.encoding)
        bindings = [op for op in compute if op.operation.name == "ttl.bind_cb"]
        groups = {int(group.storage_group_id): group for group in device.attrs["tt.dfb_storage_groups"]}
        assert len(bindings) == len(groups)
        for binding in bindings:
            group = groups[int(artifact.bindings.ir.IntegerAttr(binding.attributes["dfb_id"]).value)]
            assert f"!ttcore.tile<32x32, {group.storage_format}>" in str(binding.result.type)


def test_real_bfp_backend_compile(artifact):
    from ttl.passmanager import PassManager

    with artifact.context:
        assert artifact.operation.verify()
        manager = PassManager.parse(
            "builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})",
            context=artifact.context,
        )
        manager.enable_verifier(True)
        # Deliberately propagate backend diagnostics, including unsupported BFP
        # conversions; compilation failures must not be skipped or xfailed.
        manager.run(artifact.operation)
        assert artifact.operation.verify()
    source = artifact.inspect_source()
    assert "ttl.add" not in source
    assert "ttl.typecast" not in source
    assert "ttkernel." in source