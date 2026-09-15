from __future__ import annotations

import pytest
from tvm import IRModule, ir, tirx
from tvm.ir import SourceName, Span
from tvm.target import Target

from tilelang.tenstorrent.device_ir import (
    BUFFER_METADATA_TABLE_ATTR,
    CoreCoord,
    CoreDomain,
    DFBDescriptor,
    DeviceFunctionMetadata,
    DeviceModuleMetadata,
    LogicalKernel,
    OperationIdentity,
    TensorBacking,
    TensorDescriptor,
    VerifyTenstorrentDeviceIR,
    attach_device_function_metadata,
    attach_device_module_metadata,
)


def _span(line: int = 1) -> Span:
    return Span(SourceName("device_ir_test.py"), line, line, 1, 10)


def _make_slot(slot: str, *, role: str = "idle", noc_index=None):
    thread = "compute" if slot == "trisc" else "datamovement"
    if noc_index is None and slot != "trisc":
        noc_index = 0 if slot == "ncrisc" else 1
    metadata = DeviceFunctionMetadata(
        slot,
        thread,
        noc_index,
        LogicalKernel(f"kernel.{slot}", role, role, _span(2)),
        [],
        CoreDomain((0, 0), (1, 1)),
    )
    func = tirx.PrimFunc([], tirx.Evaluate(0))
    return attach_device_function_metadata(
        func,
        metadata,
        global_symbol=f"operation_{slot}",
        target=Target({"kind": "tenstorrent", "arch": "wormhole_b0"}),
    )


def _make_valid_module() -> IRModule:
    mod = IRModule(
        {
            "operation_trisc": _make_slot("trisc"),
            "operation_ncrisc": _make_slot("ncrisc"),
            "operation_brisc": _make_slot("brisc"),
        }
    )
    metadata = DeviceModuleMetadata(
        "wormhole_b0",
        CoreCoord(1, 1),
        OperationIdentity("operation", _span()),
    )
    return attach_device_module_metadata(mod, metadata)


def test_typed_descriptor_fields_and_json_round_trip():
    tensor = TensorDescriptor(
        0,
        (32, 32),
        "bfloat16",
        (32, 1),
        (32, 32),
        (1, 1),
        "dram",
        "interleaved",
        None,
        "input",
        0,
        _span(4),
    )
    assert tensor.global_arg_index == 0
    assert [int(value) for value in tensor.shape] == [32, 32]
    assert str(tensor.dtype) == "bfloat16"
    assert tensor.effect == "input"

    loaded = ir.load_json(ir.save_json(tensor))
    assert isinstance(loaded, TensorDescriptor)
    assert loaded.global_arg_index == tensor.global_arg_index
    assert [int(value) for value in loaded.tile_shape] == [32, 32]
    assert loaded.source_span.source_name.name == "device_ir_test.py"

    backing = TensorBacking(0, 128)
    restored_backing = ir.load_json(ir.save_json(backing))
    assert int(restored_backing.global_arg_index) == 0
    assert int(restored_backing.byte_offset) == 128

    domain = CoreDomain((0, 0), (1, 1))
    dfb = DFBDescriptor(
        0,
        "buffer.0",
        "bfloat16",
        (32, 32),
        (1, 1),
        2,
        backing,
        "ncrisc",
        domain,
        "trisc",
        domain,
        1,
        _span(5),
    )
    restored_dfb = ir.load_json(ir.save_json(dfb))
    assert isinstance(restored_dfb.tensor_backing, TensorBacking)
    assert int(restored_dfb.tensor_backing.global_arg_index) == 0
    assert int(restored_dfb.tensor_backing.byte_offset) == 128


def test_valid_three_slot_skeleton_round_trips_and_verifier_is_read_only():
    mod = _make_valid_module()
    loaded = ir.load_json(ir.save_json(mod))
    verified = VerifyTenstorrentDeviceIR()(loaded)

    assert verified.same_as(loaded)
    assert [str(value) for value in verified.attrs["tt.kernel_order"]] == [
        "trisc",
        "ncrisc",
        "brisc",
    ]
    for func in verified.functions.values():
        assert BUFFER_METADATA_TABLE_ATTR not in func.attrs


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda mod: mod.without_attr("tt.target_arch"), "tt.target_arch"),
        (lambda mod: mod.with_attr("tt.device_ir_version", 99), "expected 1"),
        (
            lambda mod: mod.with_attr("tt.kernel_order", ["brisc", "ncrisc", "trisc"]),
            "tt.kernel_order",
        ),
    ],
)
def test_module_schema_invalid_matrix(mutate, message):
    mod = mutate(_make_valid_module())
    before = ir.save_json(mod)
    with pytest.raises(Exception, match=message):
        VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before


def test_duplicate_slot_is_rejected_without_modifying_input():
    mod = _make_valid_module()
    brisc_gv = mod.get_global_var("operation_brisc")
    duplicate = mod["operation_ncrisc"].with_attr("global_symbol", "operation_brisc")
    mod[brisc_gv] = duplicate
    before = ir.save_json(mod)
    with pytest.raises(Exception, match="duplicate slot|duplicate PrimFunc"):
        VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before


def test_wrong_slot_noc_and_form_input_metadata_are_rejected():
    mod = _make_valid_module()
    ncrisc_gv = mod.get_global_var("operation_ncrisc")
    mod[ncrisc_gv] = mod[ncrisc_gv].with_attr("tt.noc_index", 1)
    with pytest.raises(Exception, match="wrong noc_index"):
        VerifyTenstorrentDeviceIR()(mod)

    mod = _make_valid_module()
    trisc_gv = mod.get_global_var("operation_trisc")
    mod[trisc_gv] = mod[trisc_gv].with_attr(BUFFER_METADATA_TABLE_ATTR, [])
    with pytest.raises(Exception, match="Form-input-only"):
        VerifyTenstorrentDeviceIR()(mod)
