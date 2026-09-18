"""Keep complete Device IR exports bounded without changing resource identity."""

from collections import defaultdict

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tvm import tirx

from testing.python.target.test_tilelang_tenstorrent_compute_values import _round
from testing.python.target.test_tilelang_tenstorrent_flash_attention import kernel, online_reference
from testing.python.target.test_tilelang_tenstorrent_lower_composition import interpret


@pytest.fixture(scope="module")
def attention_module():
    # Two cores, two head waves and three KV updates exercise repeated metadata
    # as well as values that must survive a precision-region boundary.
    config = dict(batch=1, heads=2, seq_q=64, seq_kv=96, head_dim=64, core_q=2, core_bh=1)
    frontend = tvm.IRModule({"main": kernel.make_flash_attention(**config)})
    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    return lower_tenstorrent_ir(frontend, target)


def test_complete_attention_exports_remain_bounded(attention_module):
    mod = attention_module
    text = mod.script(show_meta=True)
    serialized = tvm.ir.save_json(mod)
    # These budgets include all metadata, not just the printed instruction body.
    # Leave room for printer changes while catching duplicate metadata growth.
    text_bytes = len(text.encode())
    json_bytes = len(serialized.encode())
    assert text_bytes < 550_000
    assert json_bytes < 1_500_000
    assert "metadata =" in text
    assert int(mod.attrs["tt.device_ir_version"]) == 8
    assert len(mod.functions) == 6
    assert len(mod.attrs["tt.accumulator_table"]) == 24
    restored = tvm.ir.load_json(serialized)
    tvm.ir.assert_structural_equal(mod, restored)
    assert transform.VerifyTenstorrentDeviceIR()(restored).same_as(restored)


def test_shared_metadata_preserves_resource_and_operation_identity(attention_module):
    for mod in (attention_module, tvm.ir.load_json(tvm.ir.save_json(attention_module))):
        dfbs = mod.attrs["tt.dfb_table"]
        assert len(dfbs) > 1
        # Every hardware tile has the same shape; its immutable container may
        # be shared, whereas every logical resource remains a distinct object.
        assert all(dfb.tile_shape.same_as(dfbs[0].tile_shape) for dfb in dfbs)
        for key, identifier in (
            ("tt.dfb_table", "dfb_id"),
            ("tt.compute_value_table", "value_id"),
            ("tt.accumulator_table", "accumulator_id"),
        ):
            descriptors = mod.attrs[key]
            assert len({hash(desc) for desc in descriptors}) == len(descriptors)
            assert len({int(getattr(desc, identifier)) for desc in descriptors}) == len(descriptors)
            assert all(desc.source_span is not None for desc in descriptors)

        buffers = defaultdict(dict)
        for value in mod.attrs["tt.compute_value_table"]:
            buffer = value.buffer
            buffers[buffer.name][hash(buffer)] = buffer
        assert buffers
        for same_name in buffers.values():
            # Core-local fragments have equal names/shapes but distinct Buffer
            # and backing Var identities. SSA versions within a core share one.
            assert len(same_name) == 2
            assert len({hash(buffer.data) for buffer in same_name.values()}) == 2

        # Inspect statement occurrences, not post_order_visit (which deduplicates
        # nodes and would conceal accidentally shared effectful calls).
        calls = [
            stmt.value
            for func in mod.functions.values()
            for stmt in (func.body.seq if isinstance(func.body, tirx.SeqStmt) else [func.body])
            if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)
        ]
        assert len({hash(call) for call in calls}) == len(calls)
        markers = [call for call in calls if call.op.name == "tl.tt.compute_precision"]
        assert len(markers) > 2
        assert all(call.span is not None for call in markers)


def test_reloaded_attention_retains_bfloat16_values(attention_module):
    mod = tvm.ir.load_json(tvm.ir.save_json(attention_module))
    rng = np.random.default_rng(73)
    q = _round(rng.normal(0, 0.4, (1, 2, 64, 64)), "bfloat16")
    k = _round(rng.normal(0, 0.4, (1, 2, 96, 64)), "bfloat16")
    v = _round(rng.normal(0, 0.4, (1, 2, 96, 64)), "bfloat16")
    expected = online_reference(q, k, v, "bfloat16")
    actual = interpret(mod, [q, k, v, np.zeros_like(q)], seed=31)[0][-1]
    np.testing.assert_array_equal(actual, expected)


def _printer_descriptor(mod, attribute):
    if attribute == "tt.compute_requirements":
        return next(func.attrs[attribute] for func in mod.functions.values() if str(func.attrs["tt.kernel_slot"]) == "trisc")
    return mod.attrs[attribute][0]


def _print_descriptors(descriptors):
    func = tirx.PrimFunc([], tirx.Evaluate(0)).with_attr("test.descriptors", descriptors)
    return tvm.IRModule({"main": func}).script(show_meta=True)


def _reload_printed_metadata(text):
    import ast

    # Decode the literal supplied to load_json, without executing printed code.
    assignment = ast.parse(text).body[0]
    assert isinstance(assignment, ast.Assign)
    assert assignment.targets[0].id == "metadata"
    return tvm.ir.load_json(ast.literal_eval(assignment.value.args[0]))


@pytest.mark.parametrize(
    "attribute",
    ["tt.dfb_table", "tt.accumulator_table", "tt.compute_value_table", "tt.compute_requirements", "tt.pipe_transfer_table"],
)
def test_metadata_printer_indexes_by_identity(attention_module, attribute):
    first = _printer_descriptor(attention_module, attribute)
    equal = tvm.ir.load_json(tvm.ir.save_json(first))
    tvm.ir.assert_structural_equal(first, equal, map_free_vars=True)
    assert not first.same_as(equal)
    text = _print_descriptors([first, equal, first])
    type_key = "tl.tenstorrent." + type(first).__name__
    assert text.count(f'metadata["{type_key}"][0]') == 2
    assert text.count(f'metadata["{type_key}"][1]') == 1
    metadata = _reload_printed_metadata(text)
    assert len(metadata[type_key]) == 2
    assert not metadata[type_key][0].same_as(metadata[type_key][1])
    tvm.ir.assert_structural_equal(first, metadata[type_key][0], map_free_vars=True)
    tvm.ir.assert_structural_equal(equal, metadata[type_key][1], map_free_vars=True)


@pytest.mark.parametrize(
    "attribute",
    ["tt.dfb_table", "tt.accumulator_table", "tt.compute_value_table", "tt.compute_requirements", "tt.pipe_transfer_table"],
)
def test_metadata_printer_sessions_are_independent(attention_module, attribute):
    first = _printer_descriptor(attention_module, attribute)
    second = tvm.ir.load_json(tvm.ir.save_json(first))
    original_text = _print_descriptors([first, second, first])
    next_text = _print_descriptors([second])
    type_key = "tl.tenstorrent." + type(second).__name__
    # The same object was index 1 in the previous session; this new session
    # must emit its complete metadata and assign its first index afresh.
    assert next_text.count(f'metadata["{type_key}"][0]') == 1
    assert f'metadata["{type_key}"][1]' not in next_text
    metadata = _reload_printed_metadata(next_text)
    assert len(metadata[type_key]) == 1
    tvm.ir.assert_structural_equal(second, metadata[type_key][0], map_free_vars=True)
    repeated_text = _print_descriptors([first, second, first])
    assert repeated_text == original_text


def test_metadata_sharing_preserves_raw_float_bits_and_literal_spans(attention_module):
    import math
    import struct

    import tvm_ffi

    first_span = tvm.ir.Span(tvm.ir.SourceName("first.py"), 1, 1, 1, 2)
    second_span = tvm.ir.Span(tvm.ir.SourceName("second.py"), 2, 2, 1, 2)
    first_nan = struct.unpack("d", struct.pack("Q", 0x7FF8000000000001))[0]
    second_nan = struct.unpack("d", struct.pack("Q", 0x7FF8000000000002))[0]
    mod = attention_module.with_attr("test.raw_floats", tvm_ffi.convert([[0.0], [-0.0], [first_nan], [second_nan]]))
    mod = mod.with_attr(
        "test.literals",
        [tirx.IntImm("int32", 1, first_span), tirx.IntImm("int32", 1, second_span), tirx.IntImm("int64", 1, first_span)],
    )
    result = transform.InferTenstorrentComputeRequirements()(mod)
    numbers = result.attrs["test.raw_floats"]
    assert math.copysign(1, numbers[0][0]) == 1
    assert math.copysign(1, numbers[1][0]) == -1
    assert struct.pack("d", numbers[2][0]) == struct.pack("d", first_nan)
    assert struct.pack("d", numbers[3][0]) == struct.pack("d", second_nan)
    literals = result.attrs["test.literals"]
    assert not literals[0].same_as(literals[1])
    assert not literals[0].same_as(literals[2])
    assert literals[0].span.same_as(first_span)
    assert literals[1].span.same_as(second_span)
