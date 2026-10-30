"""Lossless descriptor families and bounded integer-column decoding."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import transform
from tilelang.tenstorrent.device_ir import (
    ComputeValueDescriptor,
    CoreDomain,
    DFBDescriptor,
    DeviceIRDescriptorFamily,
    DeviceIRIntColumn,
    compact_device_ir,
    expand_device_ir,
)
from tvm import tirx
from tvm.ir import SourceName, Span


def _span():
    return Span(SourceName("compact_tables.py"), 1, 1, 1, 10)


def _value(identifier=0, *, buffer=None, span=None, version=0):
    if buffer is None:
        buffer = tirx.decl_buffer((32, 32), "float32", name="fragment", scope="local.fragment")
    return ComputeValueDescriptor(identifier, buffer, version, -1, -1, span or _span())


def _tables(dfbs=(), values=()):
    return tvm.IRModule({}).with_attrs(
        {"tt.device_ir_version": tirx.IntImm("int32", 8), "tt.dfb_table": list(dfbs), "tt.compute_value_table": list(values)}
    )


def _normalized_tables(dfbs=(), values=()):
    # Use the existing metadata interning stage, before the size threshold for
    # automatic compaction, to model how formation supplies shared invariants.
    return transform.InferTenstorrentComputeRequirements()(_tables(dfbs, values))


@pytest.mark.parametrize(
    "values,period,explicit",
    [
        ([7], 0, False),
        ([9, 5, 1, -3], 0, False),
        ([2, 7, 12, 17], 0, False),
        ([3, 11, 3, 11, 3, 11], 2, True),
        ([3, 11, 23, 31, 43, 51, 63], 2, True),
        ([3, 11, -17, -9, -37, -29, -57], 2, True),
        ([0, 3, 9, 2, 8], 0, True),
        ([-(2**63), 2**63 - 1], 0, True),
    ],
)
def test_descriptor_integer_columns_round_trip(values, period, explicit):
    prototype = _value()
    original = _tables(values=[_value(value, buffer=prototype.buffer, span=prototype.source_span) for value in values])
    compact = compact_device_ir(original, force=True)
    (family,) = compact.attrs["tt.compact_value_families"]
    column = family.fields["value_id"]
    assert int(column.period) == period
    assert bool(len(column.values)) == explicit
    assert int(column.count) == len(values)
    for encoded in (compact, tvm.ir.load_json(tvm.ir.save_json(compact))):
        expanded = expand_device_ir(encoded)
        assert [int(value.value_id) for value in expanded.attrs["tt.compute_value_table"]] == values
        tvm.ir.assert_structural_equal(original, expanded)


def _column_module(column, *, positions=None):
    count = int(column.count)
    family = DeviceIRDescriptorFamily(
        _value(),
        positions if positions is not None else DeviceIRIntColumn(0, 1, count),
        {
            "value_id": column,
            "version": DeviceIRIntColumn(0, 0, count),
            "previous_value_id": DeviceIRIntColumn(-1, 0, count),
            "accumulator_id": DeviceIRIntColumn(-1, 0, count),
        },
    )
    return tvm.IRModule({}).with_attrs(
        {"tt.device_ir_version": 9, "tt.compact_original_version": 8, "tt.compact_dfb_families": [], "tt.compact_value_families": [family]}
    )


@pytest.mark.parametrize(
    "args,match",
    [
        ((2**63 - 1, 1, 2), "overflow"),
        ((-(2**63), -1, 2), "overflow"),
        ((0, 1, 4, [2**63 - 1, 0], 2), "overflow"),
        ((0, -1, 4, [-(2**63), 0], 2), "overflow"),
        ((1, 0, 4, [3, 5], 2), "periodic"),
        ((0, 0, 4, [3], 2), "periodic"),
        ((0, 0, 2, [3, 5], 2), "periodic"),
        ((0, 0, 4, [3, 5]), "explicit"),
        ((0, 1, 2, [3, 5]), "explicit"),
        ((0, 0, 2, [], -1), "period"),
        ((0, 0, -1), "budget|size"),
        ((0, 0, 4_000_001), "budget"),
    ],
)
def test_invalid_columns_fail_before_expansion(args, match):
    with pytest.raises(Exception, match=match):
        expand_device_ir(_column_module(DeviceIRIntColumn(*args)))


@pytest.mark.parametrize(
    "positions,match",
    [
        ((0, 0, 2), "duplicate"),
        ((-1, 1, 2), "outside"),
        ((0, 2, 2), "outside"),
        ((0, 1, 1), "length"),
    ],
)
def test_family_positions_are_unique_and_complete(positions, match):
    with pytest.raises(Exception, match=match):
        expand_device_ir(_column_module(DeviceIRIntColumn(0, 1, 2), positions=DeviceIRIntColumn(*positions)))


def _dfb(identifier, source, *, span, domain):
    return DFBDescriptor(identifier, source, "bfloat16", (32, 32), (1, 1), 1, None, "trisc", domain, "trisc", domain, 1, span)


def test_source_patterns_preserve_spelling_positions_and_span_identity():
    span = _span()
    domain = CoreDomain((0, 0), (1, 1))
    sources = [
        "compute.materialization.2.v2.core0_0",
        "buffer.0.v3.core0_0",
        "compute.materialization.5.v5.core0_0",
        "buffer.0.v7.core0_0",
        "literal_without_digits",
        "literal_without_digits",
        "buffer.00.v8.core0_0",
    ]
    original = _normalized_tables([_dfb(index, source, span=span, domain=domain) for index, source in enumerate(sources)])
    compact = compact_device_ir(original, force=True)
    families = compact.attrs["tt.compact_dfb_families"]
    assert len(families) == 5
    assert [int(family.positions.count) for family in families] == [2, 2, 1, 1, 1]
    expanded = expand_device_ir(compact)
    descriptors = expanded.attrs["tt.dfb_table"]
    assert [str(dfb.source_buffer_identity) for dfb in descriptors] == sources
    assert all(dfb.source_span.same_as(span) for dfb in descriptors)
    assert len({hash(dfb) for dfb in descriptors}) == len(sources)
    tvm.ir.assert_structural_equal(original, expanded)
    tvm.ir.assert_structural_equal(original, expand_device_ir(tvm.ir.load_json(tvm.ir.save_json(compact))))


def test_value_families_preserve_buffer_variable_and_span_identity():
    first, second = _value(), _value()
    span1, span2 = _span(), _span()
    values = [
        _value(index, buffer=buffer, span=span, version=version)
        for index, (buffer, span, version) in enumerate(
            [(first.buffer, span1, 0), (second.buffer, span1, 0), (first.buffer, span2, 1), (first.buffer, span1, 2)]
        )
    ]
    original = _normalized_tables(values=values)
    compact = compact_device_ir(original, force=True)
    assert len(compact.attrs["tt.compact_value_families"]) == 3
    expanded = expand_device_ir(compact)
    restored = expanded.attrs["tt.compute_value_table"]
    for before, after in zip(original.attrs["tt.compute_value_table"], restored):
        assert before.buffer.same_as(after.buffer)
        assert before.buffer.data.same_as(after.buffer.data)
        assert before.source_span.same_as(after.source_span)
    assert not restored[0].buffer.same_as(restored[1].buffer)
    assert not restored[0].source_span.same_as(restored[2].source_span)
    tvm.ir.assert_structural_equal(original, expanded)


@pytest.mark.parametrize("parts", [("buffer.0.v", ""), ("buffer.", "", ".tail")])
def test_noncanonical_source_patterns_are_rejected(parts):
    family = DeviceIRDescriptorFamily(
        _dfb(0, "buffer.0.v0", span=_span(), domain=CoreDomain((0, 0), (1, 1))),
        DeviceIRIntColumn(0, 1, 1),
        {"dfb_id": DeviceIRIntColumn(0, 1, 1), **{f"source.{index}": DeviceIRIntColumn(0, 0, 1) for index in range(len(parts) - 1)}},
        parts,
    )
    compact = tvm.IRModule({}).with_attrs(
        {"tt.device_ir_version": 9, "tt.compact_original_version": 8, "tt.compact_dfb_families": [family], "tt.compact_value_families": []}
    )
    with pytest.raises(Exception, match="canonical"):
        expand_device_ir(compact)


def _budget_dfb_family(count, *, prefix="buffer.", source_columns=1, offset=0):
    return DeviceIRDescriptorFamily(
        _dfb(offset, "buffer.0", span=_span(), domain=CoreDomain((0, 0), (1, 1))),
        DeviceIRIntColumn(offset, 1, count),
        {
            "dfb_id": DeviceIRIntColumn(offset, 1, count),
            **{f"source.{index}": DeviceIRIntColumn(offset, 1, count) for index in range(source_columns)},
        },
        [prefix] + ["_"] * (source_columns - 1) + [""],
    )


@pytest.mark.parametrize("families", [1, 2])
def test_source_expansion_budget_is_checked_before_allocation(families):
    # Each encoded pattern is only 8 KiB, but 40,000 logical source strings
    # would exceed 256 MiB. Splitting the rows cannot evade the module budget.
    rows = 40_000 // families
    descriptors = [_budget_dfb_family(rows, prefix="x" * 8192, offset=index * rows) for index in range(families)]
    compact = tvm.IRModule({}).with_attrs(
        {
            "tt.device_ir_version": 9,
            "tt.compact_original_version": 8,
            "tt.compact_dfb_families": descriptors,
            "tt.compact_value_families": [],
        }
    )
    with pytest.raises(Exception, match="source string budget"):
        expand_device_ir(compact)


def test_integer_expansion_budget_combines_all_tables():
    # The DFB table needs 30M decoded integers (positions + ID + eight source
    # columns), and the value table another 5M. Each fits the integer budget
    # alone, and the combined logical row count fits the existing 4M limit.
    compact = _column_module(DeviceIRIntColumn(0, 1, 1_000_000))
    compact = compact.with_attr("tt.compact_dfb_families", [_budget_dfb_family(3_000_000, source_columns=8)])
    with pytest.raises(Exception, match="integer column budget"):
        expand_device_ir(compact)


def test_auto_compaction_keeps_expanded_ir_when_source_budget_would_fail():
    # Repeated handles keep this representation-helper test small. The helper
    # does not perform semantic validation of duplicate resource identities.
    # Only one 64 KiB string is retained, while v9 would allocate 256 MiB+.
    prototype = _dfb(0, "x" * 65536 + ".0", span=_span(), domain=CoreDomain((0, 0), (1, 1)))
    original = _tables(dfbs=[prototype] * 4096)
    assert compact_device_ir(original).same_as(original)
    with pytest.raises(ValueError, match="metadata exceeds compact expansion budget"):
        compact_device_ir(original, force=True)


def test_auto_compaction_counts_shared_statement_occurrences_with_descriptors():
    # A small shared DAG represents the logical sequence without allocating
    # millions of Python/TIR nodes. Descriptors and operations share one budget.
    remaining = 4_000_000 - 4096 + 1
    statement = tirx.Evaluate(0)
    pieces = []
    while remaining:
        if remaining & 1:
            pieces.append(statement)
        remaining >>= 1
        if remaining:
            statement = tvm.ir.make_node("tirx.SeqStmt", seq=[statement, statement], span=None)
    body = tvm.ir.make_node("tirx.SeqStmt", seq=pieces, span=None)
    original = _tables(values=[_value()] * 4096)
    original = tvm.IRModule({"main": tirx.PrimFunc([], body)}, attrs=original.attrs)
    assert compact_device_ir(original).same_as(original)
    with pytest.raises(ValueError, match="module exceeds compact expansion budget"):
        compact_device_ir(original, force=True)
