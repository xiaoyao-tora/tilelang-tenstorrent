"""Native statement compaction preserves exact ordered device instructions."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent.device_ir import compact_device_ir, expand_device_ir
from tvm import ir, tirx


def _module(body):
    mod = tvm.IRModule({"main": tirx.PrimFunc([], body)})
    return (
        mod.with_attr("tt.device_ir_version", tirx.IntImm("int32", 7)).with_attr("tt.dfb_table", []).with_attr("tt.compute_value_table", [])
    )


def _call(name, args, annotations=None, span=None):
    return tirx.Evaluate(tirx.Call("void", ir.Op.get(name), args, annotations or {}, span), span)


def _statement_span(statement):
    # Stmt's native span is reflected, but the Python Stmt base is unregistered.
    fields = type(statement).__tvm_ffi_type_info__.parent_type_info.fields
    return next(field for field in fields if field.name == "span").getter(statement)


def _nodes(body, cls):
    result = []
    tirx.stmt_functor.post_order_visit(body, lambda node: result.append(node) if isinstance(node, cls) else None)
    return result


def _statements(values, *, dtype="int32", spans=None):
    statements = []
    for value in values:
        for index, op in enumerate(("tl.tt.dfb_reserve", "tl.tt.dfb_wait")):
            span = spans[index] if spans else None
            integer = tirx.IntImm(dtype, value + index, span)
            annotations = {
                "test_nested": [[integer, tirx.IntImm(dtype, 3, span)]],
                "test_expression": tirx.Add(integer, tirx.IntImm(dtype, 2, span), span),
            }
            statements.append(_call(op, [integer, tirx.IntImm(dtype, 1, span)], annotations, span))
    return tirx.SeqStmt(statements)


def test_affine_blocks_restore_nested_annotations_dtype_span_and_expression_shape():
    source = ir.SourceName("statement_compaction_test.py")
    spans = [ir.Span(source, index + 1, index + 1, 1, 12) for index in range(2)]
    sequence_span = ir.Span(source, 1, 10, 1, 12)
    body = _statements(range(100, 180, 10), dtype="int64", spans=spans)
    original = _module(tirx.SeqStmt(body.seq, sequence_span))
    compact = compact_device_ir(original, force=True)
    assert _statement_span(compact["main"].body).same_as(sequence_span)
    assert len(_nodes(compact["main"].body, tirx.For)) == 1
    assert len(_nodes(compact["main"].body, tirx.Evaluate)) == 2
    expanded = expand_device_ir(compact)
    assert _statement_span(expanded["main"].body).same_as(sequence_span)
    ir.assert_structural_equal(original, expanded)
    for index, statement in enumerate(expanded["main"].body.seq):
        call = statement.value
        assert call.span.same_as(spans[index % 2])
        assert call.args[0].span.same_as(spans[index % 2])
        assert call.args[0].dtype == "int64"
        assert isinstance(call.annotations["test_expression"], tirx.Add)
        assert call.annotations["test_nested"][0][0].span.same_as(spans[index % 2])
    restored = ir.load_json(ir.save_json(compact))
    ir.assert_structural_equal(expand_device_ir(restored), original)


def test_negative_stride_and_non_affine_integer_columns_are_lossless():
    for values in (range(80, 0, -10), [0, 1, 4, 9, 16, 25, 36, 49]):
        original = _module(_statements(values))
        compact = compact_device_ir(original, force=True)
        ir.assert_structural_equal(expand_device_ir(compact), original)


def test_distinct_buffer_variable_and_span_identities_are_not_merged():
    source = ir.SourceName("different_identities.py")
    buffers = [tirx.decl_buffer((32, 32), name="same_name") for _ in range(8)]
    variables = [tirx.Var("same_name", "int32") for _ in range(8)]
    spans = [ir.Span(source, 1, 1, 1, 2) for _ in range(8)]
    statements = [_call("tl.tt.dfb_wait", [variables[i], 1], {"buffer": buffers[i]}, spans[i]) for i in range(len(buffers))]
    original = _module(tirx.SeqStmt(statements))
    compact = compact_device_ir(original, force=True)
    assert not _nodes(compact["main"].body, tirx.For)
    expanded = expand_device_ir(compact)
    for i, statement in enumerate(expanded["main"].body.seq):
        assert statement.value.args[0].same_as(variables[i])
        assert statement.value.annotations["buffer"].same_as(buffers[i])
        assert statement.value.span.same_as(spans[i])


def _replace_compact_body(body):
    compact = compact_device_ir(_module(_statements(range(8))), force=True)
    compact.update_func(compact.get_global_var("main"), compact["main"].with_body(body))
    return compact


@pytest.mark.parametrize("dtype,base,step", [("int8", 120, 10), ("int64", 2**63 - 1, 1), ("uint32", 2**32 - 1, 1)])
def test_expansion_rejects_integer_overflow(dtype, base, step):
    variable = tirx.Var("iteration", "int64")
    value = tirx.Add(tirx.IntImm(dtype, base), tirx.Mul(tirx.Cast(dtype, variable), tirx.IntImm(dtype, step)))
    body = tirx.For(variable, 0, 2, tirx.ForKind.SERIAL, _call("tl.tt.dfb_wait", [value, 1]))
    with pytest.raises((ValueError, tvm.error.TVMError), match="overflow"):
        expand_device_ir(_replace_compact_body(body))


@pytest.mark.parametrize("minimum,extent,kind", [(-1, 2, tirx.ForKind.SERIAL), (0, 0, tirx.ForKind.SERIAL), (0, 2, tirx.ForKind.PARALLEL)])
def test_expansion_rejects_unsupported_loop_forms(minimum, extent, kind):
    variable = tirx.Var("iteration", "int32")
    body = tirx.For(variable, minimum, extent, kind, _call("tl.tt.dfb_wait", [0, 1]))
    with pytest.raises((ValueError, tvm.error.TVMError), match="compact loops require"):
        expand_device_ir(_replace_compact_body(body))


def test_nested_loop_budget_is_checked_before_expansion():
    inner = tirx.For(tirx.Var("inner", "int32"), 0, 2001, tirx.ForKind.SERIAL, _call("tl.tt.dfb_wait", [0, 1]))
    outer = tirx.For(tirx.Var("outer", "int32"), 0, 2001, tirx.ForKind.SERIAL, inner)
    with pytest.raises((ValueError, tvm.error.TVMError), match="budget"):
        expand_device_ir(_replace_compact_body(outer))


def test_expansion_rejects_empty_sequence_inside_large_loop():
    body = tirx.For(tirx.Var("iteration", "int32"), 0, 4000000, tirx.ForKind.SERIAL, ir.make_node("tirx.SeqStmt", seq=[], span=None))
    with pytest.raises((ValueError, tvm.error.TVMError), match="must not be empty"):
        expand_device_ir(_replace_compact_body(body))


def test_unit_extent_nesting_cannot_bypass_expansion_limits():
    body = _call("tl.tt.dfb_wait", [0, 1])
    for depth in range(66):
        body = tirx.For(tirx.Var(f"iteration_{depth}", "int32"), 0, 1, tirx.ForKind.SERIAL, body)
    with pytest.raises((ValueError, tvm.error.TVMError), match="nesting"):
        expand_device_ir(_replace_compact_body(body))


def test_loop_variable_outside_an_affine_integer_is_rejected():
    variable = tirx.Var("iteration", "int32")
    body = tirx.For(variable, 0, 2, tirx.ForKind.SERIAL, _call("tl.tt.dfb_wait", [variable, 1]))
    with pytest.raises((ValueError, tvm.error.TVMError), match="outside a generated affine"):
        expand_device_ir(_replace_compact_body(body))


@pytest.mark.parametrize(
    "variable_dtype,bounds_dtype,extent,step_dtype",
    [
        ("float32", "int32", 2, None),
        ("int8", "int32", 300, None),
        ("int8", "int8", 300, None),
        ("uint8", "uint8", 256, None),
        ("int32", "int32", 2, "int64"),
    ],
)
def test_reflected_loop_integer_dtypes_cannot_bypass_validation(variable_dtype, bounds_dtype, extent, step_dtype):
    # Deserialization uses reflected fields and can bypass For/IntImm constructors.
    bound = ir.make_node("ir.IntImm", dtype=tvm.DataType(bounds_dtype), value=extent, span=None)
    body = ir.make_node(
        "tirx.For",
        span=None,
        loop_var=tirx.Var("iteration", variable_dtype),
        min=tirx.IntImm(bounds_dtype, 0),
        extent=bound,
        kind=int(tirx.ForKind.SERIAL),
        body=tirx.Evaluate(0),
        thread_binding=None,
        annotations={},
        step=tirx.IntImm(step_dtype, 1) if step_dtype else None,
    )
    with pytest.raises((ValueError, tvm.error.TVMError), match="scalar integer loop variable"):
        expand_device_ir(_replace_compact_body(body))
