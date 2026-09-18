"""Hardware-independent tests for unified Tenstorrent elementwise capture."""

import pytest

from tilelang import transform, tvm
from tilelang.backend import create_backend_context
from tilelang.tenstorrent import execution_backend
from tilelang.tenstorrent import language as T
from tvm import ir, tirx


TARGET = {"kind": "tenstorrent", "arch": "wormhole_b0"}
ALLOCATIONS = "tl.alloc_buffer_annotations"
METADATA = {"tt.tile_shape": [32, 32], "tt.dfb_block_count": 2}


def _nodes(func, cls):
    result = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, cls) else None)
    return result


def _blocks(mod):
    return [node for node in _nodes(mod["main"], tirx.SBlock) if str(node.annotations.get("tl.tt.compute_kind", "")) == "elementwise"]


def _integers(values):
    return [int(value) for value in values]


def _annotation_value(value):
    """Use the same typed PrimExpr annotation values as the frontend FFI."""
    if isinstance(value, int):
        return tirx.IntImm("int32", value)
    if isinstance(value, (list, tuple)):
        return [_annotation_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _annotation_value(item) for key, item in value.items()}
    return value


def _scope(extents, variables, body, parallel=True):
    for axis in reversed(range(len(extents))):
        annotations = {
            "tl.tt.tiles_parallel": int(parallel),
            "tl.tt.tiles_stage": 0,
        }
        if axis == 0:
            annotations.update({"tl.tt.tiles_scope": 1, "tl.tt.tiles_domain": list(extents)})
        body = tirx.For(
            variables[axis],
            tirx.const(0, variables[axis].dtype),
            extents[axis],
            tirx.ForKind.SERIAL,
            body,
            annotations=_annotation_value(annotations),
        )
    return body


def _module(
    shape=(64, 64),
    *,
    b_shape=None,
    a_shape=None,
    loop_dtype="int32",
    a_scope="shared",
    b_scope="shared",
    a_buffer_options=None,
    c_scope="shared",
    c_dtype="float32",
    domain=None,
    expression=None,
    body=None,
    metadata=True,
    a_metadata=None,
    outer_loop=False,
    twice=False,
    parallel=True,
    flat_allocations=False,
):
    """Build raw frontend IR so pass diagnostics are tested without parser guards."""
    domain = shape if domain is None else domain
    a = tirx.decl_buffer(shape if a_shape is None else a_shape, "float32", name="A", scope=a_scope, **(a_buffer_options or {}))
    b = tirx.decl_buffer(shape if b_shape is None else b_shape, "float32", name="B", scope=b_scope)
    c = tirx.decl_buffer(shape, c_dtype, name="C", scope=c_scope)
    variables = [tirx.Var(f"index_{axis}", loop_dtype) for axis in range(len(domain))]
    value = a[tuple(variables)] + b[tuple(variables)] if expression is None else expression(a, b, c, variables)
    store = tirx.BufferStore(c, value, variables)
    inner = store if body is None else body(a, b, c, variables, store)
    statement = _scope(domain, variables, inner, parallel)
    if twice:
        statement = tirx.SeqStmt([statement, statement])
    if outer_loop:
        statement = tirx.For(tirx.Var("iteration", "int32"), 0, 3, tirx.ForKind.SERIAL, statement)
    buffers = [a, b, c]
    records = {buffer.data: METADATA for buffer in buffers if buffer.scope() in ("shared", "shared.dyn")} if metadata else {}
    if a_metadata is not None:
        records[a.data] = a_metadata
    records = _annotation_value(records)
    if flat_allocations:
        statement = tirx.SeqStmt([tirx.AllocBuffer(buffer, records.get(buffer.data, {})) for buffer in buffers] + [statement])
    else:
        root = tirx.SBlock([], [], [], "root", statement, alloc_buffers=buffers, annotations={ALLOCATIONS: records})
        statement = tirx.SBlockRealize([], True, root)
    func = tirx.PrimFunc([], statement).with_attr("global_symbol", "main")
    return tirx.transform.BindTarget(tvm.target.Target(TARGET))(tvm.IRModule({"main": func}))


def _canonicalize(mod):
    return transform.CanonicalizeTTElementwise()(mod)


def _effects_by_name(regions):
    return {region.buffer.name: _integers([dim.extent for dim in region.region]) for region in regions}


def _reject(mod, match):
    before = ir.save_json(mod)
    with pytest.raises((tvm.error.TVMError, ValueError), match="(?i)" + match):
        _canonicalize(mod)
    assert ir.save_json(mod) == before, "A failed analysis must not partially rewrite its input"


@pytest.mark.parametrize("shape,block_shape", [((32, 32), [1, 1]), ((64, 64), [2, 2]), ((128, 32), [4, 1])])
def test_identity_captures_full_regions_and_tile_geometry(shape, block_shape):
    mod = _canonicalize(_module(shape))
    (block,) = _blocks(mod)
    assert block.name_hint == "tl.tt.elementwise"
    assert not block.iter_vars
    assert int(block.annotations["tl.tt.tiles_stage"]) == 1
    assert _integers(block.annotations["tl.tt.logical_domain"]) == list(shape)
    assert _integers(block.annotations["tl.tt.physical_tile_shape"]) == [32, 32]
    assert _integers(block.annotations["tl.tt.block_shape"]) == block_shape
    assert list(block.annotations["tl.tt.iterator_types"]) == ["parallel", "parallel"]
    assert _effects_by_name(block.reads) == {"A": list(shape), "B": list(shape)}
    assert _effects_by_name(block.writes) == {"C": list(shape)}
    for region in [*block.reads, *block.writes]:
        assert _integers([dim.min for dim in region.region]) == [0, 0]
        assert _integers(block.annotations["tl.tt.access_maps"][region.buffer.data]) == [0, 1]
    assert not block.annotations["tl.tt.broadcast_recipes"]
    assert isinstance(block.body, tirx.BufferStore)
    assert _integers(block.body.indices) == [0, 0]
    assert isinstance(block.body.value, tirx.Add)
    assert not _nodes(mod["main"], tirx.For)
    for load in _nodes(mod["main"], tirx.BufferLoad):
        assert _integers(load.indices) == [0, 0]
    for realize in _nodes(mod["main"], tirx.SBlockRealize):
        assert not realize.iter_values
        assert int(realize.predicate) == 1
    transform.VerifyTTComputeBlocks()(mod)


def test_compound_expression_and_repeated_loads():
    mod = _canonicalize(_module(expression=lambda a, b, c, ij: a[tuple(ij)] * b[tuple(ij)] + a[tuple(ij)]))
    (block,) = _blocks(mod)
    assert isinstance(block.body.value, tirx.Add)
    assert isinstance(block.body.value.a, tirx.Mul)
    assert _effects_by_name(block.reads) == {"A": [64, 64], "B": [64, 64]}


@pytest.mark.parametrize("kind", ["constant", "cast", "exp", "inplace"])
def test_pure_expression_templates(kind):
    def expression(a, b, c, ij):
        value = a[tuple(ij)]
        if kind == "constant":
            return value * tirx.const(2, "float32")
        if kind == "cast":
            return tirx.Cast("float32", tirx.Cast("bfloat16", value))
        if kind == "exp":
            return tirx.exp(value)
        return c[tuple(ij)] + value

    (block,) = _blocks(_canonicalize(_module(expression=expression)))
    assert set(_effects_by_name(block.reads)) == ({"A", "C"} if kind == "inplace" else {"A"})
    if kind == "inplace":
        assert next(region for region in block.reads if region.buffer.name == "C").buffer.same_as(block.writes[0].buffer)
    elif kind == "constant":
        assert isinstance(block.body.value, tirx.Mul)
        assert float(block.body.value.b) == 2.0
    elif kind == "cast":
        assert isinstance(block.body.value, tirx.Cast)
        assert block.body.value.value.dtype == "bfloat16"
    else:
        assert block.body.value.op.name == "tirx.exp"


@pytest.mark.parametrize(
    "kind,b_shape,axes,logical,broadcast_axes,block_shape",
    [
        ("row", (32, 256), [-1, 1], [1, 256], [0], [1, 8]),
        ("column", (64, 32), [0, -1], [64, 1], [1], [2, 1]),
        ("scalar", (32, 32), [-1, -1], [1, 1], [0, 1], [1, 1]),
    ],
)
def test_padded_broadcast_preserves_compact_operand_recipe(kind, b_shape, axes, logical, broadcast_axes, block_shape):
    def expression(a, b, c, ij):
        indices = [ij[axis] if axis >= 0 else 0 for axis in axes]
        return a[tuple(ij)] + b[tuple(indices)]

    (block,) = _blocks(_canonicalize(_module((64, 256), b_shape=b_shape, expression=expression)))
    operand = next(region for region in block.reads if region.buffer.name == "B")
    assert _integers([dim.extent for dim in operand.region]) == logical
    assert _integers(operand.buffer.shape) == list(b_shape)
    assert _integers(block.annotations["tl.tt.access_maps"][operand.buffer.data]) == axes
    recipe = block.annotations["tl.tt.broadcast_recipes"][operand.buffer.data]
    assert str(recipe["broadcast_kind"]) == kind
    assert _integers(recipe["broadcast_axes"]) == broadcast_axes
    assert _integers(recipe["logical_region"]) == logical
    assert _integers(recipe["physical_shape"]) == list(b_shape)
    assert _integers(recipe["block_shape"]) == block_shape
    assert _integers(block.annotations["tl.tt.block_shape"]) == [2, 8]


def test_different_access_maps_for_one_buffer_are_rejected():
    _reject(_module((32, 32), expression=lambda a, b, c, ij: a[tuple(ij)] + a[0, ij[1]]), "[Aa]ccess|[Mm]ap")


@pytest.mark.parametrize("outer_loop,twice", [(False, True), (True, False)])
def test_independent_scopes_preserve_enclosing_serial_loop(outer_loop, twice):
    mod = _canonicalize(_module(outer_loop=outer_loop, twice=twice))
    assert len(_blocks(mod)) == (2 if twice else 1)
    loops = _nodes(mod["main"], tirx.For)
    assert len(loops) == int(outer_loop)
    if outer_loop:
        assert int(loops[0].extent) == 3
        assert isinstance(loops[0].body, tirx.SBlockRealize)


@pytest.mark.parametrize("broadcast", [False, True])
def test_idempotency_and_json_roundtrip_preserve_buffer_identity(broadcast):
    source = _module((64, 256), b_shape=(32, 256), expression=lambda a, b, c, ij: a[tuple(ij)] + b[0, ij[1]]) if broadcast else _module()
    mod = _canonicalize(source)
    ir.assert_structural_equal(mod, _canonicalize(mod))
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    transform.VerifyTTComputeBlocks()(restored)
    (block,) = _blocks(restored)
    for effect in [*block.reads, *block.writes]:
        assert effect.buffer.data in block.annotations["tl.tt.access_maps"]


@pytest.mark.parametrize("flat_allocations", [False, True])
def test_allocation_metadata_survives_canonicalization_and_lower_opaque_block(flat_allocations):
    before = _module(flat_allocations=flat_allocations, a_metadata={**METADATA, "tt.dfb_block_count": 3})
    mod = _canonicalize(before)
    if not flat_allocations:
        old = next(node for node in _nodes(before["main"], tirx.SBlock) if ALLOCATIONS in node.annotations)
        new = next(node for node in _nodes(mod["main"], tirx.SBlock) if ALLOCATIONS in node.annotations)
        ir.assert_structural_equal(old.annotations[ALLOCATIONS], new.annotations[ALLOCATIONS])
    lowered = transform.LowerOpaqueBlock()(mod)
    allocations = _nodes(lowered["main"], tirx.AllocBuffer)
    assert len(allocations) == 3
    for allocation in allocations:
        assert _integers(allocation.annotations["tt.tile_shape"]) == [32, 32]
        assert int(allocation.annotations["tt.dfb_block_count"]) == (3 if allocation.buffer.name == "A" else 2)


@pytest.mark.parametrize("shape", [(32,), (48, 64)])
def test_reject_unsupported_domain_geometry(shape):
    _reject(_module(shape), "rank|[Rr]ank|divisib|32")


def test_reject_dynamic_domain():
    n = tirx.Var("n", "int32")
    _reject(_module((n, 64)), "constant|[Ss]tatic|dynamic|compile-time")


def _replace_tiles_loop_extent(func, source, inner):
    output = _nodes(func, tirx.BufferStore)[0].buffer

    def change(node):
        if not isinstance(node, tirx.For) or "tl.tt.tiles_stage" not in node.annotations:
            return None
        if ("tl.tt.tiles_scope" not in node.annotations) != inner:
            return None
        if source == "bool":
            extent = tirx.const(True, "bool")
        else:
            value = (
                tirx.call_extern(node.extent.dtype, "effectful_extent") if source == "call" else tirx.Cast(node.extent.dtype, output[0, 0])
            )
            # Explicit nodes preserve the canceled call/load in raw IR.
            extent = tirx.Add(tirx.Mul(value, tirx.const(0, node.extent.dtype)), node.extent)
        return tirx.For(node.loop_var, node.min, extent, node.kind, node.body, annotations=node.annotations)

    return func.with_body(tirx.stmt_functor.ir_transform(func.body, None, change), func.span)


@pytest.mark.parametrize("size", [32, 64])
@pytest.mark.parametrize("source", ["call", "load"])
@pytest.mark.parametrize("inner", [False, True])
def test_reject_nonconstant_original_tiles_extent(size, source, inner):
    mod = _module((size, size))
    malformed = tvm.IRModule({"main": _replace_tiles_loop_extent(mod["main"], source, inner)})
    _reject(malformed, "loop extent.*(constant|bool)")


@pytest.mark.parametrize("inner", [False, True])
def test_reject_bool_tiles_extent_before_capture(inner):
    # TIR rejects bool extents even before the canonicalizer can see them.
    with pytest.raises(tvm.error.TVMError, match="scalar integer.*extent.*bool"):
        _replace_tiles_loop_extent(_module()["main"], "bool", inner)


@pytest.mark.parametrize("b_shape,indices", [((32, 32), "identity"), ((256,), "row"), ((1, 256), "row"), ((64, 256), "row")])
def test_reject_mismatched_or_unpadded_operand(b_shape, indices):
    def expression(a, b, c, ij):
        access = tuple(ij) if indices == "identity" else ((ij[1],) if len(b_shape) == 1 else (0, ij[1]))
        return a[tuple(ij)] + b[access]

    _reject(_module((64, 256), b_shape=b_shape, expression=expression), "[Ss]hape|rank|[Rr]ank|divisib|padded")


@pytest.mark.parametrize("kind", ["transpose", "offset", "coordinate", "gather", "recurrence", "unknown_call"])
def test_reject_unsupported_access_or_expression(kind):
    def expression(a, b, c, ij):
        i, j = ij
        if kind == "transpose":
            return a[j, i]
        if kind == "offset":
            return a[i + 1, j]
        if kind == "coordinate":
            return a[i, j] + tirx.Cast("float32", i + j)
        if kind == "gather":
            return a[tirx.Cast("int32", b[i, j]), j]
        if kind == "recurrence":
            return c[i - 1, j] + a[i, j]
        return tirx.call_extern("float32", "unknown_elementwise", a[i, j])

    _reject(_module(expression=expression), "indices|[Aa]ccess|loop variables|[Cc]all|pure|Phase 1")


@pytest.mark.parametrize("kind", ["stores", "conditional", "nested", "loop", "side_effect", "copy", "gemm"])
def test_reject_unsupported_scope_body(kind):
    def body(a, b, c, ij, store):
        if kind == "stores":
            return tirx.SeqStmt([store, tirx.BufferStore(a, b[tuple(ij)], ij)])
        if kind == "conditional":
            return tirx.IfThenElse(ij[0] < 10, store, None)
        if kind == "nested":
            variables = [tirx.Var("nested_i", "int32"), tirx.Var("nested_j", "int32")]
            return _scope((64, 64), variables, store)
        if kind == "loop":
            return tirx.For(tirx.Var("k", "int32"), 0, 2, tirx.ForKind.SERIAL, store)
        if kind == "side_effect":
            value = tirx.call_extern("int32", "mutate", a.data)
        else:
            value = tirx.Call("handle", ir.Op.get("tl.tileop.copy" if kind == "copy" else "tl.tileop.gemm"), [])
        return tirx.SeqStmt([tirx.Evaluate(value), store])

    _reject(_module(body=body), "[Ss]tore|body|[Nn]ested|scope|unsupported|Phase 1")


@pytest.mark.parametrize("kind", ["load", "store"])
def test_reject_predicates(kind):
    def body(a, b, c, ij, store):
        if kind == "store":
            return tirx.BufferStore(c, a[tuple(ij)], ij, predicate=ij[0] < 8)
        return tirx.BufferStore(c, tirx.BufferLoad(a, ij, predicate=ij[0] < 8), ij)

    _reject(_module(body=body), "predicate|mask")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metadata": False},
        {"a_metadata": {"tt.dfb_block_count": 2}},
        {"a_metadata": {"tt.tile_shape": [16, 32], "tt.dfb_block_count": 2}},
        {"a_metadata": {"tt.tile_shape": [32, 32], "tt.dfb_block_count": 0}},
    ],
)
def test_reject_missing_or_inconsistent_allocation_metadata(kwargs):
    _reject(_module(**kwargs), "metadata|tt.tile_shape|tt.dfb_block_count|32")


@pytest.mark.parametrize("kwargs", [{"a_scope": "global"}, {"c_scope": "global"}])
def test_reject_global_access(kwargs):
    _reject(_module(**kwargs), "shared|scope")


def _change_compute_block(mod, change):
    def rewrite(node):
        if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations:
            fields = dict(
                iter_vars=node.iter_vars,
                reads=node.reads,
                writes=node.writes,
                name_hint=node.name_hint,
                body=node.body,
                init=node.init,
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations=dict(node.annotations),
            )
            change(fields)
            fields["annotations"] = _annotation_value(fields["annotations"])
            return tirx.SBlock(**fields)
        return None

    func = mod["main"]
    rewritten = tirx.stmt_functor.ir_transform(func.body, None, rewrite)
    return tvm.IRModule({"main": func.with_body(rewritten)})


@pytest.mark.parametrize(
    "key",
    [
        "tl.tt.tiles_stage",
        "tl.tt.logical_domain",
        "tl.tt.physical_tile_shape",
        "tl.tt.block_shape",
        "tl.tt.iterator_types",
        "tl.tt.access_maps",
        "tl.tt.broadcast_recipes",
        "tl.tt.value_kinds",
    ],
)
def test_verifier_rejects_missing_contract_metadata(key):
    mod = _change_compute_block(_canonicalize(_module()), lambda fields: fields["annotations"].pop(key))
    with pytest.raises((tvm.error.TVMError, ValueError), match="tl.tt|metadata|structured"):
        transform.VerifyTTComputeBlocks()(mod)


@pytest.mark.parametrize("kind", ["geometry", "writes", "reads", "indices", "free_var", "side_effect"])
def test_verifier_rejects_corrupted_semantics(kind):
    def change(fields):
        if kind == "geometry":
            fields["annotations"]["tl.tt.block_shape"] = [4, 1]
        elif kind == "writes":
            fields["writes"] = []
        elif kind == "reads":
            fields["reads"] = list(fields["reads"][:1])
        elif kind == "indices":
            store = fields["body"]
            fields["body"] = tirx.BufferStore(store.buffer, store.value, [0, 1])
        elif kind == "free_var":
            fields["annotations"]["tl.tt.logical_domain"] = [tirx.Var("removed_i", "int32"), 64]
        else:
            fields["body"] = tirx.Evaluate(tirx.call_extern("int32", "mutate"))

    mod = _change_compute_block(_canonicalize(_module()), change)
    with pytest.raises((tvm.error.TVMError, ValueError), match="T.Tiles|TT|tl.tt|structured|elementwise"):
        transform.VerifyTTComputeBlocks()(mod)


def _frontend_function():
    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((64, 64), T.float32, annotations=METADATA)
            b = T.alloc_shared((64, 64), T.float32, annotations=METADATA)
            c = T.alloc_shared((64, 64), T.float32, annotations=METADATA)
            for i, j in T.Tiles(c):
                c[i, j] = a[i, j] + b[i, j]

    return main


def test_real_frontend_canonicalizes_before_scalar_simplification():
    mod = tirx.transform.BindTarget(tvm.target.Target(TARGET))(tvm.IRModule({"main": _frontend_function()}))
    (block,) = _blocks(_canonicalize(mod))
    assert _integers(block.annotations["tl.tt.logical_domain"]) == [64, 64]
    assert all(_integers([dim.extent for dim in region.region]) == [64, 64] for region in [*block.reads, *block.writes])


def test_pipeline_rejects_uninitialized_compute_without_codegen_or_simt(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())

    def forbidden(*args, **kwargs):
        pytest.fail("Device Lower must not invoke SIMT or code generation")

    monkeypatch.setattr(transform, "LayoutInference", forbidden)
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    monkeypatch.setattr(type(context), "codegen_device", forbidden)
    with pytest.raises((ValueError, NotImplementedError), match="producer|initializ|dataflow|copies|read-before-write"):
        context.lower(tvm.IRModule({"main": _frontend_function()}))


def test_algebraically_equivalent_indices_are_accepted():
    mod = _canonicalize(_module(expression=lambda a, b, c, ij: a[ij[0] + 0, (ij[1] // 1) * 1]))
    (block,) = _blocks(mod)
    assert _effects_by_name(block.reads) == {"A": [64, 64]}
    transform.VerifyTTComputeBlocks()(mod)


def test_reject_serial_tiles_annotation():
    _reject(_module(parallel=False), "parallel|Phase 1")


def test_constant_only_expression_captures_fill_semantics():
    result = _canonicalize(_module(expression=lambda a, b, c, ij: tirx.const(2, "float32")))
    (block,) = _blocks(result)
    assert not block.reads
    assert float(block.body.value) == 2.0
    transform.VerifyTTComputeBlocks()(result)


@pytest.mark.parametrize("kind", ["stage", "domain", "min", "step", "inner_stage", "wrapper"])
def test_reject_malformed_frontend_loop_chain(kind):
    mod = _module()

    def change(node):
        if not isinstance(node, tirx.For) or "tl.tt.tiles_scope" not in node.annotations:
            return None
        fields = dict(
            loop_var=node.loop_var,
            min=node.min,
            extent=node.extent,
            kind=node.kind,
            body=node.body,
            annotations=dict(node.annotations),
            step=node.step,
        )
        if kind == "stage":
            fields["annotations"]["tl.tt.tiles_stage"] = 1
        elif kind == "domain":
            fields["annotations"]["tl.tt.tiles_domain"] = [32, 64]
        elif kind == "min":
            fields["min"] = 1
        elif kind == "step":
            fields["step"] = 2
        elif kind == "inner_stage":
            inner = node.body
            annotations = dict(inner.annotations)
            annotations["tl.tt.tiles_stage"] = 1
            fields["body"] = tirx.For(
                inner.loop_var, inner.min, inner.extent, inner.kind, inner.body, annotations=_annotation_value(annotations)
            )
        else:
            fields["body"] = tirx.AttrStmt(tirx.const(0), "test_wrapper", 1, node.body)
        fields["annotations"] = _annotation_value(fields["annotations"])
        return tirx.For(**fields)

    func = mod["main"]
    mod = tvm.IRModule({"main": func.with_body(tirx.stmt_functor.ir_transform(func.body, None, change))})
    _reject(mod, "stage|domain|extent|zero|step|chain|loop|Loop|Phase 1")


@pytest.mark.parametrize(
    "kind", ["missing_recipe", "broadcast_axes", "logical_region", "physical_shape", "block_shape", "access_map", "region"]
)
def test_verifier_rejects_corrupted_broadcast_recipe(kind):
    mod = _canonicalize(_module((64, 256), b_shape=(32, 256), expression=lambda a, b, c, ij: a[tuple(ij)] + b[0, ij[1]]))

    def change(fields):
        region = next(region for region in fields["reads"] if region.buffer.name == "B")
        data = region.buffer.data
        if kind == "access_map":
            maps = dict(fields["annotations"]["tl.tt.access_maps"])
            maps[data] = [1, 0]
            fields["annotations"]["tl.tt.access_maps"] = maps
        elif kind == "region":
            fields["reads"] = [
                tirx.BufferRegion(item.buffer, [ir.Range(0, 64), ir.Range(0, 256)]) if item.buffer.same_as(region.buffer) else item
                for item in fields["reads"]
            ]
        else:
            recipes = dict(fields["annotations"]["tl.tt.broadcast_recipes"])
            if kind == "missing_recipe":
                del recipes[data]
            else:
                recipe = dict(recipes[data])
                recipe[kind] = [1] if kind == "broadcast_axes" else [64, 256]
                recipes[data] = recipe
            fields["annotations"]["tl.tt.broadcast_recipes"] = recipes

    malformed = _change_compute_block(mod, change)
    with pytest.raises((tvm.error.TVMError, ValueError), match="T.Tiles|TT|tl.tt|structured|elementwise"):
        transform.VerifyTTComputeBlocks()(malformed)


def test_verifier_rejects_uncanonicalized_frontend_scope():
    with pytest.raises((tvm.error.TVMError, ValueError), match="scope|canonical|frontend|T.Tiles"):
        transform.VerifyTTComputeBlocks()(_module())


def test_orphan_frontend_annotations_cannot_bypass_capture_or_verification():
    def remove_scope(node):
        if isinstance(node, tirx.For) and "tl.tt.tiles_scope" in node.annotations:
            annotations = dict(node.annotations)
            del annotations["tl.tt.tiles_scope"]
            return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body, annotations=annotations)
        return None

    mod = _module()
    func = mod["main"]
    malformed = tvm.IRModule({"main": func.with_body(tirx.stmt_functor.ir_transform(func.body, None, remove_scope))})
    _reject(malformed, "orphan|frontend|scope")
    with pytest.raises(ValueError, match="frontend|scope"):
        transform.VerifyTTComputeBlocks()(malformed)


def test_structured_annotations_require_compute_kind():
    mod = _change_compute_block(_canonicalize(_module()), lambda fields: fields["annotations"].pop("tl.tt.compute_kind"))
    with pytest.raises(ValueError, match="missing metadata tl.tt.compute_kind"):
        transform.VerifyTTComputeBlocks()(mod)


@pytest.mark.parametrize("field", ["elem_offset", "strides"])
def test_storage_descriptors_cannot_retain_removed_coordinate_binders(field):
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    canceled_coordinate = tirx.Sub(i, i)
    options = {"elem_offset": canceled_coordinate} if field == "elem_offset" else {"strides": [tirx.Add(64, canceled_coordinate), 1]}
    a = tirx.decl_buffer((64, 64), "float32", name="A", scope="shared", **options)
    c = tirx.decl_buffer((64, 64), "float32", name="C", scope="shared")
    body = _scope((64, 64), [i, j], tirx.BufferStore(c, a[i, j], [i, j]))
    root = tirx.SBlock(
        [], [], [], "root", body, alloc_buffers=[a, c], annotations={ALLOCATIONS: _annotation_value({a.data: METADATA, c.data: METADATA})}
    )
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr("target", tvm.target.Target(TARGET))
    _reject(tvm.IRModule({"main": func}), "storage.*loop variables")


@pytest.mark.parametrize("field", ["elem_offset", "strides"])
def test_verifier_rejects_free_variables_in_storage_descriptors(field):
    def change(fields):
        load = fields["body"].value.a
        variable = tirx.Var("removed_coordinate", "int32")
        canceled = tirx.Sub(variable, variable)
        options = {"elem_offset": canceled} if field == "elem_offset" else {"strides": [tirx.Add(64, canceled), 1]}
        buffer = tirx.decl_buffer(load.buffer.shape, load.buffer.dtype, scope="shared", data=load.buffer.data, **options)
        store = fields["body"]
        fields["body"] = tirx.BufferStore(store.buffer, buffer[0, 0] + store.value.b, store.indices)

    malformed = _change_compute_block(_canonicalize(_module()), change)
    with pytest.raises(ValueError, match="storage.*loop variables"):
        transform.VerifyTTComputeBlocks()(malformed)


@pytest.mark.parametrize("kind", ["call", "cast"])
def test_reject_expression_annotations_containing_removed_binders(kind):
    def expression(a, b, c, ij):
        if kind == "call":
            return tirx.Call("float32", ir.Op.get("tirx.exp"), [a[tuple(ij)]], annotations={"user_axis": ij[0]})
        return tirx.Cast("float32", a[tuple(ij)], annotations={"user_axis": ij[0]})

    _reject(_module(expression=expression), "annotations|schema")


@pytest.mark.parametrize("options", [{"strides": [0, 1]}, {"strides": [128, 1]}, {"elem_offset": 1}, {"axis_separators": [1]}])
def test_reject_unsupported_storage_views(options):
    _reject(_module(a_buffer_options=options), "compact|offset|strided|alias")


def test_explicit_compact_strides_are_supported():
    (block,) = _blocks(_canonicalize(_module(a_buffer_options={"strides": [64, 1]})))
    assert _effects_by_name(block.reads) == {"A": [64, 64], "B": [64, 64]}


def test_reject_distinct_buffer_views_sharing_data_identity():
    def expression(a, b, c, ij):
        alias = tirx.decl_buffer(a.shape, a.dtype, name="A_alias", data=a.data, scope="shared")
        return a[tuple(ij)] + alias[tuple(ij)]

    _reject(_module(expression=expression), "alias|Buffer object")


def test_int64_domain_preserves_index_width_and_tile_geometry():
    shape = (tirx.IntImm("int64", 64), tirx.IntImm("int64", 256))
    mod = _canonicalize(_module(shape, loop_dtype="int64"))
    (block,) = _blocks(mod)
    assert _integers(block.annotations["tl.tt.logical_domain"]) == [64, 256]
    assert _integers(block.annotations["tl.tt.block_shape"]) == [2, 8]
    for index in block.body.indices:
        assert index.dtype == "int64" and int(index) == 0
    for load in _nodes(mod["main"], tirx.BufferLoad):
        assert all(index.dtype == "int64" and int(index) == 0 for index in load.indices)
    transform.VerifyTTComputeBlocks()(mod)


def test_reject_single_alias_that_differs_from_its_allocation():
    def expression(a, b, c, ij):
        alias = tirx.decl_buffer((64, 64), a.dtype, name="A_alias", data=a.data, scope="shared")
        return alias[tuple(ij)]

    _reject(_module(a_shape=(32, 32), expression=expression), "alias|allocation|Buffer object")


@pytest.mark.parametrize("kind", ["extern", "gather"])
def test_reject_effectful_or_gather_indices_even_when_algebraically_cancelled(kind):
    def expression(a, b, c, ij):
        index_value = tirx.call_extern("int32", "side_effect") if kind == "extern" else tirx.Cast("int32", b[0, 0])
        # Keep the original expression intact until the pass's analyzer sees it.
        index = tirx.Add(ij[0], tirx.Mul(tirx.const(0), index_value))
        return a[index, ij[1]]

    _reject(_module(expression=expression), "indices|access|pure|gather|side.effect|Phase 1")


def test_reject_tiles_bound_to_another_target():
    mod = _module()
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target("llvm"))})
    _reject(mod, "Tenstorrent target|tenstorrent")


def test_unrelated_ir_is_unchanged():
    func = tirx.PrimFunc([], tirx.Evaluate(0)).with_attr("target", tvm.target.Target("llvm"))
    mod = tvm.IRModule({"main": func})
    ir.assert_structural_equal(mod, _canonicalize(mod))
    ir.assert_structural_equal(mod, transform.VerifyTTComputeBlocks()(mod))


def test_elementwise_pass_public_factories_and_registration():
    from tilelang.tenstorrent import transform as tt_transform

    name = "CanonicalizeTTElementwise"
    for module in (transform, tt_transform):
        assert getattr(module, name)().info.name == "tl.tenstorrent." + name
        for obsolete in ("CanonicalizeTTTiles", "CanonicalizeTTParallel"):
            assert not hasattr(module, obsolete)
            assert tvm.get_global_func("tl.tenstorrent.transform." + obsolete, allow_missing=True) is None
    assert name in tt_transform.__all__
    assert tvm.get_global_func("tl.tenstorrent.transform." + name)().info.name == "tl.tenstorrent." + name


def _parallel_module(mod):
    """Express the same iteration domain with ordinary Parallel loops."""

    def change(node):
        if isinstance(node, tirx.For) and "tl.tt.tiles_stage" in node.annotations:
            return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.PARALLEL, node.body)
        return None

    func = mod["main"]
    return tvm.IRModule({"main": func.with_body(tirx.stmt_functor.ir_transform(func.body, None, change))})


def test_reject_parallel_bound_to_another_target():
    mod = _parallel_module(_module())
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target("llvm"))})
    _reject(mod, "Tenstorrent target|tenstorrent")


@pytest.mark.parametrize("outer_frontend", ["tiles", "parallel"])
def test_reject_mixed_frontends_within_one_loop_nest(outer_frontend):
    mod = _module()

    def change(node):
        if isinstance(node, tirx.For) and ("tl.tt.tiles_scope" in node.annotations) == (outer_frontend == "parallel"):
            return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.PARALLEL, node.body)
        return None

    func = mod["main"]
    malformed = tvm.IRModule({"main": func.with_body(tirx.stmt_functor.ir_transform(func.body, None, change))})
    _reject(malformed, "loop|annotation|scope|rank|Parallel|Tiles")


@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (128, 32)])
@pytest.mark.parametrize(
    "expression",
    [
        lambda a, b, c, ij: a[tuple(ij)] + b[tuple(ij)],
        lambda a, b, c, ij: a[tuple(ij)] * b[tuple(ij)] + a[tuple(ij)],
        lambda a, b, c, ij: a[tuple(ij)] - b[tuple(ij)],
        lambda a, b, c, ij: a[tuple(ij)] * tirx.const(2, "float32"),
    ],
)
def test_parallel_and_tiles_have_the_same_structured_semantics(shape, expression):
    raw = _module(shape, expression=expression)
    tiles = _canonicalize(raw)
    parallel = transform.CanonicalizeTTElementwise()(_parallel_module(raw))
    ir.assert_structural_equal(tiles, parallel)
    transform.VerifyTTComputeBlocks()(parallel)
    ir.assert_structural_equal(parallel, transform.CanonicalizeTTElementwise()(parallel))


@pytest.mark.parametrize("kind", ["identity", "compound", "row", "column", "scalar", "inplace"])
@pytest.mark.parametrize("frontend", ["tiles", "parallel"])
def test_structured_expression_and_access_maps_match_numpy(kind, frontend):
    import numpy as np

    shape = (64, 64)
    b_shape = {"row": (32, 64), "column": (64, 32), "scalar": (32, 32)}.get(kind, shape)

    def expression(a, b, c, ij):
        lhs = a[tuple(ij)]
        if kind == "row":
            return lhs + b[0, ij[1]]
        if kind == "column":
            return lhs + b[ij[0], 0]
        if kind == "scalar":
            return lhs + b[0, 0]
        if kind == "compound":
            return lhs * b[tuple(ij)] + lhs
        if kind == "inplace":
            return c[tuple(ij)] + lhs
        return lhs + b[tuple(ij)]

    raw = _module(shape, b_shape=b_shape, expression=expression)
    lowered = _canonicalize(raw) if frontend == "tiles" else transform.CanonicalizeTTElementwise()(_parallel_module(raw))
    (block,) = _blocks(lowered)
    rng = np.random.default_rng(42)
    arrays = {
        region.buffer.name: rng.standard_normal(_integers(region.buffer.shape)).astype("float32")
        for region in [*block.reads, *block.writes]
    }
    coordinates = np.indices(shape)

    def evaluate(expr):
        if isinstance(expr, tirx.BufferLoad):
            axes = _integers(block.annotations["tl.tt.access_maps"][expr.buffer.data])
            indices = tuple(0 if axis == -1 else coordinates[axis] for axis in axes)
            return arrays[expr.buffer.name][indices]
        if isinstance(expr, tirx.Add):
            return evaluate(expr.a) + evaluate(expr.b)
        if isinstance(expr, tirx.Mul):
            return evaluate(expr.a) * evaluate(expr.b)
        raise AssertionError(f"Unexpected expression {expr}")

    a, b = arrays["A"], arrays.get("B")
    if kind == "compound":
        expected = a * b + a
    elif kind == "row":
        expected = a + b[0:1, :]
    elif kind == "column":
        expected = a + b[:, 0:1]
    elif kind == "scalar":
        expected = a + b[0, 0]
    elif kind == "inplace":
        expected = arrays["C"] + a
    else:
        expected = a + b
    np.testing.assert_array_equal(evaluate(block.body.value), expected)


def _io_function(frontend, size=32, compound=False, mixed=False, explicit_metadata=True, broadcast=False):
    @T.prim_func
    def main(
        A: T.Tensor((size, size), T.float32),
        B: T.Tensor((32 if broadcast else size, size), T.float32),
        C: T.Tensor((size, size), T.float32),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((size, size), T.float32, annotations=METADATA if explicit_metadata else {})
            b = T.alloc_shared((32 if broadcast else size, size), T.float32, annotations=METADATA if explicit_metadata else {})
            c = T.alloc_shared((size, size), T.float32, annotations=METADATA if explicit_metadata else {})
            T.copy(A, a)
            T.copy(B, b)
            if frontend == "tiles":
                for i, j in T.Tiles(c):
                    if broadcast:
                        c[i, j] = a[i, j] + b[0, j]
                    elif compound:
                        c[i, j] = a[i, j] * b[i, j] + a[i, j]
                    else:
                        c[i, j] = a[i, j] + b[i, j]
            else:
                for i, j in T.Parallel(size, size):
                    if broadcast:
                        c[i, j] = a[i, j] + b[0, j]
                    elif compound:
                        c[i, j] = a[i, j] * b[i, j] + a[i, j]
                    else:
                        c[i, j] = a[i, j] + b[i, j]
            if mixed:
                if frontend == "tiles":
                    for i, j in T.Parallel(size, size):
                        c[i, j] = c[i, j] * a[i, j]
                else:
                    for i, j in T.Tiles(c):
                        c[i, j] = c[i, j] * a[i, j]
            T.copy(c, C)

    return main


@pytest.mark.parametrize("size", [32, 64])
@pytest.mark.parametrize("source", ["call", "load"])
@pytest.mark.parametrize("inner", [False, True])
def test_pipeline_rejects_canceled_effects_in_original_tiles_extent(size, source, inner):
    from tilelang.tenstorrent.pipeline import lower_tenstorrent_ir

    func = _replace_tiles_loop_extent(_io_function("tiles", size=size), source, inner)
    mod = tvm.IRModule({"main": func})
    before = ir.save_json(mod)
    with pytest.raises(ValueError, match="loop extent.*compile-time constant"):
        lower_tenstorrent_ir(mod, tvm.target.Target(TARGET))
    assert ir.save_json(mod) == before


@pytest.mark.parametrize("size", [32, 64])
def test_pipeline_accepts_static_original_tiles_extents(size):
    from tilelang.tenstorrent.pipeline import lower_tenstorrent_ir

    mod = tvm.IRModule({"main": _io_function("tiles", size=size)})
    lowered = lower_tenstorrent_ir(mod, tvm.target.Target(TARGET))
    assert "tt.device_ir_version" in lowered.attrs
    assert "tt.ir_stage" not in lowered.attrs


@pytest.mark.parametrize("first_frontend", ["tiles", "parallel"])
def test_one_elementwise_pass_captures_mixed_scopes_and_is_idempotent(first_frontend):
    raw = tvm.IRModule({"main": _io_function(first_frontend, size=64, mixed=True)})
    raw = tirx.transform.BindTarget(tvm.target.Target(TARGET))(raw)
    lowered = _canonicalize(raw)
    first, second = _blocks(lowered)
    assert isinstance(first.body.value, tirx.Add)
    assert isinstance(second.body.value, tirx.Mul)
    assert first.writes[0].buffer.same_as(second.writes[0].buffer)
    assert any(region.buffer.same_as(first.writes[0].buffer) for region in second.reads)
    assert all(
        node.kind != tirx.ForKind.PARALLEL and "tl.tt.tiles_scope" not in node.annotations for node in _nodes(lowered["main"], tirx.For)
    )
    transform.VerifyTTComputeBlocks()(lowered)
    ir.assert_structural_equal(lowered, _canonicalize(lowered))


@pytest.mark.parametrize("size", [32, 64])
def test_pipeline_executes_elementwise_once_before_verification_and_consumption(monkeypatch, size):
    from tilelang.tenstorrent.pipeline import TENSTORRENT_LOWER_PASS_ORDER

    @tvm.instrument.pass_instrument
    class RecordPasses:
        def __init__(self):
            self.names = []
            self.top_level_names = []
            self.depth = 0

        def run_before_pass(self, mod, info):
            name = info.name.rsplit(".", 1)[-1]
            self.names.append(name)
            if self.depth == 0:
                self.top_level_names.append(name)
            self.depth += 1

        def run_after_pass(self, mod, info):
            self.depth -= 1

    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    recorder = RecordPasses()
    with tvm.transform.PassContext(instruments=[recorder]):
        lowered = context.lower(tvm.IRModule({"main": _io_function("tiles", size=size)}))
    # Legalization internally verifies and runs a nested PrimFunc pass; record
    # backend pipeline stages separately from those implementation details.
    names = [name for name in recorder.top_level_names if name in TENSTORRENT_LOWER_PASS_ORDER]
    expected = list(TENSTORRENT_LOWER_PASS_ORDER)
    assert "tt.device_ir_version" in lowered.attrs
    assert "tt.ir_stage" not in lowered.attrs
    assert names == expected
    assert recorder.names.count("CanonicalizeTTElementwise") == 1


def test_parallel_and_tiles_add_produce_equivalent_device_ir(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    parallel = context.lower(tvm.IRModule({"main": _io_function("parallel")}))
    tiles = context.lower(tvm.IRModule({"main": _io_function("tiles")}))
    ir.assert_structural_equal(parallel, tiles)
    assert any(
        isinstance(call.op, ir.Op) and call.op.name == "tl.tt.dfb_add"
        for func in tiles.functions.values()
        for call in _nodes(func, tirx.Call)
    )
    assert all(
        not any("tl.tt.compute_kind" in block.annotations for block in _nodes(func, tirx.SBlock)) for func in tiles.functions.values()
    )


@pytest.mark.parametrize("frontend", ["tiles", "parallel"])
@pytest.mark.parametrize("size,compound,mixed", [(64, False, False), (32, True, False), (64, True, False), (32, False, True)])
def test_pipeline_consumes_complete_compute_dataflow(monkeypatch, frontend, size, compound, mixed):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    lowered = context.lower(tvm.IRModule({"main": _io_function(frontend, size, compound, mixed)}))
    assert int(lowered.attrs["tt.device_ir_version"]) == 2
    assert "tt.ir_stage" not in lowered.attrs
    calls = [call for func in lowered.functions.values() for call in _nodes(func, tirx.Call) if isinstance(call.op, ir.Op)]
    computes = [call for call in calls if call.op.name == "tl.tt.dfb_compute" and str(call.annotations["tt.compute_kind"].value) != "copy"]
    assert len(computes) == (2 if mixed else 1)
    assert all(_integers(call.annotations["tt.logical_domain"]) == [size, size] for call in computes)
    assert not any(call.op.name.startswith("tl.tileop.") for call in calls)
    from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR

    VerifyTenstorrentDeviceIR()(lowered)


def test_pipeline_parallel_default_allocation_metadata_reaches_device(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    lowered = context.lower(tvm.IRModule({"main": _io_function("parallel", explicit_metadata=False)}))
    assert int(lowered.attrs["tt.device_ir_version"]) == 2
    assert "tt.ir_stage" not in lowered.attrs
    from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR

    VerifyTenstorrentDeviceIR()(lowered)


def test_pipeline_rejects_multiple_operations_atomically(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    raw = tvm.IRModule(
        {
            "small": _io_function("parallel", 32).with_attr("global_symbol", "small"),
            "large": _io_function("tiles", 64).with_attr("global_symbol", "large"),
        }
    )
    before = ir.save_json(raw)
    with pytest.raises(NotImplementedError, match="multiple frontend PrimFuncs"):
        context.lower(raw)
    assert ir.save_json(raw) == before


def test_standalone_uninitialized_compute_is_not_device_success(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")

    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), T.float32, annotations=METADATA)
            b = T.alloc_shared((32, 32), T.float32, annotations=METADATA)
            c = T.alloc_shared((32, 32), T.float32, annotations=METADATA)
            for i, j in T.Tiles(c):
                c[i, j] = a[i, j] + b[i, j]

    with pytest.raises((ValueError, NotImplementedError), match="producer|initializ|dataflow|copies|read-before-write"):
        context.lower(tvm.IRModule({"main": main}))


@pytest.mark.parametrize("outer", [True, False])
def test_reject_unrecognized_frontend_loop_annotation_schema(outer):
    mod = _module()

    def change(node):
        if not isinstance(node, tirx.For):
            return None
        if ("tl.tt.tiles_scope" in node.annotations) != outer:
            return None
        annotations = dict(node.annotations)
        annotations["unknown_hint" if outer else "tl.tt.tiles_domain"] = 1 if outer else [64, 64]
        return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body, annotations=_annotation_value(annotations))

    func = mod["main"]
    malformed = tvm.IRModule({"main": func.with_body(tirx.stmt_functor.ir_transform(func.body, None, change))})
    _reject(malformed, "annotation.*schema|schema.*annotation")


def test_reject_allocation_metadata_without_owning_allocation():
    mod = _module()
    func = mod["main"]
    root = func.body.block
    annotations = dict(root.annotations)
    entries = dict(annotations[ALLOCATIONS])
    orphan = tirx.decl_buffer((64, 64), "float32", scope="shared", name="orphan")
    entries[orphan.data] = METADATA
    annotations[ALLOCATIONS] = entries
    block = tirx.SBlock(
        root.iter_vars,
        root.reads,
        root.writes,
        root.name_hint,
        root.body,
        alloc_buffers=root.alloc_buffers,
        annotations=_annotation_value(annotations),
    )
    malformed = tvm.IRModule({"main": func.with_body(tirx.SBlockRealize([], True, block))})
    _reject(malformed, "metadata.*(allocation|owner)|allocation.*metadata")


def test_two_single_tile_adds_reach_device_stage(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    func = _io_function("parallel")

    def duplicate_compute(node):
        if (
            isinstance(node, tirx.For)
            and node.kind == tirx.ForKind.PARALLEL
            and isinstance(node.body, tirx.For)
            and node.body.kind == tirx.ForKind.PARALLEL
        ):
            return tirx.SeqStmt([node, node])
        return None

    func = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, duplicate_compute), func.span)
    lowered = context.lower(tvm.IRModule({"main": func}))
    assert int(lowered.attrs["tt.device_ir_version"]) == 2
    assert "tt.ir_stage" not in lowered.attrs
    from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR

    VerifyTenstorrentDeviceIR()(lowered)


@pytest.mark.parametrize("frontend", ["tiles", "parallel"])
@pytest.mark.parametrize("size", [32, 64])
def test_pipeline_preserves_padded_broadcast_recipe(monkeypatch, frontend, size):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    lowered = context.lower(tvm.IRModule({"main": _io_function(frontend, size, broadcast=True)}))
    assert int(lowered.attrs["tt.device_ir_version"]) == 2
    computes = [
        call
        for func in lowered.functions.values()
        for call in _nodes(func, tirx.Call)
        if isinstance(call.op, ir.Op)
        and call.op.name == "tl.tt.dfb_compute"
        and str(call.annotations["tt.compute_kind"].value) == "elementwise"
    ]
    assert len(computes) == 1
    assert [-1, 1] in [_integers(axes) for axes in computes[0].annotations["tt.access_maps"]]
    assert _integers(computes[0].annotations["tt.logical_domain"]) == [size, size]
    from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR

    VerifyTenstorrentDeviceIR()(lowered)


def test_invalid_frontend_access_reports_source_and_buffer():
    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), T.float32, annotations=METADATA)
            c = T.alloc_shared((32, 32), T.float32, annotations=METADATA)
            for i, j in T.Tiles(c):
                c[i, j] = a[j, i]

    mod = tirx.transform.BindTarget(tvm.target.Target(TARGET))(tvm.IRModule({"main": main}))
    with pytest.raises(ValueError) as error:
        _canonicalize(mod)
    assert "test_tilelang_tenstorrent_tiles_transform.py" in str(error.value)
    assert "buffer 'a'" in str(error.value)


def test_deferred_compute_cannot_bypass_topology_validation(monkeypatch):
    from testing.python.target.test_tilelang_tenstorrent_phase0_contract import FRONTEND_PROGRAMS

    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(TARGET, target_host="c", execution_backend="ttnn")
    raw = tvm.IRModule(
        {
            "compute": _io_function("tiles", 64).with_attr("global_symbol", "compute"),
            "p2p": FRONTEND_PROGRAMS["p2p"],
        }
    )
    with pytest.raises(NotImplementedError, match="multiple frontend PrimFuncs"):
        context.lower(raw)


@pytest.mark.parametrize("a_scope,b_scope", [("shared", "shared"), ("local.fragment", "local.fragment"), ("shared", "local.fragment")])
@pytest.mark.parametrize("c_scope", ["shared", "local.fragment"])
@pytest.mark.parametrize("flat_allocations", [False, True])
def test_tiles02_scope_combinations_preserve_value_kinds(a_scope, b_scope, c_scope, flat_allocations):
    source = _module(a_scope=a_scope, b_scope=b_scope, c_scope=c_scope, flat_allocations=flat_allocations)
    mod = _canonicalize(source)
    ir.assert_structural_equal(mod, _canonicalize(mod))
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    before_verify = ir.save_json(restored)
    transform.VerifyTTComputeBlocks()(restored)
    assert ir.save_json(restored) == before_verify, "The verifier must be read-only"
    (block,) = _blocks(restored)
    kinds = block.annotations["tl.tt.value_kinds"]
    assert len(kinds) == 3
    assert _effects_by_name(block.reads) == {"A": [64, 64], "B": [64, 64]}
    assert _effects_by_name(block.writes) == {"C": [64, 64]}
    allocations = (
        [node.buffer for node in _nodes(restored["main"], tirx.AllocBuffer)]
        if flat_allocations
        else next(node for node in _nodes(restored["main"], tirx.SBlock) if ALLOCATIONS in node.annotations).alloc_buffers
    )
    by_name = {buffer.name: buffer for buffer in allocations}
    for effect in [*block.reads, *block.writes]:
        buffer = effect.buffer
        assert buffer.same_as(by_name[buffer.name])
        assert str(kinds[buffer.data]) == ("compute_fragment" if buffer.scope() == "local.fragment" else "shared_dfb")
        assert _integers(block.annotations["tl.tt.access_maps"][buffer.data]) == [0, 1]
        assert buffer.dtype == "float32"
    assert _integers(block.annotations["tl.tt.block_shape"]) == [2, 2]
    assert not _nodes(restored["main"], tirx.For)
    lowered = transform.LowerOpaqueBlock()(restored)
    for allocation in _nodes(lowered["main"], tirx.AllocBuffer):
        if allocation.buffer.scope() == "local.fragment":
            assert not allocation.annotations
        else:
            assert int(allocation.annotations["tt.dfb_block_count"]) == 2


@pytest.mark.parametrize("carrier_scope", ["shared", "local.fragment"])
def test_tiles02_domain_carrier_does_not_create_effects_or_require_metadata(carrier_scope):
    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1):
            carrier = T.alloc_buffer((64, 64), T.float32, scope=carrier_scope)
            source = T.alloc_shared((64, 64), T.float32, annotations=METADATA)
            output = T.alloc_fragment((64, 64), T.float32)
            for i, j in T.Tiles(carrier):
                output[i, j] = source[i, j]

    mod = _canonicalize(tirx.transform.BindTarget(tvm.target.Target(TARGET))(tvm.IRModule({"main": main})))
    transform.VerifyTTComputeBlocks()(mod)
    (block,) = _blocks(mod)
    assert _effects_by_name(block.reads) == {"source": [64, 64]}
    assert _effects_by_name(block.writes) == {"output": [64, 64]}
    assert len(block.annotations["tl.tt.value_kinds"]) == 2
    assert _integers(block.annotations["tl.tt.logical_domain"]) == [64, 64]


def test_tiles02_fragment_does_not_require_dfb_metadata_but_shared_input_does():
    fragments = _module(a_scope="local.fragment", b_scope="local.fragment", c_scope="local.fragment", metadata=False)
    transform.VerifyTTComputeBlocks()(_canonicalize(fragments))
    _reject(_module(b_scope="local.fragment", c_scope="local.fragment", metadata=False), "metadata")
    _reject(
        _module(b_scope="local.fragment", c_scope="local.fragment", a_metadata={**METADATA, "tt.tile_shape": [16, 32]}),
        "tt.tile_shape|32x32",
    )


@pytest.mark.parametrize("operand", ["a_scope", "b_scope", "c_scope"])
def test_tiles02_parallel_rejects_rank3_fragment_operands(operand):
    mod = _parallel_module(_module((2, 32, 32), **{operand: "local.fragment"}))
    _reject(mod, "fragment.*rank-2")


@pytest.mark.parametrize("metadata", [{"tt.tile_shape": [32, 32]}, {"tt.dfb_block_count": 2}, {"tt.tensor_backed": 1}])
def test_tiles02_fragment_rejects_dfb_allocation_metadata(metadata):
    _reject(_module(a_scope="local.fragment", a_metadata=metadata), "fragment|metadata")


@pytest.mark.parametrize("count", [1, 3, 32])
def test_tiles02_shared_capacity_does_not_change_fragment_compute_geometry(count):
    source = _module(b_scope="local.fragment", c_scope="local.fragment", a_metadata={**METADATA, "tt.dfb_block_count": count})
    mod = _canonicalize(source)
    transform.VerifyTTComputeBlocks()(mod)
    (block,) = _blocks(mod)
    assert _integers(block.annotations["tl.tt.logical_domain"]) == [64, 64]
    assert _integers(block.annotations["tl.tt.physical_tile_shape"]) == [32, 32]
    assert _integers(block.annotations["tl.tt.block_shape"]) == [2, 2]


def test_tiles02_fragment_inplace_update_preserves_old_read_and_new_write():
    mod = _canonicalize(
        _module(c_scope="local.fragment", expression=lambda a, b, c, ij: c[tuple(ij)] * tirx.const(2, "float32") + a[tuple(ij)])
    )
    transform.VerifyTTComputeBlocks()(mod)
    (block,) = _blocks(mod)
    assert set(_effects_by_name(block.reads)) == {"A", "C"}
    assert set(_effects_by_name(block.writes)) == {"C"}
    previous = next(region.buffer for region in block.reads if region.buffer.name == "C")
    assert previous.same_as(block.writes[0].buffer)
    assert str(block.annotations["tl.tt.value_kinds"][previous.data]) == "compute_fragment"
    assert len(block.annotations["tl.tt.value_kinds"]) == 2


def test_tiles02_fragment_to_shared_preserves_explicit_precision_conversion():
    mod = _canonicalize(
        _module(a_scope="local.fragment", c_dtype="bfloat16", expression=lambda a, b, c, ij: tirx.Cast("bfloat16", a[tuple(ij)]))
    )
    transform.VerifyTTComputeBlocks()(mod)
    (block,) = _blocks(mod)
    assert isinstance(block.body.value, tirx.Cast)
    assert block.body.value.dtype == "bfloat16"
    assert block.body.value.value.dtype == "float32"
    assert block.reads[0].buffer.dtype == "float32"
    assert block.writes[0].buffer.dtype == "bfloat16"
    assert len(block.annotations["tl.tt.value_kinds"]) == 2


@pytest.mark.parametrize(
    "kind,b_shape,axes,logical",
    [("row", (32, 128), [-1, 1], [1, 128]), ("column", (64, 32), [0, -1], [64, 1]), ("scalar", (32, 32), [-1, -1], [1, 1])],
)
@pytest.mark.parametrize("c_scope", ["shared", "local.fragment"])
def test_tiles02_fragment_broadcast_preserves_compact_valid_region(kind, b_shape, axes, logical, c_scope):
    def expression(a, b, c, ij):
        return a[tuple(ij)] + b[tuple(ij[axis] if axis >= 0 else 0 for axis in axes)]

    mod = _canonicalize(_module((64, 128), b_shape=b_shape, b_scope="local.fragment", c_scope=c_scope, expression=expression))
    transform.VerifyTTComputeBlocks()(mod)
    (block,) = _blocks(mod)
    operand = next(region for region in block.reads if region.buffer.name == "B")
    assert _integers([dim.extent for dim in operand.region]) == logical
    assert _integers(operand.buffer.shape) == list(b_shape)
    assert str(block.annotations["tl.tt.value_kinds"][operand.buffer.data]) == "compute_fragment"
    assert _integers(block.annotations["tl.tt.access_maps"][operand.buffer.data]) == axes
    recipe = block.annotations["tl.tt.broadcast_recipes"][operand.buffer.data]
    assert str(recipe["broadcast_kind"]) == kind
    assert _integers(recipe["logical_region"]) == logical


@pytest.mark.parametrize("corruption", ["missing_operand", "extra_operand", "wrong_kind", "wrong_schema", "wrong_identity"])
def test_tiles02_verifier_rejects_forged_value_kinds(corruption):
    def change(fields):
        kinds = dict(fields["annotations"]["tl.tt.value_kinds"])
        fragment = fields["writes"][0].buffer
        if corruption == "missing_operand":
            kinds.pop(fragment.data)
        elif corruption == "extra_operand":
            kinds[tirx.Var("unrelated", "handle")] = "compute_fragment"
        elif corruption == "wrong_kind":
            kinds[fragment.data] = "shared_dfb"
        elif corruption == "wrong_schema":
            kinds[fragment.data] = 1
        else:
            kinds.pop(fragment.data)
            kinds[tirx.Var(fragment.data.name, "handle")] = "compute_fragment"
        fields["annotations"]["tl.tt.value_kinds"] = kinds

    mod = _change_compute_block(_canonicalize(_module(c_scope="local.fragment")), change)
    before_verify = ir.save_json(mod)
    with pytest.raises((tvm.error.TVMError, ValueError), match="value_kinds|metadata|schema|structured"):
        transform.VerifyTTComputeBlocks()(mod)
    assert ir.save_json(mod) == before_verify
