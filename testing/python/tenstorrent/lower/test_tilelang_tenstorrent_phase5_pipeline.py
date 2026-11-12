"""Structured pipeline lowering and transaction scheduling, without TT toolchains."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import device_ir as D
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir, tirx
from testing.python.tenstorrent._support.reference import _calls


def pipeline_shared(block_count=2):
    annotations = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": block_count}
    return T.alloc_shared((32, 32), "float32", annotations=annotations)


def make_pipeline(stages=2, extent=5, start=0, block_count=None, inout=False, wait_policy="delayed"):
    annotations = {"tt.pipeline_wait_policy": wait_policy} if stages else {}
    capacity = block_count if block_count is not None else min(max(stages, 1), max(extent, 1))

    @T.prim_func
    def program(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = pipeline_shared(capacity)
            b = pipeline_shared(capacity)
            c = pipeline_shared(capacity)
            for k in T.Pipelined(start, start + extent, num_stages=stages, annotations=annotations):
                T.copy(A, a)
                T.copy(B, b)
                for i, j in T.Tiles(c):
                    c[i, j] = a[i, j] + b[i, j] + T.Cast("float32", k)
                if inout:
                    T.copy(c, A)
                else:
                    T.copy(c, C)

    return program


def lower_pipeline(func, arch="wormhole_b0"):
    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    return D.expand_current_device_tables(TenstorrentPassPipelineBody(tvm.IRModule({"main": func}), target))


def slot_calls(mod, slot):
    func = next(f for f in mod.functions.values() if str(f.attrs["tt.kernel_slot"]) == slot)
    return list(_calls(func.body))


def lexical_slot_calls(mod, slot):
    func = next(f for f in mod.functions.values() if str(f.attrs["tt.kernel_slot"]) == slot)
    found = []

    def visit(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            found.append(node.value)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


def slot_loops(mod, slot):
    func = next(f for f in mod.functions.values() if str(f.attrs["tt.kernel_slot"]) == slot)
    found = []
    tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: found.append(node) if isinstance(node, tirx.For) else None,
    )
    return found


@pytest.mark.parametrize("stages,extent,start", [(1, 1, 0), (1, 5, 0), (2, 5, 0), (3, 7, 4), (8, 2, 3), (32, 1, 0)])
@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_pipeline_preserves_structured_transaction_epochs(stages, extent, start, arch):
    func = make_pipeline(stages, extent, start)
    mod = lower_pipeline(func, arch)
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    assert int(mod.attrs["tt.pipeline_stages"]) == stages
    assert int(mod.attrs["tt.pipeline_extent"]) == extent
    depth = min(stages, extent)
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    dfbs = mod.attrs["tt.dfb_table"]
    assert len(dfbs) == 4  # A, B, compute output, export snapshot lexical sites.
    assert [int(group.storage_group_id) for group in groups] == list(range(len(dfbs)))
    for desc in dfbs:
        assert int(desc.block_count) == depth
        assert int(desc.transaction_count_or_loop_relation) == extent
    for slot in ("trisc", "ncrisc"):
        loops = slot_loops(mod, slot)
        assert [(int(loop.min), int(loop.extent)) for loop in loops] == [(start, extent)]
        assert all(not str(loop.loop_var.name).startswith("tt_compact_") for loop in loops)
    assert not slot_loops(mod, "brisc")
    transfers = lexical_slot_calls(mod, "ncrisc")
    issue = "tl.tt.tensor_to_dfb_nd"
    wait = "tl.tt.dfb_copy_wait"
    first_wait = next(i for i, call in enumerate(transfers) if call.op.name == wait)
    assert sum(call.op.name == issue for call in transfers[:first_wait]) == 1
    assert sum(call.op.name == issue for call in transfers) == 2
    calls = transfers + lexical_slot_calls(mod, "trisc")
    for call in calls:
        if call.op.name == "tl.tt.dfb_compute" and "tt.expression" in call.annotations:
            variables = []
            tirx.stmt_functor.post_order_visit(
                call.annotations["tt.expression"],
                lambda node, variables=variables: variables.append(node) if isinstance(node, tirx.Var) else None,
            )
            assert variables, "pipeline iteration scalar must remain bound by Device For"
    assert sum(call.op.name == "tl.tt.dfb_release" for call in calls) == len(dfbs)
    assert sum(call.op.name == wait for call in calls) == 3
    repeated = lower_pipeline(func, arch)
    ir.assert_structural_equal(mod, repeated)
    assert mod.script() == repeated.script()
    assert ir.save_json(mod) == ir.save_json(repeated)
    ir.assert_structural_equal(mod, ir.load_json(ir.save_json(mod)))
    ir.assert_structural_equal(mod, VerifyTenstorrentDeviceIR()(mod))


@pytest.mark.parametrize("stages", [0, 1, 2, 3])
@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_loop_plan_keeps_input_immutable_and_reusable(stages, arch):
    """Lowering must not mutate its input, and must be re-runnable on the same input.

    This is the observable proxy for "specialize and consume one proof instance at a
    time" (see FormMulticoreProgram): an implementation that materializes a gx*gy clone
    of the structured input and specializes it in place would either mutate the caller's
    IRModule or produce a different module on the second run.
    """
    func = make_pipeline(stages=stages, extent=3, start=4)
    original = ir.save_json(func)
    first = lower_pipeline(func, arch)
    assert ir.save_json(func) == original, "lowering must not mutate its input IRModule"
    second = lower_pipeline(func, arch)
    assert ir.save_json(func) == original, "input must stay immutable across repeated lowering"
    assert ir.save_json(first) == ir.save_json(second)
    if stages:
        assert int(first.attrs["tt.pipeline_stages"]) == stages
        assert int(first.attrs["tt.pipeline_extent"]) == 3
    else:
        assert "tt.pipeline_stages" not in first.attrs


def test_pipeline_explicit_capacity_preserved_and_insufficient_diagnosed():
    mod = lower_pipeline(make_pipeline(3, 5, block_count=4))
    assert {int(desc.block_count) for desc in mod.attrs["tt.dfb_table"]} == {4}
    with pytest.raises(ValueError, match="capacity insufficient.*required=3"):
        lower_pipeline(make_pipeline(3, 5, block_count=2))


@pytest.mark.parametrize(
    "stages,extent,diagnostic", [(33, 2, "num_stages <= 32"), (2, 1025, "extent <= 1024"), (2, 0, "positive.*bounds|extent.*positive")]
)
def test_pipeline_static_limits(stages, extent, diagnostic):
    with pytest.raises((ValueError, NotImplementedError), match=diagnostic):
        lower_pipeline(make_pipeline(stages, extent))


def test_pipeline_rejects_tensor_inout_prefetch():
    with pytest.raises(NotImplementedError, match="inout Tensor"):
        lower_pipeline(make_pipeline(inout=True))


@T.prim_func
def loop_carried(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = pipeline_shared()
        for _k in T.Pipelined(3, num_stages=2):
            for i, j in T.Tiles(a):
                a[i, j] = a[i, j] + T.float32(1)
            T.copy(a, C)


def test_pipeline_rejects_loop_carried_dfb():
    with pytest.raises(ValueError, match="read-before-write|before definite initialization"):
        lower_pipeline(loop_carried)


@T.prim_func
def conditional_sites(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = pipeline_shared()
        c = pipeline_shared()
        for k in T.Pipelined(3, num_stages=2):
            T.copy(A, a)
            if k == 0:
                T.fill(c, 1)
            else:
                T.fill(c, 2)
            T.copy(c, C)


def test_pipeline_rejects_conditional_transaction_sites():
    with pytest.raises(NotImplementedError, match="conditional write sites"):
        lower_pipeline(conditional_sites)


@T.prim_func
def manual_schedule(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = pipeline_shared()
        for _k in T.Pipelined(3, num_stages=2, order=[0, 1], stage=[0, 1]):
            T.copy(A, a)
            T.copy(a, C)


def test_pipeline_rejects_manual_scheduling_annotations():
    with pytest.raises(NotImplementedError, match="manual stage/order/group|pipeline/scheduling"):
        lower_pipeline(manual_schedule)


def test_pipeline_zero_stages_preserves_serial_phase4_lower():
    mod = lower_pipeline(make_pipeline(stages=0, extent=3))
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    assert "tt.pipeline_stages" not in mod.attrs


@T.prim_func
def nested_pipeline(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = pipeline_shared()
        for _k in T.Pipelined(3, num_stages=2):
            for _j in T.Pipelined(2, num_stages=2):
                T.copy(A, a)
                T.copy(a, C)


def test_pipeline_rejects_nested_pipeline():
    with pytest.raises(NotImplementedError, match="nested T.Pipelined"):
        lower_pipeline(nested_pipeline)


@pytest.mark.parametrize("policy", ["conservative", "delayed"])
def test_pipeline_copy_completion_policy_is_preserved_for_codegen(policy):
    mod = lower_pipeline(make_pipeline(stages=3, extent=5, wait_policy=policy))
    assert str(mod.attrs["tt.pipeline_wait_policy"]) == policy
    calls = lexical_slot_calls(mod, "ncrisc")
    first_wait = next(i for i, call in enumerate(calls) if call.op.name == "tl.tt.dfb_copy_wait")
    issues = sum(call.op.name == "tl.tt.tensor_to_dfb_nd" for call in calls[:first_wait])
    assert issues == 1
    assert [(int(loop.min), int(loop.extent)) for loop in slot_loops(mod, "ncrisc")] == [(0, 5)]


def test_pipeline_invalid_copy_completion_policy():
    with pytest.raises(ValueError, match="must be conservative or delayed"):
        lower_pipeline(make_pipeline(wait_policy="unknown"))


@T.prim_func
def copy_pipeline(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = pipeline_shared()
        for _k in T.Pipelined(3, num_stages=2):
            T.copy(A, a)
            T.copy(a, C)


def test_pipeline_copy_only_materializes_snapshot_and_closed_lifetime():
    mod = lower_pipeline(copy_pipeline)
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    assert len(mod.attrs["tt.dfb_table"]) == 2
    assert {int(desc.transaction_count_or_loop_relation) for desc in mod.attrs["tt.dfb_table"]} == {3}
    computes = [call for call in lexical_slot_calls(mod, "trisc") if call.op.name == "tl.tt.dfb_compute"]
    assert len(computes) == 1
    assert all(call.annotations["tt.compute_kind"].value == "copy" for call in computes)
    VerifyTenstorrentDeviceIR()(mod)


def test_pipeline_wait_policy_requires_string():
    with pytest.raises(ValueError, match="pipeline_wait_policy must be a string"):
        lower_pipeline(make_pipeline(wait_policy=3))


def test_pipeline_maximum_static_extent_and_stage_boundary():
    mod = lower_pipeline(make_pipeline(stages=32, extent=1024))
    assert len(mod.attrs["tt.dfb_table"]) == 4
    assert {int(desc.block_count) for desc in mod.attrs["tt.dfb_table"]} == {32}
    assert {int(desc.transaction_count_or_loop_relation) for desc in mod.attrs["tt.dfb_table"]} == {1024}
    assert len(mod.attrs["tt.dfb_storage_groups"]) == 4
    assert [int(group.storage_group_id) for group in mod.attrs["tt.dfb_storage_groups"]] == list(range(4))
