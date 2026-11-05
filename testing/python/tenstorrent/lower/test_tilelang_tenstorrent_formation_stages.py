"""Standalone formation stages, temporary-plan lifecycle and instrumentation."""

import json

import pytest
from tilelang import tvm
from tvm import ir, tirx
from tilelang.tenstorrent import transform
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_formation_support import (
    before_formation, replace_add,
)

PLAN = "tt.internal.grid_plan"
STAGES = ["ValidateTenstorrentDeviceProgramInput", "FormTenstorrentGridProgram",
          "PlanTenstorrentCorePrograms"]


def with_core_grid(mod, gx, gy, dtype):
    f = mod["main"]
    x, y = f.body, f.body.body
    core_x = tirx.Var("core_x", dtype)
    core_y = tirx.Var("core_y", dtype)
    body = tirx.stmt_functor.substitute(y.body, {x.loop_var: core_x, y.loop_var: core_y})
    grid = [tirx.IntImm(dtype, gx), tirx.IntImm(dtype, gy)]
    body = tirx.For(core_y, tirx.IntImm(dtype, 0), grid[1], y.kind, body,
                    annotations=y.annotations)
    body = tirx.For(core_x, tirx.IntImm(dtype, 0), grid[0], x.kind, body,
                    annotations=x.annotations)
    return tvm.IRModule({"main": f.with_body(body).with_attr("tt.launch_grid", grid)},
                       attrs=mod.attrs)


@pytest.mark.parametrize("gx,gy", [(1, 1), (2, 3), (3, 1)])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_standalone_stages_and_rebuild(gx, gy, dtype):
    mod = with_core_grid(before_formation(), gx, gy, dtype)
    checked = transform.ValidateTenstorrentDeviceProgramInput()(mod)
    assert checked.same_as(mod)
    planned = transform.FormTenstorrentGridProgram()(mod)
    plan = planned.attrs[PLAN]
    assert plan.frontend.same_as(mod["main"])
    assert plan.body.same_as(mod["main"].body.body.body)
    assert [(t.task_id, t.x, t.y) for t in plan.tasks] == [
        (x * plan.grid_y + y, x, y)
        for x in range(plan.grid_x) for y in range(plan.grid_y)
    ]
    assert list(planned.functions) == list(mod.functions)
    assert set(planned.attrs.keys()) == set(mod.attrs.keys()) | {PLAN}
    rebuilt = transform.FormTenstorrentGridProgram()(planned)
    assert not rebuilt.attrs[PLAN].same_as(plan)
    assert ir.structural_equal(planned, rebuilt)
    result = transform.PlanTenstorrentCorePrograms()(planned)
    assert PLAN not in result.attrs
    assert all(PLAN not in f.attrs for f in result.functions.values())
    with tvm.transform.PassContext(instruments=[capture_formation_events()]):
        expected = transform.FormTenstorrentDeviceProgram()(mod)
    assert ir.structural_equal(result, expected)
    if (gx, gy, dtype) == (1, 1, "int32"):
        for serialization in ("json", "script"):
            check_plan_roundtrip(planned, result, serialization)


def check_plan_roundtrip(planned, expected, serialization):
    if serialization == "json":
        restored = ir.load_json(ir.save_json(planned))
    else:
        source = planned.script(show_meta=True)
        assert 'metadata["tl.tenstorrent.GridPlan"]' in source
        # Replay the serialized plan, not the generic parser's synthetic root.
        namespace = {"tvm": tvm}
        exec(source.split("# from tvm.script", 1)[0], namespace)
        plan = namespace["metadata"]["tl.tenstorrent.GridPlan"][0]
        restored = tvm.IRModule({str(plan.operation): plan.frontend}, attrs=plan.module_attrs)
        restored = restored.with_attr(PLAN, plan)
    assert ir.structural_equal(planned, restored, map_free_vars=True)
    result = transform.PlanTenstorrentCorePrograms()(restored)
    assert ir.structural_equal(result, expected, map_free_vars=True)


@pytest.mark.parametrize("change", ["missing", "type", "body", "grid", "metadata", "target", "module", "owner"])
def test_invalid_plans_rejected(change):
    planned = transform.FormTenstorrentGridProgram()(before_formation())
    if change == "missing":
        planned = planned.without_attr(PLAN)
    elif change == "type":
        planned = planned.with_attr(PLAN, "not a plan")
    elif change == "module":
        planned = planned.with_attr("changed", 1)
    elif change == "owner":
        planned = tvm.IRModule({"other": planned["main"]}, attrs=planned.attrs)
    else:
        f = planned["main"]
        if change == "body":
            f = f.with_body(tirx.Evaluate(1))
        elif change == "grid":
            f = f.with_attr("tt.launch_grid", [2, 1])
        elif change == "target":
            f = f.with_attr("target", tvm.target.Target("llvm"))
        else:
            f = f.with_attr("tt.buffer_metadata_table", [])
        planned = tvm.IRModule({"main": f}, attrs=planned.attrs)
    with pytest.raises((ValueError, TypeError), match="GridPlan|grid_plan"):
        transform.PlanTenstorrentCorePrograms()(planned)


@pytest.mark.parametrize("node_type,field,value,message", [
    ("tl.tenstorrent.GridTask", "task_id", 7, "GridPlan tasks"),
    ("tl.tenstorrent.GridTask", "x", 7, "GridPlan tasks"),
    ("tl.tenstorrent.GridTask", "y", 7, "GridPlan tasks"),
    ("tirx.For", "kind", 1, "serial logical Core"),
    ("tirx.For", "min", None, "must start at zero"),
])
def test_corrupted_serialized_plan_is_rejected(node_type, field, value, message):
    planned = transform.FormTenstorrentGridProgram()(before_formation())
    serialized = json.loads(ir.save_json(planned))
    nodes = [n for n in serialized["nodes"] if n.get("type") == node_type]
    assert nodes
    for node in nodes:
        node["data"][field] = node["data"]["extent"] if value is None else value
    forged = ir.load_json(json.dumps(serialized))
    with pytest.raises(ValueError, match=message):
        transform.PlanTenstorrentCorePrograms()(forged)


def test_operation_diagnostic_precedes_topology():
    mod = replace_add(before_formation(), lambda a, b: tirx.call_intrin("float32", "tirx.erf", a) + b)
    mod["main"] = mod["main"].without_attr("tt.topology_normalized")
    for stage in [transform.ValidateTenstorrentDeviceProgramInput,
                  transform.FormTenstorrentGridProgram,
                  transform.FormTenstorrentDeviceProgram]:
        with pytest.raises(NotImplementedError, match="tirx.erf"):
            stage()(mod)


def capture_formation_events():
    events = []

    @tvm.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, mod, info):
            events.append(("before", info.name.rsplit(".", 1)[-1]))

        def run_after_pass(self, mod, info):
            events.append(("after", info.name.rsplit(".", 1)[-1]))
            if info.name.endswith("FormTenstorrentGridProgram"):
                assert PLAN in mod.attrs
            if info.name.endswith("FormTenstorrentDeviceProgram"):
                assert PLAN not in mod.attrs
                assert events == [("before", "FormTenstorrentDeviceProgram")] + [
                    event for stage in STAGES for event in [("before", stage), ("after", stage)]
                ] + [("after", "FormTenstorrentDeviceProgram")]

    return Capture()
