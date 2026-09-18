"""Static stage aliases and independent accumulator lifetimes after Core expansion."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import transform
from tvm import tirx


def _normalize(core_grid=2):
    @T.prim_func
    def main(A: T.Tensor((64, 64), "bfloat16"), B: T.Tensor((64, 64), "bfloat16"), C: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(core_grid, core_grid, threads=1) as (x, y):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            c = T.alloc_fragment((32, 32), "float32")
            T.clear(c)
            for stage in T.serial(2):
                k_begin = stage * 32
                k_end = k_begin + 32
                T.copy(A[y * 32 : y * 32 + 32, k_begin:k_end], a)
                T.copy(B[k_begin:k_end, x * 32 : x * 32 + 32], b)
                T.gemm(a, b, c, clear_accum=False)
            row_begin = y * 32
            col_begin = x * 32
            T.copy(c, C[row_begin : row_begin + 32, col_begin : col_begin + 32])

    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    mod = tirx.transform.BindTarget(target)(tvm.IRModule({"main": main}))
    for compiler_pass in (
        transform.CanonicalizeTTElementwise(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.VerifyTTGemmAccumulators(),
        transform.NormalizeTenstorrentTopology(),
        transform.NormalizeTenstorrentRegions(),
    ):
        mod = compiler_pass(mod)
    return mod


@pytest.mark.parametrize("core_grid", [1, 2])
def test_stage_aliases_resolve_and_core_accumulators_remain_independent(core_grid):
    mod = _normalize(core_grid)
    binds = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda node: binds.append(node) if isinstance(node, tirx.Bind) else None)
    assert not binds
    mod = transform.LegalizeTenstorrentTileOps()(mod)
    requirements = mod["main"].attrs["tt.gemm_accumulator_requirements"]
    assert len(requirements) == core_grid**2
    if core_grid > 1:
        assert {(int(item["core_x"]), int(item["core_y"])) for item in requirements} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    kinds = []
    tirx.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: (
            kinds.append(node.annotations["tt.compute_kind"].value)
            if isinstance(node, tirx.Call) and node.op.name == "tl.tt.tile_compute"
            else None
        ),
    )
    assert kinds.count("accumulator_init") == core_grid**2
    assert kinds.count("gemm_update") == 2 * core_grid**2
    assert kinds.count("accumulator_materialize") == core_grid**2


@pytest.mark.parametrize("mutation,message", [("missing", "dominating T.clear"), ("duplicate", "unique initialization")])
def test_other_cores_cannot_satisfy_or_hide_accumulator_initialization(mutation, message):
    mod = _normalize()
    clears = 0

    def rewrite(node):
        nonlocal clears
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.name == "tl.tileop.fill":
            clears += 1
            if clears == 2:
                return tirx.Evaluate(0) if mutation == "missing" else tirx.SeqStmt([node, node])
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, rewrite, ["tirx.Evaluate"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match=message):
        transform.VerifyTTGemmAccumulators()(mod)
