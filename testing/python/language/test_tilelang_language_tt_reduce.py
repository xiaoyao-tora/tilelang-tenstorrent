"""Tenstorrent reduction construction without target lowering."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T


@pytest.mark.parametrize("kind", ["sum", "max", "min"])
@pytest.mark.parametrize("scope", ["shared", "local.fragment"])
def test_reduction_preserves_storage_and_normalizes_axis(kind, scope):
    source = tvm.tirx.decl_buffer((32, 64), "float32", scope=scope)
    output = tvm.tirx.decl_buffer((32,), "float32", scope=scope)
    call = getattr(T, f"reduce_{kind}")(source, output, dim=-1, clear=False)

    assert call.op.name == "tl.tileop.reduce"
    assert call.args[2].value == kind
    assert int(call.args[3]) == 1
    assert not bool(call.args[4])
    for region, buffer in zip(call.args[:2], (source, output)):
        assert region.args[0].buffer.same_as(buffer)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_reduction_preserves_nan_policy(kind):
    source = tvm.tirx.decl_buffer((32, 32), "float32", scope="shared")
    output = tvm.tirx.decl_buffer((32,), "float32", scope="shared")
    call = getattr(T, f"reduce_{kind}")(source, output, nan_propagate=True)
    assert bool(call.annotations["nan_propagate"])


@pytest.mark.parametrize("dim", [-3, 2, 0.5])
def test_reduction_rejects_invalid_axis(dim):
    source = tvm.tirx.decl_buffer((32, 32), "float32", scope="shared")
    output = tvm.tirx.decl_buffer((32,), "float32", scope="shared")
    with pytest.raises(ValueError, match="reduction axis"):
        T.reduce_sum(source, output, dim=dim)


def test_reduction_rejects_allreduce_scheduling():
    source = tvm.tirx.decl_buffer((32, 32), "float32", scope="shared")
    output = tvm.tirx.decl_buffer((32,), "float32", scope="shared")
    with pytest.raises(NotImplementedError, match="AllReduce batch scheduling"):
        T.reduce_sum(source, output, batch=2)
