"""Run the unchanged flash_attn_03 frontend through Device Lower and NumPy."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir, transform
from tvm import ir

from testing.python.target.test_tilelang_tenstorrent_compute_values import _round
from testing.python.target.test_tilelang_tenstorrent_lower_composition import interpret


_SOURCE = Path(__file__).parent / "fixtures" / "tenstorrent_flash_attn_03.py"
_SPEC = importlib.util.spec_from_file_location("tenstorrent_flash_attention_fixture", _SOURCE)
kernel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kernel)


def lower(config, arch="wormhole_b0"):
    source = tvm.IRModule({"main": kernel.make_flash_attention(**config)})
    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    mod = lower_tenstorrent_ir(source, target)
    assert "tt.ir_stage" not in mod.attrs
    expected_version = 7 if config.get("input_dtype") == "float32" else 8
    version = int(mod.attrs["tt.device_ir_version"])
    assert (int(mod.attrs["tt.compact_original_version"]) if version == 9 else version) == expected_version
    assert len(mod.functions) == 3 * config.get("core_q", 8) * config.get("core_bh", 4)
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(mod, restored)
    assert transform.VerifyTenstorrentDeviceIR()(restored).same_as(restored)
    return mod


def online_reference(q, k, v, dtype, mask=None):
    """Source algorithm with explicit storage/expression rounding at each step."""

    def cast(x):
        return _round(x, dtype)

    scale = np.float32(1 / np.sqrt(q.shape[-1]))
    log2e = np.float32(np.log2(np.e))
    previous_max = np.full(q.shape[:-1] + (1,), cast(-1e30), dtype="float32")
    denominator = np.zeros_like(previous_max)
    output = np.zeros_like(q)
    for start in range(0, k.shape[-2], 32):
        scores = cast(q @ k[..., start : start + 32, :].swapaxes(-1, -2))
        scores = cast(scores * scale)
        if mask is not None:
            scores = cast(scores + mask[:, start : start + 32])
        maximum = cast(np.maximum(previous_max, scores.max(axis=-1, keepdims=True)))
        rescale = cast(np.exp2(cast(previous_max - maximum) * log2e))
        probabilities = cast(np.exp2(cast(scores - maximum) * log2e))
        tile_sum = cast(probabilities.sum(axis=-1, keepdims=True, dtype=np.float32))
        denominator = cast(cast(denominator * rescale) + tile_sum)
        pv = cast(probabilities @ v[..., start : start + 32, :])
        output = cast(cast(output * rescale) + pv)
        previous_max = maximum
    return cast(output / denominator)


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("cores", [1, 2])
def test_flash_attention_device_values(arch, dtype, causal, cores):
    config = dict(
        batch=1, heads=2, seq_q=64, seq_kv=96, head_dim=64, core_q=cores, core_bh=1, input_dtype=dtype, accum_dtype=dtype, causal=causal
    )
    mod = lower(config, arch)
    rng = np.random.default_rng(42)
    q = _round(rng.normal(0, 0.4, (1, 2, 64, 64)), dtype)
    k = _round(rng.normal(0, 0.4, (1, 2, 96, 64)), dtype)
    v = _round(rng.normal(0, 0.4, (1, 2, 96, 64)), dtype)
    mask = _round(kernel.make_causal_mask(64, 96), dtype) if causal else None
    inputs = [q, k, v] + ([mask] if causal else []) + [np.zeros_like(q)]
    expected = online_reference(q, k, v, dtype, mask)
    if dtype == "float32":
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
        if mask is not None:
            scores += mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        reference = (weights / weights.sum(axis=-1, keepdims=True)) @ v
        np.testing.assert_allclose(expected, reference, rtol=2e-5, atol=2e-6)
    for seed in (3, 19):
        actual = interpret(mod, inputs, seed=seed)[0][-1]
        if dtype == "bfloat16":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    if arch == "wormhole_b0" and cores == 1 and not causal and dtype == "float32":
        assert ir.save_json(mod) == ir.save_json(lower(config, arch))


def test_default_flash_attention_complete_device_module():
    mod = lower({})
    assert len(mod.functions) == 96
    assert len(mod.attrs["tt.accumulator_table"]) == 32768


@pytest.mark.parametrize("inside_k", [False, True])
def test_reduction_cannot_snapshot_an_incomplete_k_accumulator(inside_k):
    @T.prim_func
    def invalid(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 1), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1})
            acc = T.alloc_fragment((32, 32), "float32")
            row = T.alloc_fragment((32, 1), "float32")
            T.copy(A, a)
            T.clear(acc)
            if inside_k:
                for _ in T.serial(2):
                    T.gemm(a, a, acc)
                    T.reduce_max(acc, row, dim=1)
            else:
                T.gemm(a, a, acc)
                T.reduce_max(acc, row, dim=1)
                T.gemm(a, a, acc)
            T.copy(row, C)

    with pytest.raises(NotImplementedError, match="inside its K reduction|after final materialization"):
        lower_tenstorrent_ir(tvm.IRModule({"main": invalid}), tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"}))
