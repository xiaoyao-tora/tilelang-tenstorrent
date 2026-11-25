"""BF16-stored, FP32-accumulated GEMM keeps operands and destination types apart.

TT-Lang requires ``ttl.matmul`` operands and result to share one tile dtype,
while FP32 accumulation is selected by the kernel attribute rather than by a
F32 result tile. These tests pin that split: the matmul stays BF16, an explicit
cast produces any declared F32 value, and ``fp32_dest_acc_en`` carries the
accumulation width.
"""

import importlib.util
import pathlib
import sys

import pytest

from tilelang import tvm
from tilelang.tenstorrent import codegen, lower_tenstorrent_ir
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.ttl_codegen.bindings import TTLANG_REVISION, load_bindings

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
METADATA = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}


def mixed_gemm_program():
    @T.prim_func
    def main(A: T.Tensor((64, 64), "bfloat16"), B: T.Tensor((64, 64), "bfloat16"), C: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((64, 64), "bfloat16", annotations=METADATA)
            b = T.alloc_shared((64, 64), "bfloat16", annotations=METADATA)
            accumulator = T.alloc_fragment((64, 64), "float32")
            output = T.alloc_shared((64, 64), "bfloat16", annotations=METADATA)
            T.copy(A, a)
            T.copy(B, b)
            T.clear(accumulator)
            T.gemm(a, b, accumulator)
            T.copy(accumulator, output)
            T.copy(output, C)

    return main


def _ttl_text(program):
    if importlib.util.find_spec("ttl") is None:
        pytest.skip(f"requires pinned TT-Lang compiler {TTLANG_REVISION}")
    load_bindings()
    device = lower_tenstorrent_ir(tvm.IRModule({"main": program}), TARGET)
    return codegen.build_ttl_without_compile(device, TARGET).inspect_source()


def _matmul_result_types(text):
    """Return the tile dtype of every ``ttl.matmul`` result."""
    results = []
    for line in text.splitlines():
        if "ttl.matmul" not in line or "->" not in line:
            continue
        results.append(line.rsplit("->", 1)[1].strip())
    return results


def test_mixed_gemm_keeps_bf16_matmul_and_fp32_destination():
    text = _ttl_text(mixed_gemm_program())
    assert "fp32_dest_acc_en = true" in text
    assert _matmul_result_types(text) == ["tensor<2x2x!ttcore.tile<32x32, bf16>>"]


def test_mixed_attention_casts_matmul_result_to_fp32():
    root = pathlib.Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "examples" / "tenstorrent"))
    from flash_attn import make_flash_attention

    program = make_flash_attention(
        batch=1,
        heads=1,
        seq_q=64,
        seq_kv=32,
        head_dim=32,
        core_q=2,
        core_bh=1,
        input_dtype="bfloat16",
        accum_dtype="float32",
        output_dtype="bfloat16",
    )
    text = _ttl_text(program)
    assert "fp32_dest_acc_en = true" in text
    results = _matmul_result_types(text)
    assert results
    assert all("tile<32x32, bf16>" in result for result in results)
    assert "-> tensor<1x1x!ttcore.tile<32x32, f32>>" in text
    assert "ttl.typecast" in text
