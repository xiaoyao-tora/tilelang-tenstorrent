"""Static row/column PipeNet distribution for a 2x2 block GEMM.

Run using .venv/bin/python. This emits and verifies Device IR only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from tilelang import tvm
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir


from testing.python.target.test_tilelang_tenstorrent_phase6_lower import make_row_column_gemm as make_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["wormhole_b0", "blackhole"], default="wormhole_b0")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated")
    args = parser.parse_args()
    source = tvm.IRModule({"row_column_gemm": make_kernel()})
    target = tvm.target.Target({"kind": "tenstorrent", "arch": args.arch})
    mod = TenstorrentPassPipelineBody(source, target)
    again = TenstorrentPassPipelineBody(source, target)
    ir.assert_structural_equal(mod, again)
    assert mod.script() == again.script() and ir.save_json(mod) == ir.save_json(again)
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(mod, restored)
    VerifyTenstorrentDeviceIR()(restored)
    from testing.python.target.test_tilelang_tenstorrent_phase6_semantics import run_multicore_device

    rng = np.random.default_rng(6)
    a = rng.integers(-3, 4, (64, 32)).astype(np.float32)
    b = rng.integers(-3, 4, (32, 64)).astype(np.float32)
    errors = []
    for seed in (0, 1, 19):
        actual, trace = run_multicore_device(restored, [a, b, np.zeros((64, 64), np.float32)], seed=seed)
        np.testing.assert_array_equal(actual[-1], a @ b)
        errors.append(float(np.max(np.abs(actual[-1] - a @ b))))
    transfers = [
        {
            "id": int(t.transfer_id),
            "net": int(t.pipe_net_id),
            "record": int(t.record_index),
            "src": [int(t.src_coord.x), int(t.src_coord.y)],
            "dst": [int(t.dst_coord.x), int(t.dst_coord.y)],
            "source_dfb": int(t.source_dfb_id),
            "destination_dfb": int(t.destination_dfb_id),
        }
        for t in mod.attrs["tt.pipe_transfer_table"]
    ]
    payload = {}
    for dfb in mod.attrs["tt.dfb_table"]:
        core = f"{dfb.producer_domain.begin.x},{dfb.producer_domain.begin.y}"
        size = (
            int(dfb.block_count)
            * int(np.prod([int(x) for x in dfb.tile_shape]))
            * int(np.prod([int(x) for x in dfb.block_shape_in_tiles]))
            * 4
        )
        payload[core] = payload.get(core, 0) + size
    budget = max(payload.values())
    VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", budget))
    report = {
        "arch": args.arch,
        "device_ir_version": 4,
        "grid": [3, 3],
        "function_count": len(mod.functions),
        "pipe_records": len(mod.attrs["tt.pipe_table"]),
        "dfb_generations": len(mod.attrs["tt.dfb_table"]),
        "transfers": transfers,
        "per_core_payload_bytes": payload,
        "required_per_core_budget": budget,
        "logical_numeric_seeds": [0, 1, 19],
        "maximum_absolute_errors": errors,
        "deterministic_script_and_json": True,
        "json_roundtrip_verified": True,
        "ttl_codegen": "not run",
        "ttlang_compile_only": "not run",
        "simulator": "not run",
        "hardware": "not run",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frontend.tir").write_text(source.script() + "\n")
    (args.output / "device_ir.tir").write_text(mod.script() + "\n")
    (args.output / "device_ir.json").write_text(ir.save_json(mod) + "\n")
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in mod.attrs["tt.kernel_order"]:
        (args.output / f"{name}.tir").write_text(mod[str(name)].script() + "\n")
    paths = "\n".join(
        f"- net {t['net']} record {t['record']}: {t['src']} → {t['dst']}, DFB {t['source_dfb']} → {t['destination_dfb']}" for t in transfers
    )
    (args.output / "observation.md").write_text(
        "# Phase 6 Device IR observation\n\n"
        "A 的两块沿行分发，B 的两块沿列分发；四个 worker 计算互不重叠的 C 分块。\n"
        "源 BRISC 发出显式 send/wait，目的 NCRISC recv/wait 发布 DFB，TRISC 等待后 GEMM。\n"
        "每个 DFB generation 独立分配，最后使用与传输完成后 release；没有物理 CB/L1 地址分配。\n\n"
        + paths
        + f"\n\n每 Core 静态 payload 字节：{payload}；统一预算下界 {budget}。\n"
        + f"三种逻辑调度次序下误差 {errors}；JSON round-trip 与重复 Lower 确定性通过。\n"
        + "这只是最终 Device IR 的逻辑数值检查；未运行 TTL、TT-Lang、Simulator 或硬件。\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
