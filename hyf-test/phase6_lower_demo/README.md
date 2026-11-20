# Phase 6 多 Core / PipeNet Lower 示例

从仓库根目录运行：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-phase6-cache .venv/bin/python hyf-test/phase6_lower_demo/kernel.py
```

逻辑 grid 为 3×3：`(0,1)` / `(0,2)` 将 A 分块沿行 multicast，`(1,0)` / `(2,0)` 将 B 分块沿列 multicast；`(1,1)` 至 `(2,2)` 四个 worker 执行 GEMM 并写入不相交的 C 分块，`(0,0)` 空闲。

`generated/` 包含前端 IR、完整 Device IR script/JSON、27 个 Core/slot 函数、通信与资源推导 `observation.md` 和机器可读验证结果 `verification.json`。发送为 BRISC，接收及 Tensor I/O 为 NCRISC，计算和发送 snapshot 为 TRISC。四个 Pipe record 展开为八条 point delivery，每条都有显式 source/destination DFB、send/recv/completion，完成最后消费后 release。

脚本检查重复 Lower 的结构及 script/JSON 字节确定性、JSON round-trip 后 verifier、逐 Core payload 预算，并在三种逻辑指令交错下对比 NumPy GEMM。它不生成 TTL，不调用 TT-Lang，不运行 Simulator 或硬件；数值结果仅证明此逻辑 Device IR 子集的 Lower 语义。

可用 `--arch blackhole --output /tmp/phase6-blackhole` 验证另一目标的相同 Lower 契约。完整能力边界见 `../docs/tilelang_tenstorrent_phase6_lower_contract.md`。
