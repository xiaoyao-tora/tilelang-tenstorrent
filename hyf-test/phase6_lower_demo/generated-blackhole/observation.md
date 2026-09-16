# Phase 6 Device IR observation

A 的两块沿行分发，B 的两块沿列分发；四个 worker 计算互不重叠的 C 分块。
源 BRISC 发出显式 send/wait，目的 NCRISC recv/wait 发布 DFB，TRISC 等待后 GEMM。
每个 DFB generation 独立分配，最后使用与传输完成后 release；没有物理 CB/L1 地址分配。

- net 0 record 0: [0, 1] → [1, 1], DFB 1 → 6
- net 0 record 0: [0, 1] → [2, 1], DFB 1 → 16
- net 0 record 1: [0, 2] → [1, 2], DFB 3 → 10
- net 0 record 1: [0, 2] → [2, 2], DFB 3 → 20
- net 1 record 0: [1, 0] → [1, 1], DFB 5 → 7
- net 1 record 0: [1, 0] → [1, 2], DFB 5 → 11
- net 1 record 1: [2, 0] → [2, 1], DFB 15 → 17
- net 1 record 1: [2, 0] → [2, 2], DFB 15 → 21

每 Core 静态 payload 字节：{'0,1': 8192, '0,2': 8192, '1,0': 8192, '1,1': 16384, '1,2': 16384, '2,0': 8192, '2,1': 16384, '2,2': 16384}；统一预算下界 16384。
三种逻辑调度次序下误差 [0.0, 0.0, 0.0]；JSON round-trip 与重复 Lower 确定性通过。
这只是最终 Device IR 的逻辑数值检查；未运行 TTL、TT-Lang、Simulator 或硬件。
