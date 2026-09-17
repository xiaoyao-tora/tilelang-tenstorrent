# Flash Attention 的 Device Lower

本次以 `hyf-test/kernel/flash_attn_03.py` 原始 kernel 为验收输入，保留其
4D Tensor ABI、BF16 存储类型、Python 浮点常量和在线 softmax 更新顺序。
回归副本位于 `testing/python/target/fixtures/tenstorrent_flash_attn_03.py`。
目标是工作计划 02 第 22.1 节的逻辑 Device TIR 输出；不扩展 Codegen。

## 前端与值生命周期

- Tensor copy 接受高阶 Tensor 的前导 extent-one 切片轴，局部 payload
  仍为 rank-2。原始 Tensor shape、坐标、边界和 ABI 保留。
- Reduction 支持 `[B, N] -> [B, 1]` keepdims fragment。行 fragment 使用
  逻辑 shape 和向上取整的物理 tile grid，不虚构完整的 shared 容量。
- 行 fragment 写入 padded shared 的第零列，仅当全函数证明其他列不可观察时
  转成行 broadcast；非零列读取、其他写入、alias 和不透明指针逃逸阻止转换。
- 每次独立 QK/PV GEMM 可以重新初始化已完成的 accumulator。Reduction 和
  独立 fragment 消费者仅能在完整 K 更新结束后形成精确 dtype 的快照。
  K 循环内的中途快照和后续继续累加仍被拒绝。
- DFB materialization 保留实际 accumulator 来源，包括在线 softmax 中的
  多个根；最后使用分析沿值依赖图传播，避免过早 release。
- Pipe 操作的存活性按 DFB operand 判断，transfer ID 使用独立编号空间。
  这避免大规模广播中误删 receive/completion 而留下未发布 DFB。

## Device IR v8 精度区域

全 FP32 的该 kernel 保持 Device IR v7。BF16 GEMM 与真实 FP32 标量中间
运算组合时使用 v8：例如源程序 `BF16(scores * scale)` 中的乘法保持 FP32，
不能通过把 `scale` 改成 BF16 或忽略中间类型来通过精度检查。

TRISC 中新增有序、带副作用的 `tl.tt.compute_precision(mode)`；`mode=0`
表示 16-bit DST，`mode=1` 表示 32-bit DST。它声明逻辑计算区域，后续 consumer
需要实现实际精度配置。它不承诺物理 DST 驻留或硬件切换能力。

区域切换前将当前值精确存入 DFB；切换后以新版本 identity 从 DFB 重新进入值图。
旧区域的 SSA 值不能被新区域直接引用。独立 verifier 检查精确 dtype、完整 K、
DFB reserve/publication/wait/release、版本及区域归属。

每个 TRISC 的 `tt.compute_region_requirements` 是按 marker 顺序排列的 typed
`ComputeRequirements` 数组。`tt.compute_requirements` 的汇总宽度为
`region_scoped`，各区域仍保留独立的 `bits16_required` / `bits32_required`
和 full-FP32 约束。Verifier 从实际操作重新推导要求，拒绝伪造 mode、遗漏
marker、跨区域值引用和不一致 metadata。v1–v7 的原有精度合同不变。

v8 当前只支持静态串行事务，不支持与 `T.Pipelined` 合成。实际 FP32 fragment
也不能直接在 16-bit 区域重新加载；这里没有增加隐式窄化。现有 TTL consumer
会拒绝 v8，不能将 `region_scoped` 当成一个函数级精度开关使用。

## 验证方式

`test_tilelang_tenstorrent_flash_attention.py` 使用完整原始前端函数，覆盖
BF16/FP32、Wormhole/Blackhole、causal 开关和单/多 Core。较小输入包含多次
KV 更新及多个 query/head wave。Device 流参考执行器执行两种随机 slot 调度，
对照显式舍入的 NumPy 在线 softmax；FP32 另与普通完整 softmax 对照。
BF16 Device 输出与在线参考逐元素完全一致；FP32 使用明确的浮点误差容限。
这属于逻辑 Device 数值检查，不等同于设备执行。

默认尺寸测试使用 batch=2、heads=8、Q/KV length=1024、head_dim=64、
Core grid=8×4、block=32，检查 96 个 slot PrimFunc 和 32768 个 GEMM
accumulator，并执行 JSON round-trip 和独立 verifier。

新增负向回归覆盖切片边界/不匹配、可观察 padding、alias、非法行几何、
不完整 K、伪造 DFB/value provenance、精度区域和 Pipe transfer 编号。
完整回归运行方式见 [Lower 组合能力](tenstorrent_lower_composition.md)。

2026-09-17：原生重建和 pre-commit 通过。常规 Lower 回归为 **1082 passed**
（33 条既有 Python typing 弃用警告）；将默认大尺寸用例从该次 pytest 中单独
选出，直接导入原始 kernel 做完整导出。默认 BF16 的公开 Lower 和独立 Device
verifier 通过，Lower 耗时约 **129 秒**；完整 JSON 往返后的结构比较及 verifier
也通过。静态展开后的默认 JSON 约 2.7 GiB，检查文本时可先使用相同 kernel 的
较小配置。

`device_ir.tir` 是带完整 metadata 的诊断文本；无损重载使用已验证的
`device_ir.json`。当前通用 TVM TIRX printer 会将 `tt.compute_kind` 等带点的
Call annotation 名直接打印成 keyword，因而该文本不能直接作为 Python 脚本
执行。这也发生在既有 v2 Fill 输出中；本次未修改 TVM printer，工作计划中
print/parse 文本重建这一目标仍不能据此标记完成。

本次通过该 kernel 的纵向路径，不意味着工作计划第 17 章所有阶段完成。
通用动态控制流、任意 alias/tail、通用流水调度、物理 CB/DST 分配，以及
Codegen、TTNN 和硬件验收仍按各自边界处理。
