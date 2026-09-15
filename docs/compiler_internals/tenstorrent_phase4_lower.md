# Phase 4 单 Core Lower 契约

本文件补充工作计划第十七章。这里的验收对象是完成 compute block 消费、资源规划和
Device verifier 的 Device IR。单独完成 `CanonicalizeTTElementwise` 与
`VerifyTTComputeBlocks` 只证明 structured compute 契约成立，不代表完整 Lower。
TTL Codegen、TTL parser/verifier、TT-Lang compile-only 和硬件数值验证均不属于本轮。

## 1. Pass 边界

沿用已有十二个 Pass，不增加重复的 compute 分析或数据流 Pass：

```text
BindTarget
  -> CanonicalizeTTElementwise
  -> VerifyTTComputeBlocks
  -> ValidateTenstorrentFrontendIR
  -> NormalizeTenstorrentLaunch
  -> NormalizeTenstorrentBufferMetadata
  -> NormalizeTenstorrentRegions
  -> NormalizeTenstorrentTopology
  -> LegalizeTenstorrentTileOps
  -> InferTenstorrentTensorLayout
  -> FormTenstorrentDeviceProgram
  -> VerifyTenstorrentDeviceIR
```

| Pass | 输入和职责 | 输出及禁止承担的职责 |
| --- | --- | --- |
| CanonicalizeTTElementwise | 原始 Tiles/Parallel loop、allocation metadata；证明访问关系、构建统一表达式和 effects | structured compute block；不规划 DFB、不决定 processor slot |
| VerifyTTComputeBlocks | compute block、表达式模板、访问描述和读写集合 | 验证后的同一 IR；不把合法但未消费的模板视作 Device IR |
| LegalizeTenstorrentTileOps | 规范化的 compute block、独立 TileOp 和 metadata | 显式 tile operation、无隐式 scalar compute；验证运算和 dtype 的支持范围，不创建 processor slot |
| InferTenstorrentTensorLayout | tile operation 及规范化 Buffer metadata | 一致的 Tensor/DFB tile-grid 和 layout；不重复解析前端 elementwise loop、不决定运算 def-use |
| FormTenstorrentDeviceProgram | 所有 compute 已消费、layout 完整的单 Core 程序 | 版本化 DFB、显式 transaction、TRISC/NCRISC/BRISC 和 typed module metadata |
| VerifyTenstorrentDeviceIR | Device IR module 及所有 slot | 检查 schema、操作数、dtype、shape、effect、资源归属、ABI 和资源协议；不修补错误 IR |

完整 backend pipeline 要么返回通过 `VerifyTenstorrentDeviceIR` 的 module，要么报告
具体不支持条件；不再用 `tt.ir_stage="structured"` 表示完整 Lower 成功。需要查看
capture 结果的工具和测试可直接调用前两个 compute Pass。

## 2. Operation 与资源设计

### 2.1 计算中间形式与 Device 形式

通用路径使用已注册的 `tl.tt.tile_compute` 作为 Legalize 与 Form 之间的接口，使用
`tl.tt.dfb_compute` 表示完成资源规划后的设备计算。两者必须具有明确的 operation
kind、输入顺序、结果 dtype、shape 和各 operation 的必要属性。`tile_compute`
只能存在于中间 IR，不能出现在最终 Device IR。

elementwise 的表达式是保留节点 dtype 的纯运算 DAG。Device 阶段的输入叶节点为
`tl.tt.dfb_load`，它表示读入逻辑 DFB block 的 tile 值，不是从宿主内存取 scalar。
最终表达式不得残留 frontend Buffer、BufferLoad、自由 Var 或未消费的 SBlock。
access map 与有序 input 对应。本轮每个 Buffer 只能有一种 access map；例如
`A[i, j] + A[0, j]` 明确拒绝，不在去重时丢失差异。操作 kind 字符串不能替代表达式、
轴或 accumulation 语义。当前 expression 集合不包含 comparison/select。

经典 Phase 2 Add 与 no-op 保留 Device IR v1 兼容路径。通用 operation 使用 v2；
消费者不得把未知 v2 operation 当作 v1 Add 解释。TTL Codegen 将来必须先检查
schema version，再按对应契约消费 operation。

| Operation | 参数与属性 |
| --- | --- |
| `tl.tt.tile_compute` | 参数为 `[output BufferRegion, input BufferRegions...]`；仅 Legalize→Form 中间层存在 |
| `tl.tt.dfb_compute` | 参数为 `[output dfb_id, input dfb_ids...]`；output 必须是新的 generation，只有 TRISC 可执行 |
| `tl.tt.dfb_load` | 单一 `dfb_id` 参数，result dtype 等于对应 DFB dtype；仅存在于纯 `tt.expression` 中 |
| `tl.tt.tensor_to_dfb_nd` | `[tensor_id, dfb_id, min0, extent0, ...]`；NCRISC 发布已 reserve 的 DFB |
| `tl.tt.dfb_to_tensor_nd` | `[dfb_id, tensor_id, min0, extent0, ...]`；NCRISC 消费已 wait 的 DFB |
| `tl.tt.dfb_reserve` / `tl.tt.dfb_wait` | `[dfb_id, transaction_count]`，每次调用请求一个 block transaction，count 固定为 1 |

每个 compute 的公共 annotations 是 `tt.compute_kind`、`tt.compute_dtype`、
`tt.compute_tile_shape`、`tt.logical_domain`、`tt.input_shapes` 和 `tt.access_maps`。
`tt.input_shapes` 按 input 参数保存逻辑 shape，不能用 tile-grid 代替：Reduction
移除轴后，原本 untiled 的 batch 轴可能进入最后两轴，需要重新推导 output tile-grid。
`tt.access_maps` 是按
input 参数顺序排列的整数 axis 数组，elementwise 中 `-1` 表示该输入轴读取固定零，
非负值表示对应的 output axis。它不应被误用为 GEMM/Reduction 的 axis 语义。

| kind | 额外语义 |
| --- | --- |
| `elementwise` / `typecast` / `fill` | `tt.expression` 保留运算 DAG；Fill 没有 DFB input；typecast 也不能丢失其内层复合表达式 |
| `copy` | Form 为输出 snapshot 生成的同 dtype、同 shape 值复制 |
| `transpose` | `tt.axes` 为保留 batch、交换末两轴的 permutation |
| `gemm` | `tt.transpose_a`、`tt.transpose_b`、`tt.clear`、`tt.accum_dtype`；M/N/K 从经验证的 operand shape 推导；clear 为 false 时末尾 input 是旧 accumulator |
| `reduce` | `tt.reduce_axis`、`tt.reduce_kind`、`tt.clear`、`tt.nan_propagate`、`tt.accum_dtype`；clear 为 false 时末尾 input 是旧输出值 |

`tt.clear=true` 的 sum/GEMM 使用零初始化；max/min 使用对应归约单位元。
`tt.clear=false` 从显式旧值开始累积，read-before-write 必须失败。

### 2.2 Shape 与 dtype

物理 tile 为最后两维的 32×32；batch 维是 untiled 前缀。逻辑形状
`[B0, ..., M, N]` 的 tile-grid 为 `[B0, ..., M / 32, N / 32]`。
末两维不整除时必须有明确 padding/mask 契约，本轮不隐式补零。DFB block shape
表示该逻辑 payload 中的 tile-grid，`block_count` 表示容量，不能把容量当 tile 数。

rank-1 Reduction 结果是明确例外：逻辑 `[N]`、tile-grid `[N / 32]`，物理上占据
`32 × N` tile 值的第零行，剩余行不属于逻辑结果。后续 TTL transfer 必须按逻辑
region 读写，不能把 padded 行写入 Tensor；这个契约尚未通过 TTL 或硬件验证。
rank-1 的 N 仍须为正的 32 倍数，不推广到任意 padding。

Buffer storage dtype 与表达式节点 dtype 分别检查。显式 Typecast 的目标 dtype、
GEMM/Reduction 的 accumulation dtype 和初始化规则必须保留；不得把 BF16 运算
的舍入点隐式改成 FP32，也不得将显式 FP32 accumulator 截断为 BF16。

### 2.3 顺序与 alias

通用数据流按程序的读写顺序跟踪每个 Buffer 的当前值。每次写入建立新的 logical
DFB generation，后继读绑定该 generation；同 Buffer 原地更新先读取旧值，再发布
新值。TRISC 产生并被后续 TRISC 消费的中间 DFB 不要求伪造 Tensor backing。
Form 从 Tensor 输出逆追资源依赖，移除不影响可见输出的纯计算及其 reserve/wait；
未使用的 Tensor 参数仍保留 ABI。这解决值依赖，不做物理 DFB 复用、L1 分配或多
stage 流水优化。

不同 Buffer 的 view/alias 必须有明确 identity 和 region 关系。无法证明 alias 时应
诊断；不能根据名字相同推断共享存储，也不能把 shape/dtype 不兼容的 view 当作
普通 in-place。Tensor effect 从实际 transfer 读写推导，不能沿用 Add 固定的
`input, input, output` 参数顺序。

导出 Tensor 的值通过单独的 TRISC copy snapshot DFB 交给 NCRISC，避免同一个
generation 同时拥有 TRISC 和 NCRISC 两种 consumer。对 immutable generation 的
多次 wait/read 不等于多次生产；直线程序 descriptor 的 transaction relation 为单次发布。
后续 TTL 消费者必须保证值活到最后一次使用，不能把每次 wait 都解释为释放资源。

本轮静态 serial/unrolled loop 通过代入常量展开，extent 上限为 1024，整个规划
过程上限为 65536 个 statement；常量条件在展开后选择分支。动态 loop/条件及需要
跨 slot phi/guard 平衡的动态控制流明确拒绝；下一节限定独立迭代 For 的保留范围。

### 2.4 独立迭代的结构化 For（Device Lower 已验证）

允许保留的窄范围是：整个 kernel body 由单个正静态 serial loop 包围，loop variable
不被 body 引用，body 本身包含完整的 Tensor→DFB→compute→Tensor 数据流，每次迭代
均可独立规划，不读取任何前一迭代的 DFB generation。所有 active slot 使用相同
loop 边界；idle BRISC 保持 idle。其他静态循环使用有界展开，不据此扩展动态控制流。

此时 `dfb_id` 是 lexical resource ID；实际值身份为 `(dfb_id, iteration epoch)`。
descriptor 的 `transaction_count_or_loop_relation` 等于 loop extent，而每次
reserve/wait 仍请求一个 transaction。单迭代中的 immutable publication、最后使用
和 release 都发生在迭代作用域内，不能将 attached block 值提升到循环外。

Device verifier 必须检查 active slot loop 边界一致、所有资源 transaction relation
一致，并对 loop body 的完整 def-use 和跨 slot wait-for DAG 执行同样验证。
在这些前提下，每轮从 Tensor 重新构造全部 DFB 值，以迭代归纳证明资源依赖闭合。
即使 Tensor 参数是 inout，NCRISC 的前一轮输出 transfer 也必须排在下一轮输入
transfer 前面。

TTL Codegen 后续可将该 For 映射为 `scf.for`，所有 reserve/wait/attach/compute/store
及 consumer 最后使用后的释放均放在对应 loop body 内。该 lifecycle 契约尚未经
TTL verifier 或硬件验证；本轮已通过保留 For、Tensor inout 数值语义、跨 slot 边界
一致性、transaction relation 和只读 verifier 的正向/负向回归。

## 3. 后续 TTL 映射责任

以下是预期映射契约，尚未实现或经 TTL verifier 验证：

| Device 语义 | 后续 initial TTL 的责任 |
| --- | --- |
| DFB reserve/wait、输入 load | 绑定 logical DFB，并建立对应的 reserve/wait/attach 和 block 值 |
| elementwise 表达式 DAG | 按节点 dtype 构造本轮支持的 tensor-level 算术/cast/unary SSA，不重新捕获 scalar loop |
| broadcast access map | 显式 materialize 对应维度的 block broadcast；保留 batch 维与固定 lane 的区别 |
| Fill | 构造明确 dtype 的常量和 fill |
| Typecast | 使用显式源/目的 dtype 构造 typecast，保留舍入位置 |
| Transpose | 根据 axes 置换 block 维度及 tile 内语义，不能只改变 shape |
| GEMM | 传递 A/B transpose、M/N/K、clear/accumulate 和 accumulator dtype；本轮 GEMM 无 batch |
| Reduction | 传递 reduction kind、axis、输出 shape、初始化和 accumulation dtype |
| 结果发布 | 将计算值 store 到 reserved output，保持 generation 和 transaction 依赖 |
| Tensor transfer | 根据 rank-aware region 构造 Tensor slice、copy 和 completion；不假定 rank 为 2 |

Lower 不选择 FPU/SFPU、DST 寄存器、TTKernel 指令或 EmitC/C++。这些也不能作为本轮
Device IR 验证的替代物。

## 4. 验收与兼容性检查

支持矩阵以工作计划第十七章及实际测试为准。每项“Device Lower 已完成”必须同时有：

1. 带输入/输出数据流的完整程序成功形成三个 Device slot；
2. compute block 全部消费，表达式、轴、dtype 和依赖可以从 Device IR 还原；
3. Device verifier 检查 operation 与 Tensor/DFB 元数据一致性；
4. 至少覆盖正向和对应诊断边界，且没有把 capture-only 测试计入完整 Lower。

重点回归包括 Tiles/Parallel 等价程序、经典 BF16/FP32 Add、多 operation 中间值、
重复输入、读前未定义、原地更新、未知 dtype、错误 shape、错误 DFB 依赖与 ABI。
结构化循环和条件必须证明各 slot 的执行次数、guard 和 transaction 一致；仅保留
`For`/`IfThenElse` 节点不等价于已完成控制流 lowering。

本轮不扩展 Phase 5 的容量复用、异步 transaction 调度或 `T.Pipelined`；不扩展
Phase 6 的多 Core、PipeNet、跨 Core barrier 或通信。单 Core 资源依赖验证不应被
表述为已经完成这些阶段。

## 5. 本轮验证记录（2026-09-15）

- 原生 C++ 构建：完成全量构建基础上的修改对象重编译与 `libtilelang.dylib` 链接；
  最终六个修改的 C++ translation unit 均使用 CMake/Ninja 生成的原始命令编译成功。
- 最终回归 **373 passed，29 warnings**。范围是全部
  `testing/python/target/test_tilelang_tenstorrent*.py`、backend import 测试和语言 Tiles 测试。
  warning 为 Python 3.15 相关的现有 `typing._eval_type` 弃用提示。
- 新增四组测试分别验证 compute 消费、Device 数据流/元数据、最终 Device 数值语义及
  结构化循环。数值语义组含 60 项，覆盖 Tiles/Parallel 等价、BF16 舍入、复合表达式、
  broadcast、Fill/Typecast/Transpose、GEMM transpose/clear、Reduction axis/clear/dtype、
  batch Transpose/Reduction、同 Tensor inout 和 dead Tensor transfer/processor ABI。
- verifier 负向覆盖缺失 wait、重复发布、跨 slot 环依赖、未定义读、shape/dtype/axis/
  accumulation 不一致、原始 `Array[int]` 注入、非法表达式节点和不一致 loop epoch。
- 新增 Python 测试及 Lower 接口通过 Ruff；`git diff --check` 通过。

复现命令（先 import tilelang 以初始化本仓库的 TVM 依赖路径）：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-phase4-cache .venv/bin/python - <<'PY'
import tilelang
import pytest
from pathlib import Path
paths = sorted(str(p) for p in Path('testing/python/target').glob('test_tilelang_tenstorrent*.py'))
paths += ['testing/python/backend/test_tilelang_tenstorrent_import.py',
          'testing/python/language/test_tilelang_language_tiles.py']
raise SystemExit(pytest.main(paths + ['-q', '--tb=short']))
PY
```

这些结果证明本轮限定的 Lower/Device 契约和参考数值语义，不证明 TTL 可编译性、
设备指令可用性、实际 BF16/FP32 硬件行为、L1 容量可落地或性能。

## 6. 能力矩阵

以下“实现范围”描述本轮代码的设备消费路径；验证结果需与下方验收记录一起阅读。
仅通过 capture 的表达式不计入这张表的 Device Lower 能力。

| 能力 | Device Lower 实现范围 | 明确边界 |
| --- | --- | --- |
| Phase 2 Add 回归 | 原有 32×32 BF16/FP32 Add 及 no-op 保留 v1 | 通用路径 v2 与 legacy v1 分别验证 |
| 多 tile elementwise | Tiles/Parallel 共享 consumer，正静态 rank ≥ 2，末两轴 32 整除，任意合法 batch 前缀 | 不做 mask/padding、dynamic shape、偏移/散射访问 |
| 复合表达式 | Add/Sub/Mul/Div/Min/Max/Cast；exp/exp2/log/log2/sqrt/rsqrt/tanh/sin/cos/fabs/floor/ceil；DAG 保留节点 dtype | 不支持任意 extern、带副作用 call、RHS 使用坐标；comparison/select 不在本轮表达式集合 |
| broadcast | 每个输入 axis 为对应输出 axis 或固定零；row/column/scalar 及合法 batch broadcast | broadcast 输入仍需合法物理 tile allocation；同一 Buffer 在一个表达式中使用冲突 access maps 拒绝 |
| Fill | shared full-region 常量填充；无输入的表达式计算 | 不支持动态 fill 值或调度扩展 |
| Typecast | 显式 expression cast 与 shared→shared `T.copy`，保留源/目标 dtype | 完整 shape 相同；仅 BF16/FP32，非 global transfer 的隐式转换 |
| Transpose | 独立 `T.transpose`，交换末两轴，batch 前缀保持，输入输出 dtype 相同 | rank ≥ 2、完整 region；不支持任意 axes permutation 或输入输出 alias |
| GEMM | rank-2 完整矩阵，正的 32 整除 M/N/K，transpose A/B，clear/accumulate | A/B 同 BF16 或 FP32，C/accumulation FP32；batch GEMM、非默认 GPU 调度/precision 扩展拒绝 |
| Reduction | sum/max/min，单 axis、移除该 axis，clear/accumulate；sum 用 FP32 accumulator | 输出 dtype 与输入相同或 FP32；无 keepdims、source/output alias 或任意 reducer；结果仍须满足 layout 条件 |
| batch dimension | elementwise、Fill/Typecast、末两轴 Transpose 及满足结果布局的 Reduction | 不等于已支持 batched GEMM；batch 轴在成为最后两轴时也须满足 32 整除约束 |
| dtype/accumulation | BF16/FP32 storage，DAG 保留显式 cast 和舍入点，GEMM FP32 accumulator | integer/FP16/vector storage、任意混合精度策略不支持 |
| 结构化控制流 | 正静态 serial/unrolled loop 展开，常量条件及展开后可化简条件选择；完整独立 whole-body serial loop 保留为 Device For，已验证 epoch 和 Tensor inout | 展开单 loop extent ≤ 1024、累计 ≤ 65536 个 statement；保留 loop 须同 slot 边界且无跨迭代 DFB；动态 loop/条件、`T.Pipelined` 不支持 |
| alias/in-place | 同 Buffer 的逐元素更新读取旧 generation，写入新 generation；完整同 shape/dtype 值依赖 | 部分 region、不同 Buffer 的 view/alias、strided storage、transpose/GEMM/reduce 输入输出重叠拒绝 |
| 通用数据流 | 任意 Tensor 参数顺序、多个 operation、中间 TRISC→TRISC DFB、明确输出 snapshot | 单 Core 1×1；每个 generation 固定 producer/consumer slot；从 Tensor 输出逆追消除 dead 纯计算，read-before-write 明确诊断 |

