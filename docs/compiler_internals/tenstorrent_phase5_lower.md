# Phase 5 流水与容量 Lower 契约

本文件补充工作计划第十七章，在已完成 Phase 4 Lower 上扩展 `T.Pipelined`、
Device IR 和 verifier。Phase 3 继续跳过；TTL Codegen、TTL parser/verifier、
TT-Lang compile-only、物理分配差分、Simulator 和硬件验证均延期。这里的流水是
编译器可验证的 transaction 调度，不是设备执行、性能或 L1 物理可落地的认证。

本文保留最初 v3 的实现记录。后续新增的循环前后一次性资源、静态 Tensor 切片、
持久 accumulator 和多 Core / compute-value 组合，以及更新后的 metadata 契约，
见 [Lower 组合能力](tenstorrent_lower_composition.md)。下文原始范围和测试数量
不代表当前实现的全部能力。

## 1. 范围与设计选择

本轮采用静态有界窗口流水。对 extent 为 N、请求 stage 为 S 的独立循环，
实际窗口深度 D = min(S, N)，按最多 D 次迭代分组。默认 delayed 策略下，
每个窗口的 NCRISC 先发起全部输入 copy，再等待这些 copy 完成并发布输入，之后按原迭代顺序导出结果；
conservative 策略下每次输入 issue 后紧跟 completion，两种策略均在窗口输入完成后
按原顺序导出结果。TRISC 保留计算迭代顺序。循环被实际消费和展开，不以保留
`T.Pipelined` annotations、返回 structured IR 或仅在串行程序上增加 stage 数作为成功结果。

这种调度允许多个 copy 同时在途，具有短循环、首窗口和末尾不足 D 次迭代的
明确行为，但不是通用 modulo scheduling，也不承诺稳定态吞吐、NoC 重叠或硬件
异步执行效率。支持边界是完整 kernel body 的单个、静态、独立 pipeline loop；
静态 extent 限定为 1–1024，请求 stage 限定为 1–32；不扩展动态/嵌套 pipeline、
循环内条件分支、循环携带 DFB、Tensor input/output 交叠或 partial view。
通过 `T.Pipelined(..., annotations={"tt.pipeline_wait_policy": "conservative"})`
选择保守策略，`"delayed"` 是默认值；Form 实际消费这一选择，不做自动 wait 优化。

`num_stages=0` 且无手工调度时，common frontend 不发流水 annotation，继续走
Phase 4 serial 行为。显式 `order`/`stage`/`sync`/`group` 手工调度不在本轮范围。

## 2. 值身份、事务与容量

| 概念 | 本轮语义 |
| --- | --- |
| logical Buffer | 前端逻辑存储身份；同 Buffer 的不同写入不是同一个值 |
| DFB generation | `dfb_id` 唯一标识不可变的一次写入结果，沿用 Phase 4 值依赖规则 |
| iteration epoch | v3 使用展开后的 ordinal 显式标记迭代，不依赖动态执行计数恢复值身份 |
| transaction | 每个 generation 恰好一次 reserve、publication、最终 consumer release；descriptor relation 仍为 1 |
| pipeline stage | ordinal modulo D，表示窗口内逻辑 stage；既不是 processor slot，也不是 physical CB index |
| storage group | 单次 pipeline 迭代内有界展开后的同一写入位置，跨外层迭代共用池；不同位置及 export snapshot 不混同 |
| capacity | 一个池可同时持有的 payload block 数；不是 tile-grid 大小，也不是总 transaction 数 |
| release | consumer 最后使用且异步读取完成后归还 block；`dfb_wait` 本身不释放 |

Phase 4 保留的独立 Device For 使用 `(dfb_id, iteration epoch)`，descriptor relation
为 loop extent。v3 改用每个展开 generation 一个 `dfb_id`，descriptor relation 为 1，
迭代关系在独立 metadata 中表达。不得将 v2 的 extent relation 机械复制到每个 v3
展开 generation，否则 transaction 总数会被重复计算。

每个 storage group 的 requested/inferred `block_count` 必须一致。窗口调度默认每池
推导 D 个 block；显式容量须至少 D，足够时保留。两种 wait 策略沿用这一保守窗口
预算，不声明为实际最小容量。容量依据是同时 reserve 尚未 release 的 transaction；
copy 完成但未消费、保守 wait 以及 nested acquire 持有的旧 block 都必须计入。不同 write-site 不共享物理池。内部普通静态循环若被展开，其各次写入也视作不同
位置；这里的 write-site 指展开后的计划位置，不是只按原始源码行合并。
因此本轮没有实现跨变量/跨写入点的 lifetime coloring 或 L1 地址分配。

## 3. 跨 slot 调度与生命周期闭合

v3 的输入 transfer 只发起异步 copy；`tl.tt.dfb_copy_wait(id, 1)` 才完成输入
publication。consumer 的 `tl.tt.dfb_wait(id, 1)` 必须依赖该 publication。
计算输出在 `dfb_compute` 完成后发布。输出 transfer 发起异步读取后也需要
`dfb_copy_wait`，consumer 只能在该读取完成并经过最后一次使用之后 release。

有效顺序为：

```text
reserve -> copy issue -> copy completion/publication -> consumer wait
        -> consumer read(s) -> last-use completion -> release

reserve -> compute/publication -> consumer wait -> output copy issue
        -> output copy completion -> release
```

一次 immutable publication 可以对应同一 consumer slot 多次 wait/read，但不能
由多次 wait 推导出多次 transaction 或提前 release。nested acquire 合法，前提是
所有已持有 block 都保留到最后使用，且容量及等待关系不会形成环。

verifier 将 processor slot 内程序顺序、producer publication 到 consumer wait、
以及同 storage group 的 FIFO ring reuse 依赖统一放入依赖图。这里的 ring 是逻辑
容量/归还约束，物理 CB index 和地址仍未分配。显式容量大于 D 时也不能把
ordinal modulo D 当作物理地址。第 k 个 generation
复用容量为 C 的池时，依赖第 k−C 个 generation 的 release。该跨 slot 依赖是未来
Codegen 必须实现的 reserve backpressure 契约；不是假设三个 processor 以某一个
人为串行 trace 执行。依赖图必须无环；不足容量导致的等待环必须定位到资源池。
只有全部依赖闭合才允许声明该容量下不会覆盖尚未消费的数据。

## 4. Device IR schema 兼容性

经典 Phase 2 Add/no-op 仍使用 v1，Phase 4 通用 Lower 仍使用 v2，不要求旧消费者
接受新的生命周期约束。流水使用 v3；typed DFBDescriptor 构造签名不变，新增模块
metadata 和已注册 operation 表达协议。未知版本必须拒绝，不能退化成 v1 Add。

| v3 字段或 operation | 契约 |
| --- | --- |
| `tt.dfb_storage_groups` | `Map<String, Integer>`，规范十进制 generation ID 字符串 → storage group ID |
| `tt.pipeline_relations` | `Map<String, Array<Integer>>`，规范十进制 generation ID 字符串 → `[ordinal, ordinal % D]` |
| `tt.pipeline_stages` | 请求的正静态 stage 数 S |
| `tt.pipeline_extent` | 正静态迭代数 N；D = min(S, N) |
| `tt.pipeline_wait_policy` | 可观察的 `conservative` 或 `delayed` 策略；调度已体现在 operation 顺序 |
| `tl.tt.dfb_copy_wait(id, 1)` | 完成对应 generation 的异步输入 publication 或输出读取 |
| `tl.tt.dfb_release(id, 1)` | consumer 归还对应 generation 的 payload block |
| `tt.l1_capacity_bytes` | 可选显式 payload 预算；不是默认架构可用 L1 容量 |
| `tt.l1_payload_bytes` | 可选诊断 payload 计数；如附带此字段，verifier 独立重算检查 |

这两个 map 的 key 使用规范十进制字符串，以保证独立构造和序列化后的结构确定性；
DFB operation 的 ID 和 storage group ID 仍为 integer。禁止 `"00"`、`"bogus"` 等
非规范 key，且 key 必须引用真实 descriptor。不能使用 `Map<Integer, ...>` 的对象
identity 作为跨 Lower 的稳定 key。

verifier 独立验证映射完整性、generation/iteration/stage 对应关系、每个池恰好 N 个
transaction、shape/dtype/backing/producer/consumer 一致性、同步/释放闭合及资源重用。
reserve、publication、消费及 release 必须遵守 ordinal 顺序；compute 输入输出须属
同一 epoch。全池一致置换 epoch 也不能绕过顺序检查。
序列化 round-trip 后仍需接受完整 metadata，不借助前端 Buffer 或变量恢复关系。

## 5. Pass 职责与顺序

优先扩展已有 Pass，不新建平行的数据流分析路径：

```text
BindTarget -> CanonicalizeTTElementwise -> VerifyTTComputeBlocks
 -> ValidateTenstorrentFrontendIR -> NormalizeTenstorrentLaunch
 -> NormalizeTenstorrentBufferMetadata -> NormalizeTenstorrentRegions
 -> NormalizeTenstorrentTopology -> LegalizeTenstorrentTileOps
 -> InferTenstorrentTensorLayout -> FormTenstorrentDeviceProgram
 -> VerifyTenstorrentDeviceIR
```

- Canonicalize/compute verifier 允许外层静态 serial/unrolled loop binder 作为表达式
  标量模板参数，按 lexical context 验证；elementwise 坐标和未绑定 Var 仍拒绝。
  Form 展开后将 binder 代入常量并保留 source span，不允许自由 Var 泄漏到 Device IR。
- Legalize 继续消费计算 operation，允许已由 lexical scope verifier 认证的 Var 叶
  dtype，最终由 Form 替换为常量；其余 storage/computation FP 类型检查不变。它保留
  pipeline 迭代边界与调度信息，不决定 processor/pool；流水循环本身由 Form 消费。
- Form 在已有通用 DFB def-use 规划基础上生成窗口调度、不可变 generation、storage
  group、copy completion、last-use release 和容量 metadata，产出三个 Device slot。
- Verify 接受最终 Device IR，独立检查 transaction、生命周期、容量和显式预算；
  不补发缺失 wait/release，不修补错误的 relation 或不够用的容量。
- 共享 operation 注册及 schema 常量由主 agent 整合；完整 pipeline 仍由
  `tilelang/tenstorrent/pipeline.py` 持有，不增加 engine 内目标分支。

## 6. 容量和 L1 早期检查边界

静态 payload 字节数按每个唯一 storage group 计算一次：

```text
block_bytes = product(block_shape_in_tiles) * product(tile_shape)
              * element_dtype_bytes
payload_bytes = sum(group.block_count * block_bytes)
```

最后两轴 32×32 tile 的 BF16 payload 每 tile 为 2048 字节，FP32 为 4096 字节。
rank-1 Reduction 的物理 tile padding 也必须按物理 tile 计算，不能只算逻辑元素。
只有 shape、tile-grid、dtype、capacity 均静态已知且一致时才适用该计算。

v1/v2 若附带显式预算，则按每个独立 DFB descriptor 累加容量，不凭空推断旧版本
的跨 generation storage 复用；没有预算时保持原有 Lower 行为。

若用户提供显式 `tt.l1_capacity_bytes`，payload 超过该预算时可在 Device verifier
提前失败。小于预算只证明已表示 DFB payload 的必要条件，不证明实际分配可行。
本轮不计 TT-Lang allocator alignment/fragmentation、runtime/firmware 占用、代码、
stack、semaphore、未表示的 compute scratch/DST 或架构保留区；也不据架构名称猜测
free L1。实际地址、物理 CB index、allocator 复用及差分仍属延期验证。

## 7. 代表性 kernel 与可观察调度

`hyf-test/phase5_lower_demo/kernel.py` 使用两个 FP32 32×32 输入，执行
`C = A + B + epoch`，默认 epoch 从 2 到 8、7 轮、3 stages。它逐轮导出结果，
因此可以区分 stale generation 与正确的迭代常量，不只比较重复同一数值的最终输出。

该例有输入 A、输入 B、compute C、export snapshot 四个 storage group，各组默认
容量 3。28 个 immutable generation 共享这四个池；payload 按池计为
`4 * 3 * 32 * 32 * 4 = 49152 bytes`，而不是把 28 个 generation 都分配独立容量。
窗口依次包含 ordinal `[0,1,2]`、`[3,4,5]`、`[6]`。delayed 策略第一窗口先发起
6 笔输入 copy，然后显式完成并允许消费；第三窗口只有两笔输入 copy。所有输出
snapshot 读取结束后归还，下一次同池 reserve 受旧值 release 的 backpressure 约束。

复现并生成 Device IR 的命令：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-phase5-cache .venv/bin/python hyf-test/phase5_lower_demo/kernel.py
```

默认产物目录是 `hyf-test/phase5_lower_demo/generated/`，包含 frontend、最终
`device_ir.tir`/JSON、三个 processor slot、`verification.json` 和 `observation.md`。
`--wait-policy conservative` 可观察逐 issue 的 completion，`--stages`/`--extent`/
`--start` 可改变窗口与边界。产物的实际验证状态见下一节。

## 8. 实现与验收记录

本轮已完成修改 C++ 的 native 重编译和链接，再执行最终 Device IR 回归。验收
最终单次完整回归为 **485 passed、29 warnings，3.40 秒**。其中原有 373 项与
Phase 5 新增 112 项全部通过；包括 `stages=32, extent=1024` 上界和 12 个正例的
Device script/JSON 字节确定性断言。29 个 warning 是现有 Python typing 弃用提示。

- Phase 5 pipeline：28 项，覆盖 stage 1/2/3/32、stage 大于
  extent、非零起点、短循环/尾窗口、零 stage 回退、显式容量、保守/延迟完成以及
  nested/manual/conditional 等诊断边界。
- Phase 5 resources：37 项，覆盖每池容量与完整 epoch、多个在途 copy、nested
  acquire、缺失 wait/release/copy completion、提前 release、释放后使用、容量不足、
  release→reuse 等待环、错误 relation/全池 epoch 置换、规范 key、L1 恰好预算与
  溢出检查；v1/v2 显式预算也有回归。
- Phase 5 semantics：47 项，独立 NumPy 模型运行最终 Device IR，使用有限容量存储、
  copy 延迟及不同 slot 调度，逐 iteration 检查导出值；覆盖流水与 serial 的数值
  等价、dtype、stage/边界及显式容量大于窗口深度。
- 原有 373 项继续通过，包括 Phase 2 Add、Phase 4 compute/control/device/semantics、
  backend import 及语言 Tiles，保留 v1/v2 既有 Lower 行为。
- 重复 Lower 的结构确定性、Device script/JSON 字节相等、JSON round-trip 后
  verifier 和生命周期闭合已通过。
  v3 map 使用规范字符串 key，独立构造不依赖 Integer 对象 identity。
- 默认代表性 kernel 已生成最终 Device IR：7 轮、3 stage、28 generation、4 pool、
  49152 字节 payload；模型输入 copy 在途峰值 6，7 次输出各自与 NumPy 相等，误差 0。
  生成目录为 `hyf-test/phase5_lower_demo/generated/`。
- `generated-conservative/` 的第二份产物使用 blackhole、conservative、4 stage、
  2 轮、起点 5：8 generation、4 pool、32768 字节 payload；模型 copy 在途峰值 1，
  两次输出误差为 0。两种 target arch 仅验证 Lower metadata，不代表两种硬件运行。
- 源码经 clang-format 22.1.8、Ruff 0.16.1 check/format 和 `git diff --check` 验证。
  完整 pre-commit hook 初始化因代理不可达未完成，使用缓存中的同版本工具直接验证；
  不把这些直接检查记为完整 pre-commit 运行成功。

参考模型只证明最终 Device IR 的逻辑数值语义及 dtype 舍入；不能替代 TTL 编译、
设备同步和硬件数值验证。本轮不实现 Phase 6 多 Core 或 PipeNet 通信。

完整回归复现命令（先 import tilelang 以初始化本仓库 TVM 依赖路径）：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-phase5-cache .venv/bin/python - <<'PYTEST'
import tilelang
import pytest
from pathlib import Path
paths = sorted(str(p) for p in Path('testing/python/target').glob('test_tilelang_tenstorrent*.py'))
paths += ['testing/python/backend/test_tilelang_tenstorrent_import.py',
          'testing/python/language/test_tilelang_language_tiles.py']
raise SystemExit(pytest.main(paths + ['-q', '--tb=short']))
PYTEST
```

C++ 变更必须先按 `.agents/skills/tilelang-build/SKILL.md` 重编译和链接，再运行上述
Python 测试；仅 collect-only 不计实际验收。
