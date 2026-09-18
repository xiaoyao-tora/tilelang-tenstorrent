# Phase 6 多 Core 与 PipeNet Lower 契约

本文件补充工作计划第十七章。本轮在 Phase 5 基础上实现静态多 Core 的 Lower、
最终 Device IR 和独立 verifier。Phase 3 继续跳过；TTL Codegen、TT-Lang
compile-only、Simulator、硬件正确性和长时间无死锁验证均不在本轮范围。
下文的通信完成、同步和容量约束是 Device IR 的逻辑协议；未来 Codegen 必须实现
同一协议，才能将这些结论延伸到设备执行。

## 1. 通信语义与范围

Core 坐标按 `(x, y)` 解释，grid 是正静态二维整数，domain 是 grid 内的半开矩形。
本轮对每个 Core 静态特化其程序，保留确定的 processor slot 顺序；grid 不是 CUDA
线程数，Core 数量也不是 Pipe transaction 数量。Core domain、Pipe endpoint 和
每一笔 transaction 的参与者都必须可静态确定，不依赖运行时 Tensor 数据。
本轮静态展开上限为 256 个 Core，单 PipeNet 上限为 4096 条 record；它们是编译器
的有界展开限制，不是 Tenstorrent 硬件规格。

| 模式 | 正确语义 | 本轮能力边界 |
| --- | --- | --- |
| P2P | 一个 source Core 向一个 destination Core 传递一次完整 payload | 静态 endpoint，tile 合法的完整 shared region |
| broadcast / multicast | 同一次 source payload 复制给 destination domain 内每个 Core | 逻辑 multicast 展开为确定的接收集合，不承诺物理 multicast 指令 |
| gather | 多个 source 的独立 payload 按 PipeNet record 顺序在一个 destination 接收 | 接收本身不隐含 concat、sum 或 reduction；计算必须显式表示 |
| true scatter | 同一 source 给不同 destination 发送内容可区分的 payload | 必须由 record/endpoint 选择不同输入或计算；同一 payload 多目的只是 broadcast |
| 二维 GEMM 行列分发 | A panel 沿 Core 行分发，B panel 沿 Core 列分发，每 Core 计算自己的 C block | 静态有界分发加已有 GEMM Lower；不等于完整动态 SUMMA/runtime 策略 |

二维 GEMM 的 Core `(x, y)` 对应输出 block `(row=y, column=x)`。A panel 与 y
相关，沿 x 方向分发；B panel 与 x 相关，沿 y 方向分发。source 本地直接使用
自己的 panel；不需要为了对齐接收端路径而伪造 self-send。长度为 1 的行/列没有
远端传输，应直接保留本地值，不能构造空 PipeNet。显式 self-Pipe 或包含 source 的
multicast 也可表示；它们同样建立 snapshot、receiver generation 和同步边，只有
整个依赖图闭合且无环时才接受，不能将 self-Pipe 静默当作 no-op 删除。

## 2. foreach、record identity 与 processor 责任

`PipeNet` 是有序 record 列表，不是去重后的图。operation-local net identity 与
原 record index 共同标识一条 record；相同 endpoint 的重复 record 仍是独立事件。
schema 预留 transaction occurrence，以区分同一 record 的多次使用；本轮仅支持
每 record 一次发送，`occurrence=0`，重复使用同一 record 会明确拒绝。不同 record
即使 endpoint 相同也仍独立。不能将 record 数、发送数、接收数或 payload tile 数
混为一谈。
只要一个 PipeNet 实际包含 transfer，本轮要求它的全部原 record 都各有一次 source
发送和完整 destination 集。Normalize 保存原表，Form 检查 guard 同时删掉某条
record 的 send/recv 也不能静默改变 multiplicity。仅作 predicate 或无 transfer 的
foreach 所引用的 PipeNet 不建立通信资源，这是合法的零通信程序。

`foreach_src` / `foreach_dst` 在每个 Core 上按原 record 顺序选择匹配项：

- 零匹配：不执行 body，不产生占位 send/recv 或 transaction。
- 一匹配：恰好执行一次，selected PipeRef 指向原 record。
- 多匹配：每条匹配 record 执行一次；重复 record 不合并。

selected PipeRef 是编译期 lexical 对象，不能逃逸到 foreach 外；其值是原 record
索引，不是当前 Core 的局部匹配序号。`pipe_src`、P2P 的 `pipe_dst` 和 collective 的
`pipe_dst_range` 必须在特化后成为静态坐标/范围。`is_src` / `is_dst` / `is_active`
分别是 source 集、destination 集、二者并集的成员测试。

Pipe-source 的通信操作固定分配 BRISC；接收与本轮 v4 Tensor copy 固定分配 NCRISC，
计算继续由 TRISC 承载。BRISC affinity 不表示 BRISC 是固定 writer，
也不表示三个 slot 按函数排列顺序串行执行。最终同步检查必须按 `(Core, slot)`
划分各自程序顺序，并显式连接跨 slot、跨 Core 的数据依赖。

## 3. transaction、同步与资源不变量

每一个接收 payload 是独立的 immutable DFB generation。send 与 recv 必须通过
明确的 net、record、occurrence 和 endpoint 对应，不可仅凭相同 shape/dtype 或
相同 Core pair 猜测匹配。合法 multicast 的每一个 destination 都必须闭合；
source publication 一次，不等于只有一次接收 transaction。

若某笔接收值被后续接收覆盖且没有数值消费者，仍须完成这笔通信；不能把它作为
dead compute 删除。本轮用 NCRISC 上的 `dfb_wait` + `dfb_release` 表示 drain/discard，
此特例只允许真正的 incoming Pipe generation，且 release 必须位于接收 completion
之后。它不产生数值读取，但保留 gather/重复 record 的 transaction multiplicity。

发送前通过 TRISC copy 创建 source snapshot，快照由 BRISC 独占消费；原 generation
继续由 TRISC 本地计算或创建其他快照。这样保留 DFBDescriptor 的单 consumer slot
契约，不用把 BRISC 和 TRISC 的双消费者隐藏在一个 descriptor 中。不同 multicast
destination 共享这一 record 的 snapshot，并在全部 send completion 后释放。
该方案增加逻辑 copy 和独立存储；本轮不声明它是最小 L1 或最优通信计划。

所有 producer 写入都必须先 reserve。consumer 读取必须等待对应 publication，
最后一次读取和异步通信完成后才能 release。发送源值必须一直存活到所有使用它的
send 完成；接收目标必须先有可写容量，才允许发布收到的数据。缺失或重复的
reserve/publication/completion/release、提前释放、释放后读取和跨 Core 误用均应失败。

verifier 的依赖图须同时包含每个 `(Core, slot)` 的程序顺序和通信/DFB 依赖。
无法闭合的 producer/consumer 和 wait-for 环应在 Lower 阶段失败，不依赖未来运行时
超时来发现问题。静态无环只证明 Device IR 协议中没有已表示的等待环；不证明
真实 NoC routing、semaphore 实现、硬件公平性或运行时调度没有死锁。

容量以同时存活的 payload block 计数。tile-grid 大小决定一个 block 的字节数，
`block_count` 决定保留多少 block；transaction 数不会自动增加可用容量。每 Core
的 payload 预算独立计算，不把全 grid 的总 payload 错当成单 Core L1 占用。
即使 payload 小于显式预算，也没有证明物理 L1 分配可行：地址、CB index、代码、
runtime、semaphore、scratch、alignment 和 fragmentation 仍属延期事项。

本轮 v4 每 generation 独立分配，不做跨 generation pool 复用，要求
`1 <= block_count <= 32`，按每 Core 独立 generation 的 capacity×payload 累计预算。因此本轮
容量检查与 Phase 5 的流水 pool/ring 容量推导不是同一能力；跨 Core PipeNet 与
`T.Pipelined` 组合明确拒绝，不能因单 Core v3 测试通过而将其计为完成。

## 4. Device IR v4 与 schema 兼容性

经典 Add/no-op 保留 v1，Phase 4 通用计算保留 v2，Phase 5 单 Core 流水保留 v3。
本轮静态多 Core/PipeNet 使用 v4，未知 schema 必须拒绝。已有 DFBDescriptor 和
PipeDescriptor 构造签名保持兼容，不把旧版本的 processor ABI 解释成多 Core ABI。

v4 为每个 Core 生成 TRISC、NCRISC、BRISC 三个 PrimFunc，其 `tt.core_domain`
恰好是该 Core 的 singleton 矩形。稳定顺序是先 x、再 y、最后 slot
`trisc/ncrisc/brisc`；`tt.kernel_order` 保存完整 global function name，从而区分
不同 Core 上同名 processor slot。这只是序列化顺序，不是执行顺序。

`tt.pipe_table` 保留 typed PipeDescriptor，`event_index` 是原 record index。
v4 新增 `tt.pipe_transfer_table`，成员为 typed `PipeTransferDescriptor`：

| 字段 | 契约 |
| --- | --- |
| `transfer_id` | operation 内唯一、确定的展开通信身份 |
| `pipe_net_id` / `record_index` | 引用原 PipeNet record，重复 record 不去重 |
| `occurrence` | 预留多次发送序号；本轮必须为 0 |
| `src_coord` / `dst_coord` | 一笔展开 P2P transfer 的静态坐标 |
| `source_dfb_id` / `destination_dfb_id` | 源 generation 与接收 generation，不依赖前端 Buffer 恢复关系 |
| `transaction_count` | 本轮每笔展开 transfer 必须为 1 |
| `source_span` | 诊断来源 |

verifier 要求 transfer ID 按表连续且唯一，并检查每个 `(Core, slot, net)` 的
send/recv 遵循原 record 顺序。它不把语义等价的 transfer ID 重新编号当成错误；
Lower 的字节确定性通过重复生成与序列化测试保证。

multicast 由同一 record 的多个 transfer 表达，保留原 domain 与展开 endpoint 的
对应关系。每一笔 transfer 使用 `tl.tt.dfb_pipe_send(transfer_id, source_dfb_id, 1)`
和 `tl.tt.dfb_pipe_recv(transfer_id, destination_dfb_id, 1)` 表示 issue；相应 Core
上的 `tl.tt.dfb_pipe_wait(transfer_id, dfb_id, 1)` 表示 source 读取完成或 destination
publication。通信 wait 与 consumer `dfb_wait` 职责不同，不能互相替代。
当前每笔 Pipe issue 后紧跟其 completion wait；multicast 的 source 按 destination
展开顺序逐笔完成。这是保守的逻辑调度，不复用 Phase 5 的 delayed 窗口策略，也
不宣称多笔跨 Core 通信在途或物理通信重叠。

Device verifier 从最终 descriptor、body、singleton domain 独立验证匹配与同步，
不回读前端 PipeNet JSON、不重新运行 topology normalization，也不补发遗漏 operation。
最终结果必须是带 `tt.device_ir_version=4` 的完整 Device module，不能把 Core SBlock
或通信 annotation 仍留在 structured IR 中作为成功 Lower。

## 5. Pass 顺序与职责

本轮扩展现有 Pass，无新增平行的数据流 Pass。完整顺序仍由后端 pipeline 拥有：

```text
BindTarget -> CanonicalizeTTElementwise -> VerifyTTComputeBlocks
 -> ValidateTenstorrentFrontendIR -> NormalizeTenstorrentLaunch
 -> NormalizeTenstorrentBufferMetadata -> NormalizeTenstorrentTopology
 -> NormalizeTenstorrentRegions -> LegalizeTenstorrentTileOps
 -> InferTenstorrentTensorLayout -> FormTenstorrentDeviceProgram
 -> VerifyTenstorrentDeviceIR
```

- NormalizeTopology 输入带 typed Buffer metadata 的 frontend PrimFunc；读取 launch
  grid 与 PipeNet，按 Core 常量代入 predicate、foreach 和坐标，输出带 `tt.core_x/y`
  的静态 Core SBlock 和 `tt.topology_records` 原 record 表。此中间结构尚不是最终
  Device IR，Form 完整消费原表以检查 active PipeNet 的 record 覆盖。
- NormalizeTopology 从原先的 Regions 之后前移到 Regions 之前：Core-dependent
  Tensor slice 必须先常量化，Regions 才能证明边界与 extent。这是明确的依赖重排，
  不是放宽 Regions 对不可证明动态访问的要求。
- Regions 验证已静态化的 source/destination region；Legalize 消费支持的计算，
  Layout 推导 tile-grid。它们不分配 NoC transport 或物理地址。
- Form 完全消费 Core SBlock 与通信操作，推导 generation、三 processor slot、
  PipeTransferDescriptor 和同步生命周期，产出每 Core 三个 Device PrimFunc。
- Verify 接受最终 module，验证 topology 引用、每 Core slot 完整性、transaction、
  publication/consumer/release、跨 Core 依赖无环以及每 Core payload 预算。

## 6. 尚未实现与延期验证

尚未实现：动态 topology/grid、跨设备 PipeNet、每 record 多 occurrence、跨 Core
流水、跨 generation pool 复用和物理容量 coloring、任意部分 shared payload/view、
隐含 gather reduction、通用动态 SUMMA 调度和物理 NoC route/transport/semaphore。
已有 Phase 4 dtype、GEMM/compute shape 与 alias 限制继续有效。v4 Tensor 只支持
无 alias 的 interleaved DRAM、compact row-major strides；Tensor inout 顺序不支持。
不同 Core 输出 region 重叠会拒绝，不推断某个 Core 是最终 winner。

延期验证：TTL Codegen 与 TTL parser/verifier、TT-Lang guard/schedule/provenance/SPSC
及 compile-only、物理 L1/CB 分配差分、Simulator、硬件数值与性能、长时间无死锁。
独立 NumPy 模型只能验证最终 Device IR 的逻辑数值和依赖语义，不称为模拟器或
硬件执行认证。

## 7. 实现与验收记录

修改的 native C++ 已重编译并链接，最终完整回归为 **580 passed、29 warnings，
3.94 秒**。Phase 6 新增 **95 项**：Lower 24、resources 39、semantics 32；Phase 2
Add、Phase 4、Phase 5、backend import 与语言 Tiles 的原有 **485 项**全部通过。
warnings 为现有 Python typing 弃用提示。设计目标、前端构造或测试 collect 成功
没有计为最终 Device Lower 验收。

| 测试组 | 实际覆盖 |
| --- | --- |
| Lower，24 项 | P2P、collective、true scatter、gather、反序 source、重复 record、zero/one/many、discard、self-Pipe、包括 source 的 multicast、2D GEMM、无 Pipe 多 Core、256 Core 上界、确定性/序列化/idempotence；257 Core、重复 occurrence、缺 receiver、跨 Core pipeline、guard 丢 record 及 malformed schema 拒绝 |
| Resources，39 项 | producer/consumer 和 multicast 接收集闭合，record 顺序、transaction 数、重复 op、missing completion/wait/release、提前 release/释放后使用、incoming-only discard、capacity 与每 Core 预算、跨 Core output overlap、等待环、schema/slot/Core 身份与未知资源引用 |
| Semantics，32 项 | 最终 Device IR 经独立逻辑模型运行：FP32 P2P/multicast、true scatter/重复覆盖、gather/反序/discard、2×2 source 兼 compute GEMM、3×3 专用分发 GEMM；BF16 多 tile 与舍入；不同 slot 调度和异步传输延迟 |

Lower 与 resources 证明静态契约，semantics 模型证明受支持的逻辑数值行为。
三者都没有运行 TTL、TT-Lang compile-only、Simulator 或硬件，不能替代这些延期
验证。Phase 5 的流水回归通过，也不表示跨 Core PipeNet 与流水组合已支持。

代表性 kernel 为 `hyf-test/phase6_lower_demo/kernel.py`：在 3×3 逻辑 grid 中，
Core `(0,1)`、`(0,2)` 分别发送 A 的两行 panel，Core `(1,0)`、`(2,0)` 分别发送
B 的两列 panel；四个 worker `(1,1)`、`(1,2)`、`(2,1)`、`(2,2)` 分别接收 A/B
并计算互不重叠的 32×32 C block，`(0,0)` 空闲。这里采用专用分发 Core，不将
该示例描述为 source 同时参与计算的完整 SUMMA 实现。

输入 A 为 FP32 64×32，B 为 FP32 32×64，C 为 FP32 64×64。两个 PipeNet
各有两条 collective record，每条 record 有两个 destination，共四条 record、
八笔 point delivery。默认 `generated/` 已生成 frontend/final Device IR、JSON、
27 个 Core/slot 文件、`verification.json` 与 `observation.md`；
`generated-blackhole/` 是另一 arch metadata 的同等产物。

两份产物均有 **24 个独立 DFB generation**。四个 sender 各使用输入 generation
加 send snapshot，`2 * 32 * 32 * 4 = 8192 bytes`；四个 worker 各使用接收 A、
接收 B、GEMM C、export snapshot，`4 * 32 * 32 * 4 = 16384 bytes`。此例显式
block_count=1，因此每 Core 统一预算 **16384 bytes** 通过 verifier；idle Core
没有 DFB payload。全 grid 合计 payload 不用于判断单 Core 的 L1 预算。

两种 arch 的 seed 0、1、19 三种逻辑调度下，输出与 NumPy `A @ B` 相等，最大绝对
误差均为 **0**。重复 Lower 的 script/JSON 字节相同、JSON round-trip 后 verifier
均已通过。架构 metadata 通过不表示两种设备均已运行。

3×3 kernel factory 放在 tracked 的
`testing/python/target/test_tilelang_tenstorrent_phase6_lower.py`，demo 复用该 factory；
tracked 测试没有依赖被 Git ignore 的 `hyf-test` 文件。完整 CMake native build
成功，最后 native 修改另经 CMake 生成的 compile/link 命令重新构建；使用直接命令
是因 Ninja log 的 premature-recovery 导致反复全量，不跳过 native 编译。旧 header
warning 不影响链接。Ruff check/format 与 `git diff --check` 通过。

复现：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-phase6-cache .venv/bin/python - <<'PYTEST'
import glob
import tilelang
import pytest
paths = sorted(glob.glob('testing/python/target/test_tilelang_tenstorrent_*.py'))
paths += ['testing/python/backend/test_tilelang_tenstorrent_import.py',
          'testing/python/language/test_tilelang_language_tiles.py']
raise SystemExit(pytest.main(paths + ['-q', '--tb=short']))
PYTEST
TILELANG_CACHE_DIR=/tmp/tilelang-phase6-cache .venv/bin/python hyf-test/phase6_lower_demo/kernel.py
TILELANG_CACHE_DIR=/tmp/tilelang-phase6-cache .venv/bin/python hyf-test/phase6_lower_demo/kernel.py --arch blackhole --output hyf-test/phase6_lower_demo/generated-blackhole
```

运行前须按 `.agents/skills/tilelang-build/SKILL.md` 重编译最新 native 修改；仅用旧
shared library 跑 Python 测试不计本轮验证。
