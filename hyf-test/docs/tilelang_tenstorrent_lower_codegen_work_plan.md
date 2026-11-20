# TileLang Tenstorrent Lower 与 TTL Codegen 工作计划

## 1. 文档目标

本文定义 TileLang Tenstorrent backend 中两个连续阶段的工作：

```text
阶段一：Lower
Frontend TIRX -> Device TIR IRModule

阶段二：Codegen
Device TIR IRModule -> TTL MLIR Module
```

第 1–16 章包含整体设计目标，不应全部解读为已实现能力。当前交付状态以第十七章
的分阶段矩阵和验收记录为准；Phase 6 本轮仅推进 Lower/Device IR/verifier，Phase 3
及所有 TTL/TT-Lang/Simulator/硬件执行验证继续延期。

前端原语以 `code/tilelang_ttlang_frontend_primitives.md` 为输入契约。相关原语已在
其他分支完成，本文不重新设计或实现这些原语。合入该分支时仍需核对实际 TIRX
节点、属性名和对象身份是否符合本文假设。

本文覆盖：

- Frontend TIRX 的消费边界；
- Lower 阶段的分析、规划、Kernel 划分和 Device TIR 物化；
- Device TIR `IRModule` 的 Module、PrimFunc、资源和操作契约；
- Codegen 阶段的类型转换、SSA 构造、TTL operation 生成和 Module 组装；
- 与 `TTLGenericCompiler` 的对应关系；
- verifier、诊断、测试、代码布局和分阶段交付。

本文不覆盖：

- TileLang 前端原语实现；
- `TTL -> TTKernel -> EmitC -> C++` 的具体 pass 实现；
- TTNN Program、缓存、设备管理和 Kernel launch；
- 跨设备 PipeNet；
- TTKernel NoC、semaphore 或 CB 物理指令的手写生成。

## 2. 总体架构

正式编译主线为：

```text
+----------------------------------------+
|              TileLang DSL              |
+----------------------------------------+
                    |
         TileLang eager frontend
                    |
                    v
+----------------------------------------+
|             Frontend TIRX              |
+----------------------------------------+
                    |
            Tenstorrent Lower
                    |
                    v
+----------------------------------------+
|          Device TIR IRModule           |
|                                        |
|            - TRISC PrimFunc            |
|           - NCRISC PrimFunc            |
|            - BRISC PrimFunc            |
+----------------------------------------+
                    |
         Tenstorrent TTL Codegen
                    |
                    v
+----------------------------------------+
|        initial TTL MLIR Module         |
+----------------------------------------+
                    |
     TT-Lang ttl-to-ttkernel pipeline
                    |
                    v
+----------------------------------------+
|    TTKernel MLIR / EmitC MLIR / C++    |
+----------------------------------------+
```

各阶段的核心职责如下：

- Frontend TIRX 表达用户的统一逻辑程序；
- Lower 决定语句归属、逻辑资源、Core domain 和 backend slot；
- Device TIR 是 Codegen 的唯一权威输入；
- Codegen 确定性地把 TIRX 类型、语句和 metadata 转为 TTL MLIR；
- TT-Lang 继续负责同步补全、物理 DFB、Pipe transport、DST、循环调度和 TTKernel
  lowering。

Codegen 可以理解为“面向 Device TIRX 的 `TTLGenericCompiler`”，但不复用
`TTLGenericCompiler` 的 Python AST、closure 或 TTNN Tensor capture 机制。

## 3. 术语与阶段边界

### 3.1 Frontend TIRX

Frontend TIRX 是 TileLang eager frontend 产生的 `tvm.tirx.PrimFunc` 或
`tvm.IRModule`。它保留：

- `T.Kernel` 的逻辑 launch nest；
- Tensor 参数和 `match_buffer`；
- `T.alloc_shared` 及 DFB metadata；
- `tl.tileop.copy`、GEMM、Reduction、Fill、Transpose 等高层 TileOp；
- `T.Parallel`、`T.Pipelined`、条件和循环；
- BufferLoad/BufferStore 表达式 DAG；
- Pipe、PipeNet、CoreRange、foreach 和 PipeRef copy；
- source span。

Frontend TIRX 仍是一个统一程序，尚未形成 TRISC、NCRISC、BRISC 三个设备函数。

### 3.2 Lower

Lower 是目标相关的语义变换阶段：

```text
Frontend TIRX
    -> validate
    -> normalize
    -> analyze
    -> plan
    -> partition
    -> materialize
    -> verify
    -> Device TIR IRModule
```

Lower 可以使用短生命周期、不可变的 typed analysis plan，但这些 plan 不是公开 IR，
不序列化为另一套 `TTProgramGraph`，也不成为 Codegen 的旁路输入。原子改写完成后，
所有 Codegen 必需信息都必须存在于 Device TIR 中。

### 3.3 Device TIR IRModule

Device TIR 仍使用 TVM TIRX `IRModule`、`PrimFunc`、`Stmt`、`PrimExpr`、`Buffer` 和
target-specific Call/Attr，不引入另一套通用表达式 IR。

一个 TileLang operation 对应一个设备程序。当前 TT-Lang backend 对每个设备程序
提供三个固定 processor slot：

| Slot | thread kind | 必需属性 | 主要用途 |
| --- | --- | --- | --- |
| TRISC | compute | `tt.kernel_thread=compute` | elementwise、GEMM、reduce、store |
| NCRISC | data movement | `tt.noc_index=0` | 普通 Tensor/DFB copy、Pipe destination |
| BRISC | data movement | `tt.noc_index=1` | 第二个 DM slot，当前优先承载 Pipe source |

Device TIR 中三个 slot 都必须存在。无工作 slot 使用合法空函数体，不通过缺少函数来
表达 idle。

### 3.4 Codegen

Codegen 只接受通过 Device TIR verifier 的 `IRModule`。它负责：

- 创建并初始化 MLIR Context；
- 注册 TTL、TTCore、TTKernel 以及必要的上游 dialect；
- 转换 Tensor、Tile、DFB、Pipe 和 scalar 类型；
- 为三个 slot 创建 `func.func`；
- 建立 TIR Var/Buffer/resource ID 到 MLIR SSA Value 的映射；
- 生成 `ttl.*`、`arith.*`、`scf.*` 和 `func.*` operation；
- 设置 Module/Function attributes 和 `Location`；
- 组装并验证 initial TTL MLIR Module；
- 输出确定性的 TTL source module。

Codegen 不重新执行 kernel partition、DFB ownership 推导或 Core topology 推断。

## 4. 两阶段职责表

| 语义 | Lower 输出 | Codegen 工作 | TT-Lang 后续工作 |
| --- | --- | --- | --- |
| Tensor ABI | 参数顺序、读写类型、shape、dtype、layout | 构造 encoded ranked tensor type 和函数参数 | runtime 参数物化 |
| Core grid | 静态二维 grid、Core domain | 设置 `ttl.launch_grid`，生成 core 查询/predicate | 可选 core specialization |
| Kernel | logical identity、slot、thread kind、函数体 | 生成 `func.func` 和 kernel attributes | 转换为 TTKernel thread |
| DFB | logical ID、block shape、tile、capacity、owner | 生成 `ttl.bind_cb` 和 acquire | 物理 index、复用、预算检查 |
| Copy | 方向、region、owner、completion dependency | 生成 `ttl.tensor_slice`、`ttl.copy` 和必要的显式 wait | 补齐缺失 copy wait |
| DFB 生命周期 | producer/consumer transaction 和 acquire scope | 生成 `ttl.cb_reserve`/`ttl.cb_wait`/`ttl.attach_cb` | 补齐或规范化 push/pop |
| Compute | codegen-ready tile expression/TileOp | 生成 tensor-level compute op 和 `ttl.store` | compute/tile 化、DST 和 schedule |
| PipeNet | 有序边、重复事件、endpoint、Core domain | 生成 Pipe/PipeNet region 和 transfer op | guard/schedule 验证和 transport |
| 物理资源 | 不固定 | 不固定 | CB index、DST、semaphore、NoC protocol |

## 5. Frontend TIRX 输入契约

Lower 入口必须验证前端分支提供的实际契约，不能从名字或访问模式猜测原语语义。

### 5.1 Launch 契约

- `T.Kernel(x, y, threads=1)` 提供静态二维逻辑 Core grid；
- `blockIdx.x/y` 表示逻辑 Core 坐标；
- `threadIdx` extent 必须为 1，且不能存在影响结果的 `threadIdx` 数据依赖；
- launch grid、CoreRange 和 Pipe endpoint 必须使用一致坐标系；
- 一个待编译 operation 只能有一个权威 launch grid。

Lower 必须在 `MaterializeKernelLaunch(false)` 丢弃 thread extent 或 CUDA
`MaterializeKernelLaunch(true)` 赋予 SIMT 语义之前读取 launch 信息。

### 5.2 Tensor 与 Buffer 契约

每个参数 Buffer 必须保留：

- 参数序号和 source parameter identity；
- element shape、dtype、stride 和 scope；
- input/output/inout effect；
- alias、view 和 in-place 约束；
- tile layout、memory layout 和 sharding metadata；
- source span。

每个 `T.alloc_shared` 候选 DFB 必须保留稳定 Buffer identity，以及以下 metadata：

```text
tt.tile_shape
tt.dfb_block_count
tt.tensor_backed
```

其中 `tt.tensor_backed` 必须指向稳定的 Tensor 参数 ID 和 byte offset，不能保存运行时
TTNN Tensor 对象。

### 5.3 Copy 与计算契约

普通 `T.copy` 必须保留完整 source/destination `BufferRegion`。Pipe copy 必须在
Frontend TIRX 中已经区分：

```text
BufferRegion -> BufferRegion
DFB block -> source PipeRef
destination PipeRef -> DFB block
```

计算必须保留以下至少一种 codegen 可识别形式：

- `tl.tileop.gemm/reduce/fill/transpose/copy`；
- canonical `T.Parallel` 下单一输出 BufferStore 及其表达式 DAG；
- target-specific、结构化的 topology/Pipe region。

Lower 不从 CUDA intrinsic、标量化 DMA 循环或已删除的 annotation 恢复高层语义。

### 5.4 Topology 契约

PipeNet 必须显式保留：

- operation-local PipeNet identity；
- Pipe 构造顺序；
- 重复 Pipe 的事件身份；
- point-to-point 与 collective contract；
- source coordinate 和 destination CoreRange；
- `foreach_src`/`foreach_dst` 的结构化 region；
- selected PipeRef 的 source/destination 类型；
- `is_src`、`is_dst`、`is_active` predicate。

重复 Pipe 不能因结构相等而被 CSE 或去重。

## 6. Device TIR 总体契约

### 6.1 唯一权威

Device TIR 必须是 Lower 的完整输出，不允许 Codegen 同时读取：

- Python closure；
- 前端 builder 对象；
- TTNN Tensor；
- PassContext side channel；
- 全局可变表；
- 临时 JSON graph；
- 另一个持久化 program graph。

允许 Lower 在一次 pass 内使用本地 typed plan。允许把 Device TIR 打印为调试文本，
但重载后必须仅靠 `IRModule` 恢复同样的 TTL 输出。

### 6.2 Module 级 metadata

Device TIR Module 至少需要以下类型化信息。具体属性名在实现前冻结；下列名称为本文
建议的内部契约：

| 属性 | 内容 |
| --- | --- |
| `tt.device_ir_version` | Device TIR schema version |
| `tt.target_arch` | `wormhole_b0` 或 `blackhole` |
| `tt.launch_grid` | 两个正整数 |
| `tt.operation_identity` | 稳定 operation ID |
| `tt.tensor_table` | Tensor ABI descriptor 数组 |
| `tt.dfb_table` | logical DFB descriptor 数组 |
| `tt.pipe_table` | 保序 Pipe/PipeNet descriptor 数组 |
| `tt.kernel_order` | TRISC、NCRISC、BRISC 的稳定顺序 |

Descriptor 应实现为只保存数据的 TVM Object/Node 或其他类型化 ObjectRef，不使用
自由格式字符串和层层嵌套的无类型 `Map<String, ObjectRef>`。

### 6.3 Tensor descriptor

每个 Tensor descriptor 至少包含：

```text
global_arg_index
shape
dtype
strides
tile_shape
tile_grid_shape
memory_space
memory_layout
shard_spec
effect = input | output | inout
alias_group
source_span
```

Tensor descriptor 只描述编译期 ABI，不包含地址、device handle 或 live Tensor。

### 6.4 DFB descriptor

每个 logical DFB descriptor 至少包含：

```text
dfb_id
source_buffer_identity
element_dtype
tile_shape
block_shape_in_tiles
block_count
tensor_backing
producer_slot/domain
consumer_slot/domain
transaction_count_or_loop_relation
source_span
```

约束：

- `dfb_id` 在整个 operation 内唯一并跨三个 PrimFunc 保持一致；
- provisional CB index 可以等于 `dfb_id`，但不具有物理含义；
- 物理 CB index、复用和最终 L1 address 不进入 Device TIR；
- `block_count` 表示完整 block 数量，不是 tile 数；
- tensor backing 使用 Tensor ABI index，不使用 Buffer 名字；
- 每个 launch node 上必须可证明为单 producer、单 consumer。

### 6.5 Pipe descriptor

每条 Pipe record 至少包含：

```text
pipe_net_id
event_index
src_coord
dst_begin
dst_end
contract = point_to_point | collective
payload_dfb_id
source_span
```

`event_index` 保留用户构造顺序和重复事件。任何 map/set 只可用于查询，不能决定最终
输出顺序。

### 6.6 PrimFunc 契约

每个 operation 固定生成三个 slot PrimFunc。每个函数至少带：

| 属性 | 说明 |
| --- | --- |
| `global_symbol` | 稳定、唯一的函数名 |
| `calling_conv` | `DEVICE_KERNEL_LAUNCH`，供 host/device Filter 识别 |
| `target` | Tenstorrent Target |
| `tt.kernel_slot` | `trisc`、`ncrisc` 或 `brisc` |
| `tt.kernel_thread` | `compute` 或 `datamovement` |
| `tt.noc_index` | DM 函数为 0 或 1；TRISC 不设置 |
| `tt.logical_kernel` | logical identity、kind 和 role |
| `tt.tensor_arg_indices` | 本函数使用的全局 Tensor 参数序号 |
| `tt.core_domain` | 本函数生效的 launch node domain |

三个函数不通过名称推断 slot；名称只用于可读性和 symbol identity。

函数 ABI 规则：

- Tensor 参数按全局 ABI index 的稳定升序出现；
- 只向函数传递它实际使用的 Tensor；
- DFB 和 Pipe 不是函数参数，由函数入口的逻辑 ID 绑定；
- 编译期 scalar 物化为常量；
- runtime scalar 必须有明确 TTL ABI 支持，否则 Lower 报 unsupported；
- 函数返回 void，输出通过 Tensor/DFB side effect 表达。

### 6.7 函数体不变量

Device PrimFunc body 必须满足：

- 已移除统一前端 operation wrapper；
- 只包含当前 slot 所属的 side-effect statement；
- 必要的纯控制和纯索引表达式可以复制到多个 slot；
- 不存在跨 PrimFunc SSA/Var 引用；
- 跨 slot 数据依赖只通过 logical DFB、Tensor 或 Pipe 表达；
- DFB acquire scope 显式标记为 producer reserve 或 consumer wait；
- copy 已带 transfer kind、source/destination region 和 owner；
- elementwise region 已归一化为 codegen 可识别的单输出 tile expression；
- loop、predicate 和 topology region 保持结构化；
- idle slot 使用规范 no-op body；
- 所有可诊断节点保留 source span。

## 7. Lower Pass Pipeline

### 7.1 Pipeline 顺序

当前顺序以 `tilelang/tenstorrent/pipeline.py` 中的
`TENSTORRENT_LOWER_PASS_ORDER` 和 `TenstorrentPassPipelineBody` 为准。以下 12 个
Pass 已接入；`BindTarget` 复用 TIRX Pass，其余属于 Tenstorrent backend。
本章区分完整 Device Lower 与独立的 structured compute 捕获。经典 Add/no-op 保留
Device IR v1，Phase 4 通用计算使用 v2；准确能力和验证范围见第十七章及
[tilelang_tenstorrent_phase4_lower_contract.md](tilelang_tenstorrent_phase4_lower_contract.md)。
多 Core 与 TTL Codegen 仍是后续设计。

```text
Frontend TIRX：T.Tiles / T.Parallel，以及独立的 copy、launch、topology
    -> 01 BindTarget(tenstorrent)
    -> 02 CanonicalizeTTElementwise
    -> 03 VerifyTTComputeBlocks
    -> 04 ValidateTenstorrentFrontendIR
    -> 05 NormalizeTenstorrentLaunch
    -> 06 NormalizeTenstorrentBufferMetadata
    -> 07 NormalizeTenstorrentRegions
    -> 08 NormalizeTenstorrentTopology
    -> 09 LegalizeTenstorrentTileOps（全部消费；不支持时明确诊断）
    -> 10 InferTenstorrentTensorLayout
    -> 11 FormTenstorrentDeviceProgram
    -> 12 VerifyTenstorrentDeviceIR
    -> Device TIR IRModule（tt.device_ir_version）
```

Pass 02 在一次遍历中将两种前端表达汇合到同一 structured compute 契约，Pass 03
立即检查该契约。Pass 04 检查外围前端结构，Pass 05 至 08 完成规范化。
Pass 09 负责已支持计算的设备指令选择；完整 Lower 必须继续运行 Pass 10 至 12。

Pass 02 必须紧跟 `BindTarget`：Target 决定普通 Parallel 的 backend 所有权；此时
原始 loop binder、Tiles annotation 和 allocation metadata 仍完整。它必须早于
buffer metadata 规范化，以保留 Tiles 要求显式 metadata、Parallel 可使用默认值
的区别，也必须早于任何 loop 标量化、block 消除和 SIMT layout 推导。
`VerifyTTComputeBlocks` 放在 capture 后，使后续 Pass 读取经过验证的表达式模板。

各 Pass 的任务速览如下：

| Pass | 主要任务 |
| --- | --- |
| `01 BindTarget` | 绑定 Tenstorrent Target。 |
| `02 CanonicalizeTTElementwise` | 统一捕获 Tiles 和受支持的 rank ≥ 2 Parallel，生成 `tl.tt.elementwise` SBlock；不选择 Add 指令。 |
| `03 VerifyTTComputeBlocks` | 验证表达式模板、完整读写 effects、access maps、tile geometry 和 broadcast recipes。 |
| `04 ValidateTenstorrentFrontendIR` | 验证外围 Frontend TIRX、launch、Buffer metadata、前端原语和控制流。 |
| `05 NormalizeTenstorrentLaunch` | 规范 launch grid 和逻辑 Core 坐标。 |
| `06 NormalizeTenstorrentBufferMetadata` | 建立稳定 Buffer ID 与类型化 allocation metadata。 |
| `07 NormalizeTenstorrentRegions` | 规范 copy region 和 transfer kind，保留 compute 的逻辑访问关系。 |
| `08 NormalizeTenstorrentTopology` | 规范 topology 输入；当前能力外的形式明确诊断。 |
| `09 LegalizeTenstorrentTileOps` | 消费已验证的 structured compute 和支持的独立 TileOp；超出支持范围明确诊断。 |
| `10 InferTenstorrentTensorLayout` | 生成 Tensor/DFB layout 描述，供 Device program formation 使用。 |
| `11 FormTenstorrentDeviceProgram` | 分析 dataflow/DFB transaction，生成三个 processor slot 的 Device TIR。 |
| `12 VerifyTenstorrentDeviceIR` | 验证 Device TIR schema、ABI、资源和 slot protocol。 |

两个前端共享 capture 分析器、compute verifier、设备 consumer 和后续 Device
pipeline。独立兄弟 scope 可混用 Tiles 与 Parallel；同一循环链必须使用一致的
前端形式。`CanonicalizeTTTiles` 与 `CanonicalizeTTParallel` 的独立 Python/FFI
factory 已删除，pipeline 只调用一次 `CanonicalizeTTElementwise`。

完整 backend pipeline 不再返回 `tt.ir_stage="structured"`。需要研究或测试 capture
契约时，直接调用 `CanonicalizeTTElementwise` / `VerifyTTComputeBlocks`；这类结果
不能标记为 Device Lower 完成，也不能交给 TTL Codegen。非法 IR、不支持能力与错误
数据流在所属 Pass 报错。TTL 生成与硬件执行仍待实现。

### 7.2 所有 Pass 的共同契约

每个 Pass 都必须满足：

- 输入前置条件和输出后置条件可由 verifier 检查；
- malformed IR 在改写前诊断，unsupported capability 与 malformed IR 分开报告；
- 不依赖 Buffer 名字、Python 对象 identity、visitor 顺序或 PassContext side channel；
- 保留 Buffer identity、Pipe record 顺序和 source span；
- 非局部决策先在不可变 IR 上完成分析和 plan，再执行原子改写；
- 失败不留下部分改写；
- 对已经规范化的同版本输入幂等，或明确拒绝错误阶段的输入；
- 每个新增 IR 形式同时更新下游 Pass、verifier 和 Codegen；
- 不读取 TTNN live object，不创建 TTL/TTKernel operation。

统一的 Pass 测试至少包含：

```text
positive transformation
already-canonical input
malformed input diagnostic
unsupported input diagnostic
metadata and span preservation
deterministic output
downstream verifier acceptance
```

### 7.3 Pass 01：`BindTarget(tenstorrent)`

| 项目 | 契约 |
| --- | --- |
| 粒度 | 现有 `tirx.transform.BindTarget` Module Pass |
| 输入 | 未绑定 Target，或已绑定等价 Tenstorrent Target 的 Frontend TIRX |
| 读取 | BackendContext 中规范化后的 Target |
| 写入 | 每个待编译 `PrimFunc` 的 `target` attribute |
| 输出 | 计算语义不变、Target 明确的 Frontend TIRX |
| 失败 | 已存在冲突 Target、缺少必需 `arch`、不支持的 Target option |

本 Pass 只建立 Target 所有权，不解释 launch，不改写 TileOp，也不创建 Device
PrimFunc。重复绑定同一 Target 不改变 IR；尝试覆盖不同 Target 必须失败。

测试覆盖 Target 属性、重复执行、冲突 Target 和无 TT-Lang 安装时可运行。

### 7.4 Pass 02：`CanonicalizeTTElementwise`

| 项目 | 契约 |
| --- | --- |
| 粒度 | PrimFunc Pass |
| 输入 | 已绑定 Tenstorrent Target、原始 allocation metadata 与 loop binder 完整的 Frontend TIRX |
| 读取 | Tiles scope annotation 或 Parallel loop nest、单一 BufferStore 的表达式树、Buffer identity/shape/dtype 和 allocation annotation |
| 写入 | opaque `SBlockRealize` / `SBlock`，名称为 `tl.tt.elementwise`，以及 compute annotations、reads/writes 和表达式模板 |
| 输出 | 两种前端共享的 structured compute IR，`tl.tt.tiles_stage=1`；外围串行循环和独立 scope 保留 |
| 失败 | 非 Tenstorrent Target、非法 domain/access/metadata、混合或嵌套 scope、多个 store、谓词、动态形状或表达式内副作用 |

统一 Canonicalizer 在外层 loop 以前序方式识别完整循环链：优先识别显式
`tl.tt.tiles_scope`，否则处理 `ForKind::kParallel`。先分析并验证整个 scope，再一次
替换，避免先改写内层 loop 导致外层 domain 丢失。已经形成的 compute block 原样
保留；重复执行幂等，分析失败不部分修改输入。

两种前端的共同支持范围为静态、正值、rank ≥ 2 且最后两维能被 `(32, 32)` 整除的 domain、单一
输出 store，以及 compatible shared / shared.dyn allocation。Tiles 要求显式
`tt.tile_shape` 和 `tt.dfb_block_count`；普通 Parallel 可以使用 backend 默认值。
表达式支持算术、cast、编译期标量和受支持的纯 unary call，可包含复合表达式以及
原地逐元素更新；capture 不局限于 Add，也不表示这些表达式均已有设备指令实现。

输出 block 保存以下语义：

- `tl.tt.compute_kind`、`tl.tt.logical_domain`、`tl.tt.physical_tile_shape`；
- `tl.tt.block_shape` tile grid，例如 `(2, 64, 64)` domain 对应 `(2, 2, 2)`，
  batch 前缀不作 32 分块，不折叠为单一 tile count；
- parallel iterator 类型、`tl.tt.tiles_stage=1`；
- 按 Buffer 去重的完整逻辑 `reads` / `writes` BufferRegion；
- 按 `buffer.data` identity 索引、与原 loop binder 无关的 access maps；
- row / column / scalar broadcast recipes，不创建展开后的 broadcast allocation；
- 保留运算树、将 load/store 坐标归零的表达式模板；该 body 不是可直接执行的标量代码。

allocation metadata 仍归 allocation block 所有，不能混入 compute annotation
替代其归属。合法访问包括 `[i, j]`、`[0, j]`、`[i, 0]` 和 `[0, 0]`，broadcast
输入必须有匹配的物理 tile allocation。transpose、offset、gather/scatter、RHS
使用坐标值、global store 和未支持的 alias/view 均明确拒绝。copy、GEMM、Reduction
和通信是独立 operation，不能藏入 elementwise scope。

测试覆盖两种前端的等价效果、完整 region、tile grid、广播、复合表达式、原地更新、
混合兄弟 scope 的两种顺序、幂等性、非法循环链及失败时输入不变。

### 7.5 Pass 03：`VerifyTTComputeBlocks`

| 项目 | 契约 |
| --- | --- |
| 粒度 | 只读 PrimFunc Pass |
| 输入 | 含 structured compute block 的 PrimFunc；无此类 block 时保持不变 |
| 读取 | compute annotations、allocation identity/metadata、expression template、effects、access maps 和 broadcast recipes |
| 写入 | 无 |
| 输出 | 与输入结构相同、满足 structured compute 契约的 PrimFunc |
| 失败 | annotation 缺失或类型错误、stage/geometry 不一致、region 或访问描述与表达式不匹配、非法 allocation 或副作用 |

验证器独立于设备指令能力，合法复合表达式或多 tile 计算不会仅因 Device consumer
尚未实现而验证失败。它检查表达式模板的纯度、归零坐标、唯一 store、完整 effects、
逻辑 domain 与物理 tile/grid 的一致性，以及每个 Buffer 的访问和广播描述。

pipeline 在 capture 后立即调用，Legalize 入口再次检查该契约。
测试分别篡改 annotations、effects、表达式与 allocation，验证诊断且输入不变。

### 7.6 Pass 04：`ValidateTenstorrentFrontendIR`

| 项目 | 契约 |
| --- | --- |
| 粒度 | 只读 Module Pass |
| 输入 | 已绑定 Tenstorrent Target，且 elementwise 已捕获并通过 compute verifier 的 Frontend TIRX |
| 读取 | PrimFunc、launch nest、Buffer、TileOp、loop、annotation、topology、span |
| 写入 | 无 |
| 输出 | 与输入对象结构相同的、满足前端结构契约的 IRModule |
| 失败 | 前端节点 malformed、引用悬空、Target 不匹配、明确不支持的 GPU-only 语义 |

本阶段入口验证检查外围前端结构是否可被后续 Pass 安全读取；compute block 的内部
契约由 Pass 03 验证。它不提前完成 layout、dataflow 或 kernel partition。至少验证：

- 每个 operation 有唯一可识别的 `T.Kernel` launch region；
- 参数与 `buffer_map` 完整，allocation Buffer identity 可追踪；
- `tt.*` metadata 类型正确，Pipe/PipeNet/selected PipeRef 引用闭合；
- copy、compute 和 topology 节点位于允许的结构化控制流中；
- 不存在 CUDA/ROCm/Metal intrinsic 或无法解释的 thread/warp 语义；
- 必需诊断节点具有 source span。

测试必须证明本 Pass 完全不改写 IR，并分别检查 malformed 与 unsupported 诊断。

### 7.7 Pass 05：`NormalizeTenstorrentLaunch`

| 项目 | 契约 |
| --- | --- |
| 粒度 | PrimFunc Pass |
| 输入 | 含前端 `blockIdx`/`threadIdx` launch nest 的已验证 PrimFunc |
| 读取 | launch loop、extent、thread binding、CoreRange 和 Core-coordinate use |
| 写入 | 规范化 launch metadata 和逻辑 Core X/Y 表达 |
| 输出 | 无 GPU SIMT 含义、具有唯一二维逻辑 Core grid 的 PrimFunc |
| 失败 | 动态或非正 grid、unsupported Z 维、thread extent 非 1、threadIdx 数据依赖 |

处理步骤：

1. 读取连续的 `blockIdx.x/y/z` 与 `threadIdx.x/y/z` launch nest；
2. 将二维 block domain 规范为 `(core_x, core_y)`；
3. 将 grid 写入类型化 `tt.launch_grid` metadata；
4. 将 body 中合法的 block index use 改写为逻辑 Core 查询；
5. 将 CoreRange 和 Pipe endpoint 统一到同一坐标系；
6. 删除不再需要的 GPU thread-binding 外壳；
7. 保留 launch 和坐标使用点的 source span。

本 Pass 不生成三个函数。多 Core 最终仍是三个函数覆盖整个 grid，通过 Core
coordinate 和 predicate 选择每个 launch node 的工作。

测试覆盖 1D/2D grid、非法 Z 维、动态 extent、threadIdx use、坐标映射和重复执行。

### 7.8 Pass 06：`NormalizeTenstorrentBufferMetadata`

| 项目 | 契约 |
| --- | --- |
| 粒度 | PrimFunc Pass |
| 输入 | launch 已规范化、Buffer identity 仍完整的 PrimFunc |
| 读取 | 参数 Buffer、SBlock allocation、scope、annotation、alias/view、span |
| 写入 | 稳定 Buffer ID 和类型化 `TTBufferMetadata` |
| 输出 | 每个 Tensor/DFB/fragment 候选都有唯一、可追踪 metadata 的 PrimFunc |
| 失败 | metadata 类型/范围错误、annotation 冲突、backing 引用非法、identity 丢失 |

本 Pass 完成：

- 按参数序号和 IR 顺序分配 deterministic Buffer ID；
- 分类 `tensor`、`logical_dfb_candidate`、`compute_fragment` 和 scalar/local；
- 规范 `tt.tile_shape`、`tt.dfb_block_count` 和 `tt.tensor_backed`；
- 区分 explicit、inferred 和 unset 状态；
- 记录 alias/view 与 source Buffer 的关系；
- 把 tensor backing 转成 Tensor ABI index 加 byte offset；
- 将 metadata 与原 Buffer source span 绑定。

本 Pass 不分配 `dfb_id`，不推导 producer/consumer，也不选择物理 CB。测试覆盖每种
scope、annotation、alias、backing、Buffer rewrite 后 identity 保留和确定性 ID。

### 7.9 Pass 07：`NormalizeTenstorrentRegions`

| 项目 | 契约 |
| --- | --- |
| 粒度 | PrimFunc Pass |
| 输入 | Buffer metadata 已规范化的 PrimFunc |
| 读取 | BufferLoad/Store、BufferRegion、slice、copy、PipeRef endpoint、loop domain |
| 写入 | canonical BufferRegion、transfer kind、typed Pipe endpoint |
| 输出 | 每个 transfer 均有显式 source/destination region 或 Pipe endpoint |
| 失败 | rank/extent 不匹配、越界、动态不可证明 region、非法 PipeRef 方向或逃逸 |

本 Pass 完成负索引、unit extent、rank 对齐和 copy region 规范化，并把 copy 分类为：

```text
Tensor -> DFB
DFB -> Tensor
DFB -> DFB
DFB -> Pipe source
Pipe destination -> DFB
```

Region 此时仍使用 element coordinate；Device 分支的 tile coordinate 换算所需
信息由 Pass 10 的 layout 推导产生。compute 的逻辑 domain、effects 和 access maps
已由 Pass 02 保存。禁止把 copy 展开为 DMA、逐元素 loop 或 target instruction。

测试覆盖完整/局部 region、slice、越界、alias view、Pipe send/receive 和 span 保留。

### 7.10 Pass 08：`NormalizeTenstorrentTopology`

| 项目 | 契约 |
| --- | --- |
| 粒度 | PrimFunc Pass |
| 输入 | 含已完成前端 topology primitive 的规范化 PrimFunc |
| 读取 | CoreRange、Pipe、PipeNet、foreach、predicate、selected PipeRef、payload region |
| 写入 | 保序 typed Pipe record 和结构化 topology region |
| 输出 | 可静态求值 Core domain、event order 和 endpoint contract 的 PrimFunc |
| 失败 | endpoint 越界、空 PipeNet、混合 contract、PipeRef 逃逸、payload 不兼容 |

本 Pass 必须保留 Pipe 构造顺序和重复事件，为每条 record 分配 deterministic
`event_index`，并规范 point-to-point/collective destination range、source/destination
foreach region 和 `is_src/is_dst/is_active` predicate。

本 Pass 不生成 semaphore、receive address、transport group 或物理 NoC protocol。
Phase 6 的最终 Device verifier 已将静态 event correspondence 和 wait-for cycle
列为本轮验收项；未来生成 TTL 后仍须独立运行 TT-Lang verifier，当前并未执行。
Phase 6 将本 Pass 前移至 BufferMetadata 之后、Regions 之前，以便先静态化
Core-dependent region；实际输入输出与顺序见第十七章及 Phase 6 契约。

测试覆盖 P2P、broadcast、gather、scatter、重复 Pipe、zero/one/many match、非法作用域
和 deterministic record order。

### 7.11 Pass 09：`LegalizeTenstorrentTileOps`

| 项目 | 契约 |
| --- | --- |
| 粒度 | 对外 pipeline Pass；内部先验证 structured compute，再做 PrimFunc 改写 |
| 输入 | 已完成 launch、Buffer、region 和 topology 规范化的 structured compute IR 与独立 TileOp |
| 读取 | 已验证的 `tl.tt.elementwise` block、表达式模板、access maps、dtype、tile geometry 和 Buffer metadata |
| 写入 | 兼容的 `tl.tt.tile_add` 或通用 `tl.tt.tile_compute` 中间 operation |
| 输出 | compute block 已消费、语义属性完整、可继续 layout 和 Device formation 的 IR |
| 失败 | malformed structured contract、未捕获的 elementwise loop、未知 TileOp、unsupported dtype/shape/axis/alias |

两个前端都由 Pass 02 建立 compute 契约，随后共享设备 consumer。通用 expression DAG
保留算术/比较/cast 等节点、节点 dtype 和每个输入的 access map；不是把原始 SBlock
简单包装进新的 op。Fill、Transpose、GEMM 和 Reduction 由各自 TileOp 的显式语义
字段消费，完整支持边界见第十七章。经典 Phase 2 Add 保留兼容 operation。

本 Pass 不生成 `ttl.*`，不拆分 processor slot，不为表达式生成标量逐元素循环。
DFB ID、generation、producer/consumer 和 Tensor ABI 由后续 Form Pass 统一生成。
超出已支持范围的计算明确失败；不再保留 block 并由 pipeline 返回 structured 成功。
独立 capture/verify Pass 仍可检查比完整 Device Lower 更宽的前端表达式集合。

### 7.12 Pass 10：`InferTenstorrentTensorLayout`

| 项目 | 契约 |
| --- | --- |
| 粒度 | PrimFunc Pass |
| 输入 | 已完成 Pass 09 指令选择、无待消费 compute block，且 Buffer metadata 与访问 region 已规范化的 PrimFunc |
| 读取 | element shape、dtype、tile constraint、memory scope、sharding、access region、Target |
| 写入 | 逻辑 Tensor/Tile/DFB layout metadata |
| 输出 | Device program formation 和后续 Codegen 可读取的 layout 描述 |
| 失败 | 不支持 dtype/layout/sharding、tile 不整除且无 padding、约束冲突；物理 L1 容量检查仍延期 |

本 Pass 位于 topology 规范化与 tile-op legalization 之后。运算选择所需的 dtype、
tile shape 与 access maps 已由 capture 和 metadata 规范化提供，不依赖本 Pass；
未消费的模板不能进入本阶段。rank ≥ 2 保留 untiled batch 前缀，末两维按 32×32
划分。rank-1 Reduction 结果的物理表示须遵守 Phase 4 专项契约，不能隐式推广为
任意 padding 支持。

layout 的长期设计明确区分以下信息；sharding、padding 等能力仍须按当前实现边界诊断：

```text
element shape
physical tile shape
tile-grid shape
DFB block shape in tiles
memory space
memory layout
shard mapping
padding/mask contract
```

它是 Tenstorrent 专用的逻辑 layout 推导，不复用带 GPU lane/warp 假设的 CUDA
`LayoutInference`。它不分配 DST、物理 L1 address 或物理 CB index。

测试按 dtype、rank、tile shape、DRAM/L1、interleaved/sharded、padding 和 target arch
组成参数矩阵。

### 7.13 Pass 11：`FormTenstorrentDeviceProgram`

| 项目 | 契约 |
| --- | --- |
| 粒度 | Module Pass |
| 输入 | Pass 02 至 10 已完成、无待消费 compute block，并满足 Device ABI/dataflow 的规范化 IR |
| 读取 | Module 中全部 operation、Buffer identity、effect、region、layout、topology、span |
| 写入 | 新 Device TIR Module、三个 slot PrimFunc、Module/Function/resource metadata |
| 输出 | 完整但尚未包含 TTL operation 的 Device TIR `IRModule` |
| 失败 | owner 歧义、跨 slot scalar、DFB 非 SPSC、slot 超限、transaction 无法闭合 |

这是唯一改变程序并发结构的 Pass。本轮通用实现由 `GeneralDataflowPlanner` 在
不可变输入上按程序顺序构造 DFB generation 和两个 slot 的 operation 序列，再统一
物化 module。静态控制流先展开/选择，输出 snapshot 保持单 consumer slot；未引入
新的独立 dataflow Pass。Phase 5 在同一 Form 内扩展静态窗口流水、storage group、
copy completion 与 release；能力及边界见第十七章。以下通用 collector、细分 plan、
Pipe 和物理 L1 规划仍是后续架构设计，不代表已经实现。

长期设计先在不可变输入上运行
`FrontendIRCollector`，建立 parent、Buffer、region、def-use、effect、loop、topology 和
span 索引。Collector 只提供事实，不决定 placement，也不是独立 IR。

随后构造短生命周期的不可变 plan：

```text
ProgramPlan
TensorPlan
DFBPlan
ComputeRegionPlan
TransferPlan
PipeNetPlan
LogicalKernelPlan
SlotAssignmentPlan
TransactionPlan
```

规划顺序为：

```text
collect facts
    -> analyze Tensor/DFB/compute/Pipe dataflow
    -> form logical DFB and transaction
    -> partition logical kernels
    -> assign TRISC/NCRISC/BRISC slots
    -> validate the complete plan
    -> atomically materialize Device TIR
```

数据流边至少包含：

```text
Tensor -> DFB
DFB -> Compute
Compute -> DFB
DFB -> Tensor
DFB -> Pipe
Pipe -> DFB
```

Logical DFB 规划负责 deterministic `dfb_id`、block shape、block count、
producer/consumer、acquire mode、transaction/loop relation 和早期 L1 可行性检查。
物理 index 复用和权威 CB budget 仍由 TT-Lang 完成。

Kernel partition 使用以下初始归属：

| Anchor | logical kind/affinity |
| --- | --- |
| Tensor -> DFB、DFB -> Tensor | data movement |
| elementwise/GEMM/reduce/store | compute |
| Pipe source send | data movement，Pipe-source affinity |
| Pipe destination receive | 普通 data movement |
| DFB acquire/release | 对应 block transaction 的 owner |
| pure index/predicate | 证明安全后复制到使用它的 kernel |

当前 slot capacity 为一个 compute slot 和两个 data-movement slot：

```text
compute       -> TRISC
ordinary DM   -> NCRISC
pipe-source DM or second compatible DM -> BRISC
```

普通 Tensor read 和 write 可以位于同一个 NCRISC。无工作 slot 生成 idle PrimFunc。
超过容量且无法合法合并时失败，不能固定假设 NCRISC 只读或 BRISC 只写。

所有 plan 验证成功后，`DeviceIRRewriter` 一次创建新 Module，并写入：

- `tt.device_ir_version`、Target、launch grid 和 operation identity；
- Tensor、DFB 和 Pipe descriptor；
- TRISC、NCRISC、BRISC 三个 `PrimFunc`；
- `global_symbol`、`calling_conv=DEVICE_KERNEL_LAUNCH` 和 `target`；
- slot、thread kind、`noc_index`、logical kernel、Tensor ABI 和 Core domain；
- acquire/transaction marker、结构化函数体和 source span。

本 Pass 不输出独立 `TTProgramGraph`，plan 不进入 Codegen。若输入已经是同版本且通过
验证的 Device TIR，本 Pass 返回不变；存在 version 标记但结构不完整时必须失败。

测试覆盖 Add 的 TRISC/NCRISC/idle BRISC、第二 DM slot、slot overflow、跨 slot value、
DFB SPSC、transaction、Pipe affinity、原子失败、确定性和重复执行。

### 7.14 Pass 12：`VerifyTenstorrentDeviceIR`

| 项目 | 契约 |
| --- | --- |
| 粒度 | 只读 Module Pass |
| 输入 | `FormTenstorrentDeviceProgram` 产生或 cache 反序列化的 Device TIR |
| 读取 | 全部 Module/Function/resource metadata 和三个函数体 |
| 写入 | 无 |
| 输出 | 与输入结构相同、可安全进入 TTL Codegen 的 Device TIR |
| 失败 | 任一 schema、引用、slot、ABI、dataflow、transaction 或 capability 不变量失败 |

本 Pass 执行第 9 章列出的 Module、Function、operation 和全程序检查。它至少在以下
位置运行：

- Lower pipeline 末尾；
- host/device Filter 后的 device module；
- TTL Codegen 入口；
- Device TIR cache/serialization 重载后。

任何 verifier 错误都必须带 Pass 名称、PrimFunc symbol、logical resource ID 和原始
source span。测试使用系统化 invalid matrix，并证明失败输入未被修改。

### 7.15 实现文件与注册

| Pass | 实现文件 |
| --- | --- |
| `BindTarget` | 复用 `tirx.transform.BindTarget` |
| `CanonicalizeTTElementwise` | `src/tenstorrent/transform/canonicalize_tiles.cc` |
| `VerifyTTComputeBlocks` | `src/tenstorrent/transform/canonicalize_tiles.cc` |
| `ValidateTenstorrentFrontendIR` | `src/tenstorrent/transform/validate_frontend_ir.cc` |
| `NormalizeTenstorrentLaunch` | `src/tenstorrent/transform/normalize_launch.cc` |
| `NormalizeTenstorrentBufferMetadata` | `src/tenstorrent/transform/normalize_buffer_metadata.cc` |
| `NormalizeTenstorrentRegions` | `src/tenstorrent/transform/normalize_regions.cc` |
| `NormalizeTenstorrentTopology` | `src/tenstorrent/transform/normalize_topology.cc` |
| `LegalizeTenstorrentTileOps` | `src/tenstorrent/transform/legalize_tile_ops.cc` |
| `InferTenstorrentTensorLayout` | `src/tenstorrent/transform/infer_tensor_layout.cc` |
| `FormTenstorrentDeviceProgram` | `src/tenstorrent/transform/form_device_program.cc` |
| `VerifyTenstorrentDeviceIR` | `src/tenstorrent/transform/verify_device_ir.cc` |

C++ Pass 按 TileLang/TVM 现有模式使用 `CreatePrimFuncPass` 或 `CreateModulePass` 并注册
FFI factory。`tilelang/tenstorrent/transform/__init__.py` 只暴露 factory wrapper，
`tilelang/tenstorrent/pipeline.py` 是 Pass 顺序的唯一权威位置。

统一 factory 为 `tl.tenstorrent.transform.CanonicalizeTTElementwise`，Python
入口为 `tilelang.tenstorrent.transform.CanonicalizeTTElementwise()`，通用
`tilelang.transform` facade 同步转发。两个旧 factory 不再注册；保留原
`canonicalize_tiles.cc` 文件名不代表仍有 Tiles 专用 Pass。
compute annotation 常量集中在 `src/tenstorrent/transform/attr.h`。

共享 `src/transform/lower_opaque_block.cc` 已补充 allocation metadata 按 data
identity 迁移，但 `LowerOpaqueBlock` 未加入该 pipeline。通用 Simplify、block
消除、buffer flattening、SIMT layout 和 scalar loop lowering 都不能处理尚未消费
的表达式模板。

合并回归以 `testing/python/target/test_tilelang_tenstorrent_tiles_transform.py`
为主，结合 phase0/phase1/phase2 和 backend 测试。实际 PassInstrument 验证两条
输出分支中 capture 只执行一次，并检查第 7.1 节的顺序；CPU 模板求值验证逻辑索引
与表达式语义。代码提交 `cd278886` 的相关回归结果为 287 passed、5 skipped（GPU），
native 编译链接与 pre-commit 检查通过；未验证 Tenstorrent 硬件、BF16 设备舍入、CB 同步和 TTNN
执行。上述结果不代表规划中的全部 Device/TTL 能力已完成。

## 8. 共享 Pass 兼容性与禁止边界

本章不是第 7 章主流水线的补充 Pass 列表，只规定 Tenstorrent-owned Pass 与现有
TileLang 通用 Pass 的兼容边界。基线实现只包含第 7.1 节列出的 12 个 Pass
（Device 分支完整执行）；任何共享 Pass 一旦加入，必须同时在第 7.1 节标明精确
位置并补齐前后置条件。

### 8.1 可复用但必须审计的通用 Pass

| 通用 Pass | 可能的插入位置 | 加入前必须证明 |
| --- | --- | --- |
| `LegalizeNegativeIndex` | `NormalizeTenstorrentRegions` 前 | Region 等价，Buffer identity 和 span 不变 |
| `IfStmtBinding` | Region/TileOp normalization 前 | predicate 与 topology region 结构不改变 |
| `UnrollLoop` | topology normalization 前 | 只展开显式 unroll，Pipe event multiplicity 正确保留 |
| `Simplify` | 局部 normalization 后 | 不删除 target op、typed metadata、空但有语义的 region |
| `CanonicalizeLegacyReducer` | `LegalizeTenstorrentTileOps` 前 | Reduction 语义、dtype 和 source identity 保持 |
| `VerifyReducerEpoch` | reducer canonicalization 后 | 只读验证，不引入 GPU schedule 假设 |

每个候选 Pass 的兼容性审计必须包含：

- 对全部受支持 TIR 节点执行 before/after structural diff；
- 验证 Buffer ID、DFB metadata、Pipe record 顺序和 source span；
- 验证重复 Pipe、空 region 和 side-effect operation 不被错误 CSE/DCE；
- 验证 Pass 对 Tenstorrent target-specific op 采用保守行为；
- 运行 Lower 正向、负向、幂等和 deterministic 测试；
- 在审计记录中冻结允许的 Pass option。

审计完成前，Tenstorrent-owned Pass 应自行实现所需的窄规范化逻辑，不应直接复用
CUDA pipeline 的 Pass 组合。

### 8.2 Device TIR 形成前禁止执行

```text
CUDA/ROCm/Metal target-specific pass
GPU-specific LayoutInference
ProducerConsumerWarpSpecialized
LowerBlackwell2SM
LowerTileOp
FlattenBuffer
StorageRewrite
MergeSharedMemoryAllocations
VectorizeLoop
ThreadSync
LowerDeviceKernelLaunch
```

这些 Pass 会删除、标量化或改写 Tile、BufferRegion、拓扑、Buffer identity 或 source
span，使 `FormTenstorrentDeviceProgram` 只能猜测原始语义。即使其中某个 Pass 将来
增加 Tenstorrent 支持，也必须先作为第 7 章显式 Pass 重新定义输入和输出契约。

### 8.3 DeviceCodegen 前的共享准备阶段

当前 `tilelang/engine/lower.py` 在调用 backend `DeviceCodegen` 前统一执行：

```text
LowerIntrin
Simplify
HoistBroadcastValues
```

这些 Pass 位于 `VerifyTenstorrentDeviceIR` 之后，因此对已经冻结的 Device TIR 不应
被默认视为安全。实施时二选一：

1. 为 `DeviceCodegen` 增加 backend-specific `prepare` hook；Tenstorrent 使用 identity
   prepare 加 `VerifyTenstorrentDeviceIR`；
2. 对三项 pass 完成完整保留性证明和测试，并在其后重新运行 verifier。

优先采用方案 1，避免已冻结的 Device TIR 在 Codegen 入口前发生未受控变化。

## 9. Device TIR Verifier

### 9.1 Module 与 Function 检查

- schema version、target arch 和二维 launch grid 合法；
- Tensor、DFB、Pipe、Kernel ID 唯一且引用闭合；
- v1/v2/v3 operation 内恰好有 TRISC、NCRISC、BRISC 三个 slot；v4 每个 Core
  各有三个 singleton-domain slot，`tt.kernel_order` 使用完整 global function name；
- slot、thread kind 和 `noc_index` 一致；
- `calling_conv=DEVICE_KERNEL_LAUNCH`；
- Tensor 参数与 `tt.tensor_arg_indices` 一一对应；
- 不存在跨函数 Var/Buffer 引用；
- idle body 只包含允许的 no-op；
- source span 和 logical kernel identity 存在；
- 不含 TTNN/Python 对象、TTL/TTKernel/EmitC op 或 GPU intrinsic。

### 9.2 Operation 与全程序检查

- 每个 op 在 capability registry 中有唯一 TTL lowering；
- copy direction、endpoint、BufferRegion、rank 和 tile 换算合法；
- compute region 输入输出、domain 和 dtype 明确；
- DFB acquire mode 与 producer/consumer role 一致；
- selected PipeRef 不逃逸；
- 每个 side effect 恰有一个 slot owner；
- logical DFB 在每个 launch node 上满足 SPSC；
- transaction acquire/use/release 关系闭合；
- PipeNet record 顺序和 multiplicity 未改变；
- 所有跨 slot 依赖通过 Tensor、DFB 或 Pipe 表达；
- Device TIR 输出确定，重复 Lower 不改变结构或 ID。

## 10. Codegen 总体设计

### 10.1 与 `TTLGenericCompiler` 的关系

| 工作 | `TTLGenericCompiler` | Tenstorrent Codegen |
| --- | --- | --- |
| 函数输入 | 单 slot Python AST | 单 slot Device PrimFunc |
| 符号表 | Python name -> SSA | TIR Var/Buffer/resource ID -> SSA |
| 类型 | TTNN capture/DFB object -> MLIR type | Device descriptor -> MLIR type |
| 控制流 | Python `if/for` -> SCF | TIR `IfThenElse/For` -> SCF |
| 计算 | Python operator/call -> `ttl.*` | TileOp/expression -> `ttl.*` |
| 输出 | 一个 `func.func` | 一个 `func.func` |

Codegen 不解析 Python AST、不执行 closure、不绑定 live TTNN Tensor、不调用
`split_function_body`，也不根据 statement 内容决定 slot。产品代码不依赖 TT-Lang
测试 helper 或私有 AST frontend。

### 10.2 组件拆分

```text
TTLModuleCodegen
  |- TTLContextFactory
  |- TTLTypeConverter
  |- TTLAttributeConverter
  |- TTLResourceEmitter
  |- TTLFunctionEmitter
  |    |- TTLStmtEmitter
  |    `- TTLExprEmitter
  |- TTLOpLoweringRegistry
  |- TTLLocationMapper
  `- TTLModuleVerifier
```

`TTLModuleCodegen` 只负责 orchestration。类型、资源、语句、表达式和 location 分别
测试，避免形成一个同时做分析和 emission 的大 visitor。

### 10.3 每次编译的 Context

```text
Device IRModule
target arch
compiler options affecting initial TTL
MLIR Context and Module insertion point
TIR Var -> MLIR Value map
Buffer identity -> MLIR Value map
dfb_id -> ttl.bind_cb Value map
pipe event ID -> ttl.create_pipe Value map
source span -> MLIR Location mapper
```

不得使用跨编译全局 DFB counter、随机 operation identity 或依赖进程顺序的 cache key。

## 11. TTL Module 组装

### 11.1 Context 和 Module

使用 TT-Lang 生产 binding：

```text
ttl.ir
ttl.dialects.ttl
ttl.dialects.ttcore
ttl.dialects.ttkernel
ttl.passmanager
ttl.passes
```

TT-Lang 保持 optional/lazy import。缺少依赖时，import TileLang 仍应成功；只有选择
Tenstorrent TTL Codegen 时才报告可操作错误。

从 Device TIR 生成 `ttl.launch_grid` 和 `ttl.target_arch`。函数按 TRISC、NCRISC、
BRISC 稳定顺序插入；该顺序不表示运行时串行执行。

### 11.2 Function attributes

每个 `func.func` 至少设置：

```text
ttl.kernel_thread
ttl.logical_kernel
ttl.crta_indices
ttl.base_cta_index
ttl.noc_index        # 仅 data movement
```

映射规则：

```text
TRISC   -> #ttkernel.thread<compute>
NCRISC  -> #ttkernel.thread<noc>, ttl.noc_index = 0
BRISC   -> #ttkernel.thread<noc>, ttl.noc_index = 1
```

`ttl.crta_indices` 来自 `tt.tensor_arg_indices`。`ttl.base_cta_index` 使用 initial TTL
契约要求的 provisional 值，后续 DFB finalization 可以更新它。

### 11.3 Tensor、DFB 与 Pipe

Tensor descriptor 转换为：

```text
element shape
    -> tile-grid device shape
    -> !ttcore.tile<h x w, dtype>
    -> ranked tensor type
    -> TTL/TTCore layout encoding
```

必须使用 Device TIR 中明确的 dtype、memory space、layout、sharding、tile 和 padding
contract；无法表示时应报错，不能静默采用默认 `32 x 32` 或 L1 layout。

每个函数只为其引用的 DFB 生成 `ttl.bind_cb`，携带 block shape、tile type、
block count、provisional index、module-wide `dfb_id` 和可选 tensor backing。同一
`dfb_id` 在不同函数中必须具有等价类型。

Pipe 按 descriptor sequence 生成 `ttl.create_pipe`，保留坐标、contract、PipeNet ID、
event order 和 source location。查询 map 不能改变 emission 顺序。

## 12. TIR 到 TTL 的映射

### 12.1 资源与生命周期

| Device TIR 语义 | Initial TTL MLIR |
| --- | --- |
| logical DFB declaration | `ttl.bind_cb` |
| producer acquire | `ttl.cb_reserve` + `ttl.attach_cb` |
| consumer acquire | `ttl.cb_wait` + `ttl.attach_cb` |
| explicit producer release | `ttl.cb_push` |
| explicit consumer release | `ttl.cb_pop` |
| tensor-backed DFB | `ttl.bind_cb` 的 `tensor_backing` attribute |

若 release 未在 Device TIR 固定，initial TTL 允许暂缺 push/pop，由
`ttl-insert-cb-sync` 完成。Codegen 必须保证 acquire 和 use 足以让该 pass 证明正确
release 点。

### 12.2 Tensor slice 与 copy

| Device TIR transfer | Initial TTL MLIR |
| --- | --- |
| Tensor region -> DFB block | `ttl.tensor_slice` + `ttl.copy` |
| DFB block -> Tensor region | `ttl.tensor_slice` + `ttl.copy` |
| fixed completion dependency | `ttl.wait` |
| deferred completion dependency | 保留 handle，交给 `ttl-insert-copy-wait` |

Element coordinate 到 tile coordinate 的换算必须使用 Device TIR 的 tile/layout
descriptor，不允许由 Codegen 猜测默认 tile shape。

### 12.3 Pipe/PipeNet

| Device TIR 语义 | Initial TTL MLIR |
| --- | --- |
| `foreach_src` region | `ttl.pipenet_foreach_src` |
| `foreach_dst` region | `ttl.pipenet_foreach_dst` |
| source selection | `ttl.select_pipe_src` |
| destination selection | `ttl.select_pipe_dst` |
| receive setup | `ttl.pipe_transfer.create` + `ttl.pipe_transfer.post` |
| source send | `ttl.pipe_transfer.send` |
| receive completion | `ttl.pipe_transfer.wait` |
| `is_src/is_dst/is_active` | 对应 TTL predicate op |

Codegen 只进行结构转换。send/post/wait 数量、条件对应和 wait-for cycle 由 TT-Lang
verifier 再次验证。

### 12.4 Elementwise

Device TIR canonical expression DAG 按 capability registry 映射到 tensor-level TTL op：

```text
Add/Sub/Mul/Div
comparison
exp/log/sqrt/rsqrt/tanh
abs/neg/relu/sigmoid
其他 TTLElementwiseOps.def 中已注册 operation
```

Codegen 使用明确的 dtype 和 result type 构造 SSA。不能把 BufferStore 逐元素循环直接
翻译成大量 TTL scalar op；Lower 必须先证明并保留 tile expression region。

### 12.5 结构化计算

| Device TIR | Initial TTL MLIR |
| --- | --- |
| GEMM TileOp | `ttl.matmul` |
| Reduction TileOp | `ttl.reduce` |
| Transpose TileOp | `ttl.transpose` |
| Fill TileOp | `ttl.fill` |
| Broadcast | `ttl.block.broadcast` |
| Typecast | `ttl.typecast` |
| 写入 reserved output block | `ttl.store` |

必须传递 transpose、axis、accumulate、acc dtype、broadcast dimension、tile shape 和
source location。FPU/SFPU 选择、tile-level expansion 和 DST 分配不在 Codegen 中完成。

### 12.6 控制流与坐标

| TIRX | MLIR |
| --- | --- |
| static/structured `For` | `scf.for` |
| `IfThenElse` | `scf.if` |
| integer/float constant | `arith.constant` |
| index arithmetic | `arith.*` / `index.*` |
| logical Core X/Y | `ttl.core_x` / `ttl.core_y` |

第一版拒绝：

- 无界或无法结构化的控制流；
- 跨 slot loop-carried SSA；
- 任意指针算术；
- GPU thread/lane/warp intrinsic；
- 无法证明 Pipe event dynamic count 的控制流。

### 12.7 Source location

每个生成 op 使用原 TIRX `Span` 创建 `Location.file`。合成 operation 使用所属源
statement 的 location，必要时使用 fused/callsite location。

Codegen 错误至少报告：

```text
stage = TTL Codegen
Device PrimFunc symbol and slot
TIR node kind
logical DFB/Pipe/kernel ID
source file:line:column
unsupported mapping or violated invariant
```

## 13. Codegen Verifier 与输出

### 13.1 分层验证

Codegen 执行：

```text
VerifyTenstorrentDeviceIR
    -> emit one func.func
    -> verify temporary function/module
    -> assemble all functions
    -> verify complete TTL Module
    -> deterministic print
```

完整 Module verifier 必须在任何 TT-Lang lowering pass 之前运行。

### 13.2 Initial TTL 输出边界

Initial TTL 必须已经包含：

- Tensor types/layout；
- `ttl.launch_grid` 和 `ttl.target_arch`；
- 三个 kernel slot 函数及属性；
- logical DFB `dfb_id`；
- `ttl.bind_cb`、reserve/wait/attach；
- TensorSlice、copy 和 transfer handle；
- tensor-level compute/store；
- Pipe/PipeNet 结构；
- source location。

Initial TTL 可以尚未包含：

- 所有 copy `ttl.wait`；
- 所有 `ttl.cb_push`/`ttl.cb_pop`；
- 最终 physical CB index；
- DST index；
- `ttl.compute`、tile op 和 subblock schedule；
- TTKernel、EmitC 或 C++。

### 13.3 输出承载

语义输出是一个 TTL MLIR `Module`。为满足 TileLang `DeviceCodegen` 返回 runtime
Module 的接口，需要一个明确 format 的 source module：

```text
format = "ttl"
source = deterministic TTL MLIR text
metadata = target arch, schema version, digest, source map summary
```

不能把 TTL 伪装成 C/CUDA source。若现有 source module 不能保存 format 和 metadata，
在 `src/tenstorrent/runtime/` 增加后端本地 ModuleNode。

## 14. TT-Lang 交接边界

Codegen 完成后，下一阶段的最小契约为：

```text
TTL MLIR Module/text
    -> parse
    -> verify
    -> ttl-to-ttkernel-pipeline
    -> TTKernel / optional EmitC
    -> C++ translation
```

当前 TT-Lang pipeline 负责：

- materialize loop state；
- insert copy wait；
- create producer compute/intermediate DFB；
- convert tensor-level TTL to compute；
- insert CB synchronization；
- verify PipeNet；
- form Pipe transport；
- finalize DFB index；
- configure compute、assign DST、subblock 和 scheduling；
- validate DFB SPSC 和 CB budget；
- convert TTL to TTKernel；
- optional EmitC。

生产路径应使用 TT-Lang 生产模块或新增窄公共 API。`ttlang-opt` 作为 parser、verifier、
pass 和 golden 测试工具，不是 TileLang 正常 Codegen 必须启动的外部进程。

## 15. Capability Registry

Lower 和 Codegen 共享一个只读 capability registry，但分别查询不同状态：

```text
Frontend form supported by Lower?
Device TIR form has unique TTL lowering?
TTL dialect/pipeline supports target arch?
```

建议 key：

```text
operation
input/output dtype
accumulation dtype
rank and shape class
tile shape
memory layout and sharding
single/multi Core
control-flow placement
target arch
```

状态为 `supported`、`legalizable`、`deferred` 或 `unsupported`。只有 Lower golden、
TTL parser/verifier 和 TT-Lang compile-only 全部通过后，能力才可标为
compiler-supported；硬件支持状态另行记录。

## 16. 建议代码布局

以下为待新增位置：

```text
tilelang/tenstorrent/
  __init__.py
  backend.py
  pipeline.py
  capabilities.py
  device_ir.py
  analysis/
    collector.py
    launch.py
    buffers.py
    dataflow.py
    compute.py
    topology.py
    partition.py
    transactions.py
  transform/
    __init__.py
  codegen/
    module.py
    types.py
    attributes.py
    resources.py
    function.py
    stmt.py
    expr.py
    registry.py
    locations.py

src/tenstorrent/
  CMakeLists.txt
  ir/
    device_metadata.h
    device_metadata.cc
  transform/
    validate_frontend_ir.cc
    normalize_launch.cc
    normalize_buffer_metadata.cc
    normalize_regions.cc
    infer_tensor_layout.cc
    legalize_tile_ops.cc
    normalize_topology.cc
    form_device_program.cc
    verify_device_ir.cc
  runtime/
    ttl_source_module.cc

testing/python/tenstorrent/
  lower/
    test_frontend_validation.py
    test_launch.py
    test_buffers.py
    test_regions.py
    test_layout.py
    test_tile_ops.py
    test_compute.py
    test_partition.py
    test_transactions.py
    test_topology.py
    test_device_ir_verifier.py
  codegen/
    test_types.py
    test_resources.py
    test_control_flow.py
    test_compute.py
    test_pipenet.py
    test_ttl_module.py
    test_ttl_determinism.py
```

分工建议：

- Python `pipeline.py` 只组合 pass；
- C++ 实现高性能 TIR traversal、typed metadata 和原子 Module rewrite；
- Python Codegen 使用 TT-Lang MLIR bindings 构造 TTL；
- `backend.py` 注册 `PassPipeline` 与 `DeviceCodegen`；
- TT-Lang、TTNN 保持 optional/lazy import；
- shared `tilelang/engine/lower.py` 不增加 Tenstorrent 字符串分支。

## 17. 实施阶段

### Phase 0：冻结接口

工作：

1. 合入或 checkout 已完成的前端原语分支；
2. 打印单 Core Add、P2P、collective/gather 的真实 Frontend TIRX；
3. 冻结 frontend op/attr、Buffer metadata 和 topology region contract；
4. 冻结 Device TIR schema v1；
5. 冻结 initial TTL Module contract；
6. 确认 TT-Lang production binding 和版本；
7. 决定 `DeviceCodegen.prepare` hook；
8. 冻结第 7.1 节的 12-Pass 顺序、独立 capture 与完整 Device Lower 的边界和共同契约。

退出条件：三类输入 TIRX 均可打印并 round-trip；direct TTL MLIR 是唯一产品路线；
Device TIR 和 TTL 属性名无歧义；不依赖 generated TT-Lang Python。

### Phase 1：Device TIR 骨架

工作：

1. 实现 typed Module/Function/DFB/Pipe metadata；
2. 实现 `ValidateTenstorrentFrontendIR`；
3. 实现 `NormalizeTenstorrentLaunch`；
4. 实现 `NormalizeTenstorrentBufferMetadata`；
5. 实现 `NormalizeTenstorrentRegions`；
6. 实现 `InferTenstorrentTensorLayout` 的基础 tiled layout；
7. 实现 frontend collector 和 `FormTenstorrentDeviceProgram` 骨架；
8. 实现 `VerifyTenstorrentDeviceIR` 基础；
9. 接入 Backend `PassPipeline` 并输出 `01_device.tir`。

退出条件：单 Core、无 Pipe 输入生成三个 slot；idle slot 可表示；metadata/span 可
round-trip；无 TT-Lang 安装时 Lower 测试可独立运行。

### Phase 2：Add 的完整 Lower

工作：

1. 完成 `LegalizeTenstorrentTileOps` 的 BF16/FP32 Add 路径；
2. 在 `FormTenstorrentDeviceProgram` 中分析 Global -> shared、shared compute、
   shared -> Global 数据流；
3. 规划三个 logical DFB；
4. 形成 compute 和普通 DM logical region；
5. 将 compute 放入 TRISC；
6. 将输入和输出 copy 合并到 NCRISC；
7. 生成 idle BRISC；
8. 插入 acquire/transaction marker；
9. 增加 Device TIR golden 和负向测试。

退出形态：

```text
TRISC  = reserve output + wait inputs + add/store
NCRISC = reserve/read inputs + wait/write output
BRISC  = idle
```

Add 是第一条纵向验收，不是通用架构上限。

### Phase 3：Add 的 TTL Codegen

工作：

1. 初始化 MLIR Context/dialect；
2. 实现 BF16/FP32 tiled Tensor type；
3. 实现 DFB type 与 `ttl.bind_cb`；
4. 实现三个函数和 Module attributes；
5. 实现 reserve/wait/attach、TensorSlice、copy、add、store；
6. 保留 source location；
7. 输出并 verify initial TTL；
8. 使用 TT-Lang Add initial TTL 做结构化 differential test。

退出条件：TTL parser/verifier 通过；normalized TTL 与 TT-Lang unified Add 结构等价；
`ttl-to-ttkernel-pipeline` compile-only 通过；输出不包含 TTKernel、EmitC 或 C++。

### Phase 4：通用单 Core 计算

Phase 4 的原始目标覆盖 Lower、Device verifier、TTL Codegen、TTL verifier 和
compile-only。本轮只推进 Lower 和 Device verifier；其余阶段单独延期，不宣称整个
Phase 4 或可执行 backend 完成。详细契约见
[tilelang_tenstorrent_phase4_lower_contract.md](tilelang_tenstorrent_phase4_lower_contract.md)。

#### 本轮 Lower 实现矩阵

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

#### Device IR 交付契约

- Legalize 输出 `tl.tt.tile_compute`；Form 消费后生成 `tl.tt.dfb_compute`，表达式输入
  叶节点为 `tl.tt.dfb_load`，不残留 frontend Buffer/Var 或 compute SBlock。
- 通用 schema v2 的 transfer 为 rank-aware region，DFB 每次写入建立新 generation。
  Tensor effect、dtype、tile-grid、backing 和 processor ABI 均从数据流推导；从 Tensor 输出逆追
  消除不影响可见输出的纯计算，保留未使用 Tensor 参数的 ABI。
- 导出 Tensor 时创建单独 snapshot DFB，保持每个 generation 的 consumer slot 唯一；
  不以重复使用同一个可覆盖资源模拟 in-place。
- 完整 pipeline 返回通过 Device verifier 的 IR；不再返回
  `tt.ir_stage="structured"` 作为成功结果。capture-only 检查继续通过独立 Pass 提供。

#### 验证与延期

本轮在 native rebuild 后运行 compute 单元测试、通用 Device 集成测试和原有 Add
回归，并检查 Tiles/Parallel 等价结果、负向条件、DFB 依赖和 metadata 一致性。
新增测试位于 `testing/python/target/test_tilelang_tenstorrent_phase4_compute.py`、
`test_tilelang_tenstorrent_phase4_device.py`、`test_tilelang_tenstorrent_phase4_control.py`
和 `test_tilelang_tenstorrent_phase4_semantics.py`；最后一组用独立 NumPy 模型执行最终 Device IR，覆盖 dtype 舍入、GEMM/Reduction
语义和序列化，不重跑前端分析，也不是 TTL 或硬件数值认证。
最终修改对象重编译并链接后，全部 Tenstorrent target 测试、backend import 和语言 Tiles
回归为 **373 passed，29 warnings**；warnings 是现有 Python typing 弃用提示。
最终 Device IR 数值语义组 60 项全部通过，结构化循环保留及反例也已通过。
复现命令、operation 映射和诊断边界同时存于纳入本次交付的
`docs/compiler_internals/tenstorrent_phase4_lower.md`；本目录本身被仓库 Git ignore。

尚未实现：动态结构化控制流、batched GEMM、任意 alias/view、更多 dtype/accumulation、
任意 transpose/reducer、partial compute region 和一般 padding/mask。
明确延期：所有 TTL 生成及 parser/verifier、TT-Lang compile-only/differential、模拟器和
硬件数值验证；Phase 5 的本轮 Lower 扩展见下一节，L1 物理规划仍延期；
Phase 6 多 Core/PipeNet 通信。

### Phase 5：流水与容量

原始目标覆盖 `T.Pipelined` transaction relation、block_count 推导、多个在途 copy、
保守与延迟 wait、nested acquire、DFB capacity/L1 早期检查，以及 TT-Lang 物理分配
差分验证。本轮仅推进 Lower、Device IR 和 Device verifier；Phase 3 继续跳过，
不把 TTL Codegen、compile-only 或硬件能力计为完成。详细设计见
[tilelang_tenstorrent_phase5_lower_contract.md](tilelang_tenstorrent_phase5_lower_contract.md)。

#### 实现范围与延期矩阵

以下状态依据修改 C++ 的 native rebuild、最终 Device IR 测试及数值模型验收；
“已完成”仅指本轮限定的 Lower/Device verifier，不代表完整可执行 backend。

| 条目 | 状态 | 本轮 Lower 范围 | 边界/延期 |
| --- | --- | --- | --- |
| `T.Pipelined` 消费 | 受限支持 | 静态独立 whole-body loop，按 D=min(stages, extent) 窗口实际展开并重排输入 copy | 不是通用 modulo scheduling；动态/嵌套 pipeline、内部条件和手工调度不支持 |
| stage/iteration/transaction | 已完成 | v3 独立 generation ID 与 `[ordinal, ordinal % D]` relation；每 generation descriptor transaction=1 | 不复用 v2 保留 For 的 extent relation；逻辑 stage 不等于 physical CB index |
| 多个在途 copy | 已完成 | 默认 delayed 策略每窗口先发起多笔输入 copy，显式 completion 后 publication；输出 copy 完成后才 release | 不声明 NoC 性能重叠或设备执行验证 |
| 保守/延迟 wait | 已完成 | `tt.pipeline_wait_policy` 选择逐 issue 完成或窗口批量完成；显式 consumer wait 与最后使用 release | 不实现自动最优 wait placement 或 TT-Lang sync pass |
| DFB `block_count` | 受限支持 | 同一迭代展开后写入位置的 storage group 默认 D 个 block；显式足够容量保留，按保守窗口预算检查 | 不做跨 write-site 物理 coloring/地址分配 |
| nested acquire | 已完成 | 多个持有 block 在生命周期图中同时计数，最后使用后逐一归还 | 不推广成任意多 owner 或跨 Core acquire |
| 容量/覆盖检查 | 已完成 | slot 顺序、publication→wait、旧 occupant release→reuse reserve 统一 DAG；不足容量/环依赖失败 | 安全结论以未来 Codegen 实现同一 backpressure 契约为前提 |
| L1 早期检查 | 受限支持 | 唯一 storage group 的静态 tiled payload×capacity；可选显式预算 | 未包括 allocator/runtime/code/scratch/fragmentation；小于预算不是物理分配可行性证明 |
| schema 兼容性 | 已完成 | Phase 2 Add/no-op 保留 v1，Phase 4 保留 v2，流水使用 v3；typed DFBDescriptor 签名不变 | v3 消费者必须检查新 metadata/operation，不能回退为 v1 Add |
| TT-Lang 物理分配差分 | 延期验证 | 本轮不执行 | Phase 3/TTL Codegen 尚未实现，本轮不运行 compile-only、Simulator 或硬件 |

#### 退出条件与验证记录

不同 stage 数必须生成确定、可序列化且通过 verifier 的最终 Device IR。同步与
release 必须闭合，容量不足可定位，资源 reuse 必须依赖旧值完成消费；不能仅保留
流水 annotation 或返回 `tt.ir_stage="structured"`。数值验证执行最终 Device IR 的
独立 NumPy 模型，并保留 Phase 2 Add 和 Phase 4 全部既有 Lower 回归。

最终单次完整回归 **485 passed、29 warnings，3.40 秒**：Phase 5 pipeline 28 项、
resources 37 项、semantics 47 项与原有 373 项均通过。包含 S=32、N=1024 上界，
以及 12 个正例的 Device script/JSON 字节确定性断言；warning 是现有 Python typing
弃用提示。

代表性 kernel 是 `hyf-test/phase5_lower_demo/kernel.py`。默认生成
`hyf-test/phase5_lower_demo/generated/device_ir.tir` 和 JSON、三个 slot、
`verification.json`、`observation.md`：3 stages、7 轮、起点 2，28 个 immutable
DFB generation 共用 4 个容量为 3 的 pool，静态 payload 为 **49152 bytes**。
模型输入 copy 在途峰值为 6，逐轮 7 次输出均与 NumPy 相等，误差为 0；重复 Lower、
Device script/JSON 字节确定性、JSON round-trip 后 Device verifier、恰好 L1 payload
预算均通过。`generated-conservative/` 另含 blackhole、4 stages、2 轮、起点 5 的
conservative 调度：8 generation、4 pool、32768 字节 payload、copy 在途峰值 1，
两次输出误差为 0。两种 arch 仅经过 Lower 验证，均未在硬件运行。

复现时先 import tilelang 以初始化本仓库 TVM 依赖路径：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-phase5-cache .venv/bin/python -c 'import glob, tilelang, pytest; raise SystemExit(pytest.main(sorted(glob.glob("testing/python/target/test_tilelang_tenstorrent_*.py")) + ["testing/python/backend/test_tilelang_tenstorrent_import.py", "testing/python/language/test_tilelang_language_tiles.py", "-q"]))'
TILELANG_CACHE_DIR=/tmp/tilelang-phase5-cache .venv/bin/python hyf-test/phase5_lower_demo/kernel.py
```

数值模型使用有限容量存储、异步 copy 延迟和不同 slot 调度，并对比 serial 与流水
输出；这是最终 Device IR 的逻辑数值验证，不是 TTL 或硬件认证。
尚未实现：通用 modulo scheduling、动态/嵌套流水、循环携带依赖、手工 order/stage、
跨 write-site 物理容量复用、真实 L1 地址/CB index 分配。
延期验证：TTL Codegen/parser/verifier、TT-Lang compile-only/物理分配差分、Simulator、
硬件正确性与性能。以上是 Phase 5 的交付范围；Phase 6 多 Core/PipeNet 的本轮
Lower 扩展见下一节。

### Phase 6：多 Core 与 PipeNet

本轮在已完成 Phase 5 基础上推进静态多 Core/PipeNet 的 Lower、Device IR 与
verifier；Phase 3 继续跳过。详细契约见
[tilelang_tenstorrent_phase6_lower_contract.md](tilelang_tenstorrent_phase6_lower_contract.md)。
“完成”仅用于实际通过测试的 Lower 能力，不代表 executable backend 或硬件认证。

#### 设计与实现边界

Core grid/domain 采用正静态二维 grid 和半开范围，按 Core 特化。P2P 是单 source
到单 destination；broadcast/multicast 是同一 payload 的多目标复制；gather 是
按 record 顺序的多笔独立接收，不隐含 reduce；true scatter 必须实际传递可区分的
payload，不能以同值多目标复制代替。二维 GEMM 在 Core 行分发 A、列分发 B，
source 本地使用 panel，单 Core 轴不产生远端通信。

foreach 零匹配为空体，一匹配执行一次，多匹配按原 record 顺序执行，重复 record
保留 multiplicity。Pipe source 固定 BRISC affinity，接收使用 NCRISC，计算使用
TRISC；每 Core 三个 slot 的序列化顺序不代表运行时串行顺序。每 record 本轮只有
一次发送，重复使用同一 record、动态 topology 及跨 Core `T.Pipelined` 明确拒绝。

跨 Core 使用 Device IR v4，保留 Add v1、Phase 4 v2、Phase 5 v3。v4 新增 typed
PipeTransferDescriptor，把原 record 显式关联到 source/destination Core 与独立
DFB generation。source/recv 的 issue 和 completion、consumer wait 与最后使用
release 组成可验证协议；每 `(Core, slot)` 的程序顺序和跨 Core 数据依赖统一检查。
每 generation 独立存储，正 capacity 与每 Core payload budget 可验证；本轮没有
把 Phase 5 pool/ring 复用推广到 PipeNet。

现有 NormalizeTopology 前移至 BufferMetadata 之后、Regions 之前，使 Core-dependent
slice 先常量化再验证 region；Form 消费静态 Core SBlock 并生成每 Core 三个最终
Device PrimFunc。没有新增平行的 pass pipeline，不允许以 annotations 或 structured
IR 代替完整 Lower 成功结果。

#### 能力和验证状态

以下状态依据最终 native rebuild 与实际回归；“已完成”仅指本轮限定的
Lower/Device IR/verifier 范围。

| 条目 | 状态 | 本轮范围与边界 |
| --- | --- | --- |
| Core grid/domain、P2P、broadcast/multicast | 已完成 | 正静态 grid ≤ 256 Core、完整 tiled payload；含 self-Pipe，逻辑 multicast 不承诺物理 NoC multicast |
| gather、true scatter | 已完成 | 数值模型验证 record 顺序、反序 source、可区分 payload、重复覆盖及显式 discard；不隐含 reduce/concat |
| 二维 GEMM 行列分发 | 受限支持 | 2×2 source 兼计算和 3×3 专用分发 Core 两种静态 GEMM 已验证；非动态/循环调度的通用 SUMMA |
| BRISC affinity、foreach zero/one/many、重复 record | 已完成 | 每 record 一次发送，重复 endpoint record 独立保留；active net 不允许 guard 静默丢 record |
| producer/consumer、transaction、同步闭合 | 已完成 | 最终 Device IR 独立检查，含缺失/重复 transaction、completion、release 与跨 Core 等待环 |
| DFB 容量、生命周期、每 Core payload 预算 | 受限支持 | 独立 generation、block_count 1–32、source snapshot 完成后释放；不支持跨 Core pipeline pool/ring |
| schema 确定性与序列化 | 已完成 | v4 typed descriptor，重复 script/JSON 字节相同、round-trip verifier；v1/v2/v3 回归保留 |
| 动态 topology、每 record 多 occurrence、跨 Core 流水、物理存储复用 | 尚未实现 | Lower 明确诊断，不因已有 frontend 构造能力计为完成 |
| TTL Codegen、TT-Lang compile-only/guard/schedule/provenance/SPSC | 延期验证 | 本轮不实现、不运行 |
| Simulator、硬件数值/性能/长时间无死锁 | 延期验证 | 本轮不运行；Lower 无环和 NumPy 逻辑模型不能替代 |

#### 退出条件

本轮退出条件是最终 Device IR 确定、可序列化并通过独立 verifier；合法 topology
与 zero/one/many、重复 record、错误 transaction、缺失同步、容量不足及覆盖/死锁
风险均有测试；独立模型验证支持范围内的逻辑数值语义；保留 Add、Phase 4、Phase 5
回归。在 `hyf-test` 下交付代表性 kernel、最终 Device IR、通信路径与资源说明。
原计划的 Codegen golden、TT-Lang verifier、Simulator 与硬件长时间测试继续作为
后续执行能力退出条件，不能在本轮标记完成。

#### 最终验收记录

修改的 native C++ 已重编译并链接后，最终完整回归 **580 passed、29 warnings，
3.94 秒**。其中 Phase 6 新增 **95 项**：Lower 24、resources 39、semantics 32；
原有 **485 项**全部保留，包括 Add、Phase 4、Phase 5、backend import 和语言 Tiles。
warnings 是现有 Python typing 弃用提示。Lower 正例覆盖两种 arch metadata、
256 Core 上界、零匹配/重复 record、自发送、collective、反序 gather、true scatter
和 2D GEMM；负例覆盖 257 Core、错误/重复 occurrence、guard 丢 record、缺 receiver、
错误 transaction、缺同步、提前 release、容量不足、输出覆盖和跨 Core 死锁环。
逻辑模型使用不同 slot 调度和延迟传输，包含 BF16 多 tile、FP32 通信和 GEMM，
不是 TTL、模拟器或硬件执行。

代表性 kernel `hyf-test/phase6_lower_demo/kernel.py` 已在 `generated/` 与
`generated-blackhole/` 分别生成 wormhole_b0/blackhole 最终 Device IR、JSON、
27 个 Core/slot 文件、`verification.json` 和 `observation.md`。3×3 逻辑 grid 上
有四个 worker 和四个分发 Core：4 条 collective record 展开为 8 笔 transfer，
24 个独立 DFB generation。每个 source 占 8192 bytes，每个 worker 占 16384 bytes；
每 Core 显式预算 16384 bytes 通过 verifier。三种逻辑调度 seed 0/1/19 的 GEMM
最大绝对误差均为 0，两种 arch 的重复 script/JSON 字节确定性与 JSON round-trip
verifier 均通过；这些 arch 名称只经过 Lower 验证。

新增 tracked 测试不依赖被 Git ignore 的 `hyf-test` 文件：3×3 kernel factory
保存在 Phase 6 Lower 测试模块，demo 复用该 factory。C++ 经完整 CMake 构建，最后
修改另经 CMake 生成的编译/链接命令重新构建；Ruff check/format 与 `git diff --check`
通过。native 旧 header warning 不影响构建；不把这些检查称为硬件验证。

复现命令（先 import tilelang 初始化 TVM 依赖路径）：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-phase6-cache .venv/bin/python -c 'import glob, tilelang, pytest; raise SystemExit(pytest.main(sorted(glob.glob("testing/python/target/test_tilelang_tenstorrent_*.py")) + ["testing/python/backend/test_tilelang_tenstorrent_import.py", "testing/python/language/test_tilelang_language_tiles.py", "-q"]))'
TILELANG_CACHE_DIR=/tmp/tilelang-phase6-cache .venv/bin/python hyf-test/phase6_lower_demo/kernel.py
TILELANG_CACHE_DIR=/tmp/tilelang-phase6-cache .venv/bin/python hyf-test/phase6_lower_demo/kernel.py --arch blackhole --output hyf-test/phase6_lower_demo/generated-blackhole
```

### Phase 7：生产化

完成 deterministic hash/cache、source map、阶段化诊断、schema/version compatibility、
optional dependency、wheel packaging、compile trace、fuzz/negative matrix、架构差异和性能
回归门禁。

## 18. 测试策略

### 18.1 Lower 测试，不依赖 TT-Lang

| 测试 | 验证 |
| --- | --- |
| Frontend TIR contract | 原语分支生成预期节点、metadata、span |
| normalization unit | shape、region、layout、topology 规范化 |
| analysis unit | def-use、effect、DFB、Core domain、partition |
| Device TIR golden | 三 slot、attrs、body、资源 descriptor |
| verifier negative | 所有 malformed/unsupported 输入明确失败 |
| determinism | 重复编译 structural-stable |
| idempotence | 重复 Lower 不变化或明确拒绝 |
| serialization | print/parse 后 Codegen 输入语义不变 |

### 18.2 Codegen 测试，需要 TT-Lang binding

| 测试 | 验证 |
| --- | --- |
| type unit | Tensor/Tile/DFB/layout 类型 |
| op unit | 每个 Device op 的唯一 TTL lowering |
| function golden | thread、noc、logical kernel、ABI attrs |
| Module golden | grid、arch、顺序、资源 identity |
| parser/verifier | initial TTL 结构合法 |
| frontend differential | 与 TT-Lang Python frontend 的 normalized initial TTL 等价 |
| compile-only | 完整 TTL pipeline 到 TTKernel/EmitC/C++ |
| diagnostics | TIR span 映射到 TTL error |
| optional dependency | 未安装 TT-Lang 时错误边界正确 |

Differential test 比较 function slot/arg type、logical DFB、acquire/copy/compute/store
dataflow、Pipe records 和 location presence，不比较易变 SSA 名称。

### 18.3 能力与负向矩阵

能力矩阵至少包含 operation、dtype/acc dtype、element/tile-grid shape、block_count、
DRAM/L1/sharding、single/multi Core、pipeline stage、control flow、target arch 和 alias。

负向测试必须覆盖：

- dynamic/invalid grid 和 threadIdx 数据依赖；
- unsupported dtype/layout/tile shape；
- 非整除 shape 且无 padding contract；
- Buffer metadata identity 丢失；
- copy region 越界、rank/extent 不匹配；
- 无法归一化的 scalar loop；
- 超过 compute/DM slot capacity；
- 跨 slot scalar SSA；
- DFB 非 SPSC、容量不足；
- Pipe endpoint 越界、PipeRef 逃逸或方向错误；
- Pipe event 次数/条件不匹配；
- unknown TileOp/target op 和 GPU-only intrinsic；
- Device TIR schema 或 TT-Lang version 不兼容。

## 19. 编译产物与调试

两个阶段必须可独立导出：

```text
00_frontend.tir
01_device.tir
02_initial_ttl.mlir
```

可附加纯数据 manifest，记录 operation、arch、grid、Tensor ABI、三个 kernel symbol、
logical DFB、PipeNet、source map、版本和 digest。Manifest 用于诊断和缓存，不是
Codegen 的权威输入。

错误按阶段分类：

```text
Frontend contract error
Lower normalization error
Lower planning/capacity error
Device TIR verifier error
TTL Codegen mapping error
TTL parser/verifier error
TT-Lang pipeline error
```

## 20. 主要风险与控制措施

| 风险 | 控制措施 |
| --- | --- |
| Lower 前丢失高层语义 | 在 `LowerTileOp`、FlattenBuffer、StorageRewrite 前接管并测试 metadata/span |
| Device TIR 与 plan 双重权威 | plan 只存在于一次原子 pass；Codegen 只接受 Device TIR |
| 三 slot 退化为固定 reader/writer 模板 | 先 partition 语义 region，再映射 slot；允许 NCRISC 读写和 BRISC idle |
| Codegen 重新承担分析 | verifier 要求唯一 owner/mapping；缺失 metadata 直接失败 |
| 重复实现 TT-Lang | 物理 DFB、同步、Pipe transport、DST、TTKernel 归 TT-Lang |
| 前端分支契约漂移 | 从真实 TIRX dump 冻结 contract，原语和 Lower 测试同步更新 |
| pre-codegen pass 修改冻结 IR | backend-specific prepare hook 或逐 pass 重新验证 |
| TT-Lang API 不稳定 | 锁定 submodule，只依赖生产 binding，补窄公共 API |

## 21. 非目标

这两个阶段明确不做：

- 从任意低层标量 TIR 恢复 TileOp；
- 把 CUDA TMA/WGMMA/warp/barrier 机械翻译为 Tenstorrent；
- 生成 TT-Lang Python 文件并再次运行 AST frontend；
- 依赖 `exec`、Python closure 或 live TTNN Tensor；
- 在 Device TIR 固定 physical CB index、DST、semaphore 或 NoC 指令；
- 在 Codegen 中运行 TTNN、创建 device 或 launch Program；
- 把 Add 的 Buffer/函数名字硬编码成通用规则；
- 把调试 JSON/manifest 作为编译器内部交换 IR。

## 22. 完成标准

### 22.1 Lower 完成

- [ ] 已完成前端原语分支的真实 TIRX contract 已冻结；
- [ ] Frontend TIRX 可确定性生成 Device TIR `IRModule`；
- [ ] 每个 operation 有 TRISC、NCRISC、BRISC 三个 slot PrimFunc；
- [ ] idle slot、logical identity、thread kind 和 `noc_index` 正确；
- [ ] Tensor ABI、DFB、PipeNet、Core domain 和 source span 完整；
- [ ] 所有 side effect 有唯一 owner 且无跨函数隐式依赖；
- [ ] Device TIR verifier 覆盖正向和负向路径；
- [ ] 不依赖 TT-Lang/TTNN 即可运行 Lower 测试；
- [ ] Device TIR 可打印、round-trip 和确定性哈希。

### 22.2 Codegen 完成

- [ ] Codegen 只读取 Device TIR；
- [ ] 使用类型化 MLIR binding，不以字符串模板为主生成器；
- [ ] 三个 PrimFunc 稳定生成三个 TTL `func.func`；
- [ ] Tensor/Tile/DFB/Pipe/控制流和计算 mapping 完整；
- [ ] Module/Function attributes 满足 TT-Lang pipeline 契约；
- [ ] logical `dfb_id` 跨函数一致；
- [ ] source span 转为 MLIR Location；
- [ ] initial TTL Module parser/verifier 通过；
- [ ] 与 TT-Lang frontend 的代表性 initial TTL 结构等价；
- [ ] `ttl-to-ttkernel-pipeline` compile-only 通过；
- [ ] output 中不包含 TTKernel、EmitC、C++ 或 runtime object；
- [ ] 缺少 TT-Lang 时保持 TileLang import 可用并给出明确错误。

### 22.3 完整能力完成

- [ ] Add 只作为第一条验收，不限制架构；
- [ ] elementwise、GEMM、Reduction、Fill、Transpose、Typecast 已覆盖；
- [ ] multi-tile、pipeline 和 block_count 已覆盖；
- [ ] P2P、collective、gather、scatter 和 2D GEMM topology 已覆盖；
- [ ] Lower、Codegen、TTL verifier、compile-only 测试矩阵一致；
- [ ] Simulator 与硬件验证范围明确，不以静态测试代替运行正确性。

## 23. 关键源码索引

### TileLang

- Frontend 到 TIRX：[code/tilelang_dsl_to_initial_tirx.md](./tilelang_dsl_to_initial_tirx.md)；
- 前端原语契约：[code/tilelang_ttlang_frontend_primitives.md](./tilelang_ttlang_frontend_primitives.md)；
- Lower 入口：[tilelang-tenstorrent/tilelang/engine/lower.py:109](../tilelang-tenstorrent/tilelang/engine/lower.py#L109)；
- DeviceCodegen 前准备：[tilelang-tenstorrent/tilelang/engine/lower.py:92](../tilelang-tenstorrent/tilelang/engine/lower.py#L92)；
- Backend context：[tilelang-tenstorrent/tilelang/backend/module.py:188](../tilelang-tenstorrent/tilelang/backend/module.py#L188)；
- PassPipeline：[tilelang-tenstorrent/tilelang/backend/pass_pipeline/pipeline.py:11](../tilelang-tenstorrent/tilelang/backend/pass_pipeline/pipeline.py#L11)；
- DeviceCodegen：[tilelang-tenstorrent/tilelang/backend/device_codegen.py:31](../tilelang-tenstorrent/tilelang/backend/device_codegen.py#L31)；
- launch 物化：[tilelang-tenstorrent/src/transform/materialize_kernel_launch.cc:103](../tilelang-tenstorrent/src/transform/materialize_kernel_launch.cc#L103)；
- CUDA `LowerTileOp` 边界：[tilelang-tenstorrent/tilelang/cuda/pipeline.py:133](../tilelang-tenstorrent/tilelang/cuda/pipeline.py#L133)。

### TT-Lang

- backend slot：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/ttl_api.py:129](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/ttl_api.py#L129)；
- logical kernel split：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/atom_split.py:419](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/atom_split.py#L419)；
- slot AST 物化：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/atom.py:136](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/atom.py#L136)；
- `TTLGenericCompiler`：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py:145](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py#L145)；
- Tensor type：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py:83](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py#L83)；
- DFB binding：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py:917](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py#L917)；
- TTL function：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py:992](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/_src/ttl_ast.py#L992)；
- Module 组装：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/ttl_api.py:2110](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/ttl_api.py#L2110)；
- Python pipeline：[tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/ttl_api.py:2294](../tilelang-tenstorrent/3rdparty/tt-lang/python/ttl/ttl_api.py#L2294)；
- registered pipeline：[tilelang-tenstorrent/3rdparty/tt-lang/lib/Dialect/TTL/Pipelines/TTLPipelines.cpp:19](../tilelang-tenstorrent/3rdparty/tt-lang/lib/Dialect/TTL/Pipelines/TTLPipelines.cpp#L19)；
- TTL op：[tilelang-tenstorrent/3rdparty/tt-lang/include/ttlang/Dialect/TTL/IR/TTLOps.td:26](../tilelang-tenstorrent/3rdparty/tt-lang/include/ttlang/Dialect/TTL/IR/TTLOps.td#L26)；
- TTL pass：[tilelang-tenstorrent/3rdparty/tt-lang/include/ttlang/Dialect/TTL/Passes.td](../tilelang-tenstorrent/3rdparty/tt-lang/include/ttlang/Dialect/TTL/Passes.td)。

## 24. 最后总结

```text
Lower
  理解 Frontend TIRX 的程序语义
  建立逻辑资源和并发数据流
  完成 TRISC/NCRISC/BRISC 划分
  生成唯一、完整、可验证的 Device TIR IRModule

Codegen
  像 TTLGenericCompiler 一样构造类型、SSA、region 和 TTL operation
  但输入是三个 Device PrimFunc，不是 Python AST
  不再做 kernel partition 或资源推断
  输出一个通过 verifier 的 initial TTL MLIR Module

TT-Lang
  完成同步、物理资源、Pipe transport、DST、TTKernel 和 EmitC/C++ lowering
```

Lower 的完成标准是“Device TIR 已无语义歧义”；Codegen 的完成标准是“每个 Device
TIR 节点都有唯一、确定且可验证的 TTL MLIR 映射”。
