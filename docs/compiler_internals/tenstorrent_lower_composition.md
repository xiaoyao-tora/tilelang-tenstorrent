# Tenstorrent Lower 组合能力

本次补齐工作计划 `tilelang_tenstorrent_lower_codegen_work_plan_02.md` 中值版本、
materialization、控制流、流水和多 Core 组合的 Lower 缺口。成功边界是经过独立
verifier 检查的 Device IR；不新增 TTL Codegen、运行时或硬件能力。

## 已补齐的路径

| 路径 | Lower 行为 |
| --- | --- |
| Transpose / Reduction → fragment | 先生成精确 dtype 的 DFB 结果，再以 identity compute value 回到值图；保留几何、依赖及 release |
| 局部数据相关纯计算分支 | 两个分支各含一次同目标、同域纯赋值时转换为 Select；允许缺省 else，但要求旧目标已初始化 |
| 多 Core v7 | 每个 Core 独立的逻辑 Buffer 身份、值版本和 accumulator；通信经 materialized shared DFB |
| v7 独立流水 | 展开静态 iteration，保持各 epoch 的值依赖和 borrowed DFB 生命周期，按有限容量池复用 |
| 持久 accumulator 流水 | v5 单 Core / v6 SUMMA 保留循环前初始化、循环内 K 更新和循环后一次 materialize，不插入中途 dtype 舍入 |
| 流水 Tensor 切片 | 接受可证明全部迭代边界合法的静态大小切片及整数别名；在 formation 中按 iteration 代入 |
| PipeNet 与流水 | 保留每次通信 occurrence、Core 归属、端点 epoch 和 release/reuse 依赖 |

条件表达式可含比较、Boolean 组合和 Select；存储类型仍是 BF16/FP32。分支谓词
须读取已初始化局部 shared/fragment 的零索引，且符合现有 broadcast/access-map
契约。带 copy、Pipe 通信等副作用的运行时分支不会被推测执行。这里没有新增动态
Device If ABI，也不支持任意分支内多次写入的值合并。

## 组合协议与 verifier

不新增 Device IR 版本。旧程序继续使用原有版本；v7 保留 compute-value schema，
通过已有拓扑表和流水属性表达组合。流水 metadata 可用于 v3–v7，v1/v2 携带这些
属性会被拒绝。多 Core verifier 按实际 PrimFunc/Core 检查值的定义、使用及精度，
禁止跨 Core 直接引用值或借用 DFB。

`tt.dfb_storage_groups` 和 `tt.pipeline_relations` 仍以规范十进制 DFB ID 字符串
为 key。循环内每池包含 N 个 generation，各 relation 为 `[ordinal, ordinal % D]`，
`D = min(stages, extent)`。循环前后的一次性 DFB 使用独立池、relation `[-1, 0]`，
不把其容量乘以流水深度，也不允许伪装为有缺失 epoch 的循环资源。

verifier 独立检查池形状/dtype/归属、容量、copy completion、最后使用、release
及复用依赖图；预算按每个 Core 的唯一池统计。Pipe 源与目标须处在相同 epoch。
保存的旧值和 identity 链仍借用原 DFB，不能在最后的传递消费者读取前释放。
诊断包含当前 PrimFunc、operation/resource ID，并在可用时附带源位置。

SUMMA 的 NCRISC 若转发本地加载的 panel，会在首次 send 前完成该 panel 的
Tensor copy。其余输入保留窗口预取，避免接收端读取尚未发布的数据。

## 能力查询和保留边界

`tilelang.tenstorrent.capabilities.lower_capability` 返回 `supported`、`legalizable`、
`deferred` 或 `unsupported`，并附带原因和 Device IR 版本。查询描述实现范围，
不替代实际 shape、初始化、容量和数据流检查。原 `gemm_capability` API 保留，
Lower、TTL mapping、compile-only 与硬件验证状态分别记录。

以下仍不在本次 Lower 支持范围：一般动态循环/有副作用分支、跨流水 iteration
的任意 fragment 状态、v7 持久值 accumulator 的流水合成、动态 tail/mask、任意
strided/alias view、sharded Tensor、非 32×32 compute tile、batched GEMM，以及
没有明确 dtype 合同的隐式精度转换。静态 pipeline 的 extent 为 1–1024、stage 为
1–32；不实现通用 modulo scheduling、物理 CB/DST 分配或 NoC 性能优化。

## 验证

2026-09-17：原生重建成功，下面的完整 Lower 回归 **1013 passed、33 warnings**
（12.49 秒）；warning 来自现有 Python typing 注解弃用提示。变更文件通过完整
pre-commit hooks 和 `git diff --check`。本次没有运行 TTL compiler 或硬件测试。

新增回归分别覆盖分支数值/旧值与初始化、算子回到 fragment、多 Core 值隔离、
持久 full-K 精度、Pipe 与流水、切片和能力查询。Device 参考执行器使用有限池容量、
延迟 copy completion 和随机 slot 交错检查输出及死锁；这些验证不等同于设备执行。
对损坏 Device IR 的测试覆盖跨 Core 值/DFB、错误 epoch、资源池和生命周期，
并验证 JSON round-trip 后仍由独立 verifier 接受合法 IR。

先运行 `cmake --build build -j 4`，再执行 Lower 回归：

```sh
TILELANG_CACHE_DIR=/tmp/tilelang-lower-plan02-cache .venv/bin/python - <<'PY'
import tilelang
import pytest
from pathlib import Path
paths = sorted(str(p) for p in Path('testing/python/target').glob('test_tilelang_tenstorrent*.py')
               if 'ttl_codegen' not in p.name)
paths += ['testing/python/backend/test_tilelang_tenstorrent_import.py',
          'testing/python/language/test_tilelang_language_tiles.py']
raise SystemExit(pytest.main(paths + ['-q', '--tb=short']))
PY
```
