<img src="./images/logo-row.svg" alt="TileLang logo" />

<div align="center">

# Tile Language
[![PyPI version](https://badge.fury.io/py/tilelang.svg)](https://badge.fury.io/py/tilelang)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tile-ai/tilelang)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/TUrHyJnKPG)
[![Puzzles](https://img.shields.io/badge/🧩_Learn-TileLang_Puzzles-blueviolet)](https://github.com/tile-ai/tilelang-puzzles)
[![LSP](https://img.shields.io/badge/LSP-TileLang_LSP-orange?logo=visualstudiocode&logoColor=white)](https://github.com/tile-ai/tilelang-lsp)

[Documentation](https://tilelang.com/) ·
[Installation](https://tilelang.com/get_started/Installation.html) ·
[Examples](https://github.com/tile-ai/tilelang/tree/main/examples) ·
[Releases](https://github.com/tile-ai/tilelang/releases) ·
[Contributing](https://github.com/tile-ai/tilelang/blob/main/CONTRIBUTING.md)
</div>

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

<img src="./images/MatmulExample.png" alt="TileLang tiled matrix multiplication example" />

## Latest News

- **2026-08-04 — [TileLang LSP open sourced](https://github.com/tile-ai/tilelang-lsp):** published a Language Server Protocol implementation for TileLang with inlay hints for buffer shapes, dtypes, scopes, and inferred layouts, plus hover details and precise diagnostics.
- **2026-08-03 — [TileLang v0.1.13](https://github.com/tile-ai/tilelang/releases/tag/v0.1.13):** shipped the multi-backend language dialect, source locations in compiler diagnostics, new CUDA and Metal hardware paths, and a broad set of correctness fixes. This release removes several legacy APIs; read the compatibility notes before upgrading.
- **2026-07-30 — [SM120 NVF4 block-scaled MMA](https://github.com/tile-ai/tilelang/pull/2364):** added an optimized Blackwell path for `T.mma_gemm_blockscaled` and a corresponding SM120 example.
- **2026-07-28 — [Metal 4 cooperative-tensor GEMM](https://github.com/tile-ai/tilelang/pull/2252):** added cooperative-tensor `T.gemm` support for Apple M5, while retaining the simdgroup fallback for unsupported shapes and systems.

<details>
<summary><strong>Earlier news (2025–2026)</strong></summary>

### 2026

- **2026-07-28 — [Source-aware compiler diagnostics](https://github.com/tile-ai/tilelang/pull/2751):** carried Python source locations into TIRX and surfaced them in compiler errors.
- **2026-07-24 — [Multi-backend language dialect](https://github.com/tile-ai/tilelang/pull/2734):** reorganized the language layer around shared semantics with static CUDA, ROCm, and Metal dialects.
- **2026-07-23 — [Block-causal attention for dLLM](https://github.com/tile-ai/tilelang/pull/2499):** added fixed-length and variable-length block-causal attention examples for diffusion language models.
- **2026-07-22 — [IR Lower Trace](https://github.com/tile-ai/tilelang/pull/2725):** introduced a debugging tool for inspecting IR changes across every compiler pass and the final code-generation step.
- **2026-07-22 — [DeepSeek V3.2 sparse MLA backward](https://github.com/tile-ai/tilelang/pull/2592):** selected the launch width adaptively from the head-block size.
- **2026-07-21 — [DeepSeek V3.2 top-k optimization](https://github.com/tile-ai/tilelang/pull/2659):** improved the sparse-attention top-k selector's memory access pattern, delivering approximately 1.9× higher performance in the reported benchmark.
- **2026-07-16 — [Compiler pass timing](https://github.com/tile-ai/tilelang/pull/2622):** added profiling for compiler passes with a configurable reporting threshold.
- **2026-07-12 — [IKET profiler integration](https://github.com/tile-ai/tilelang/pull/2515):** added CUDA timeline instrumentation and profiling support.
- **2026-07-08 — [TileLang v0.1.12](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12):** added the LLVM backend, tile scheduler, backend registry, pass visualizer, and expanded Blackwell support.
- **2026-07-06 — [Pass Visualizer](https://github.com/tile-ai/tilelang/pull/2449):** introduced a structure-tree browser for inspecting compiler transformations.
- **2026-06-26 — [Cross-host CUDA binary cache](https://github.com/tile-ai/tilelang/pull/2459):** enabled compiled CUDA binaries to be reused across compatible hosts.
- **2026-06-24 — [Tile scheduler](https://github.com/tile-ai/tilelang/pull/2441):** introduced persistent tile-scheduling primitives for kernel authors.
- **2026-06-24 — [Backend CodeGen registry](https://github.com/tile-ai/tilelang/pull/2442):** moved device and [host CodeGen](https://github.com/tile-ai/tilelang/pull/2446) dispatch behind a backend registry.
- **2026-06-18 — [LLVM backend](https://github.com/tile-ai/tilelang/pull/2409):** added CPU lowering and execution through LLVM.
- **2026-06-18 — [Arbitrary-layout TMA lowering](https://github.com/tile-ai/tilelang/pull/2380):** enabled TMA transfers for swizzled shared-memory layouts.
- **2026-06-16 — [Pass Diff](https://github.com/tile-ai/tilelang/pull/2375):** added compiler-pass IR comparison for debugging lowering changes.
- **2026-06-08 — [TileLang v0.1.11](https://github.com/tile-ai/tilelang/releases/tag/v0.1.11):** expanded scan, pipeline, backend, CUDA, ROCm, and Metal functionality.
- **2026-05-25 — [Scan operators](https://github.com/tile-ai/tilelang/pull/2262):** introduced tile-level scan primitives.
- **2026-05-25 — [TileLang v0.1.10](https://github.com/tile-ai/tilelang/releases/tag/v0.1.10):** broadened AMD and Blackwell support, added initial Metal GEMM, improved Windows packaging, and expanded autotuning.
- **2026-05-24 — [CDNA4 MXFP4](https://github.com/tile-ai/tilelang/pull/2132):** added FP4 E2M1 matrix-core support for AMD gfx950.
- **2026-05-22 — [Metal simdgroup GEMM](https://github.com/tile-ai/tilelang/pull/1869):** added the first Metal `T.gemm` path using `simdgroup_matrix` MMA.
- **2026-05-20 — [Cluster copies](https://github.com/tile-ai/tilelang/pull/1908):** introduced `T.copy_cluster` for TMA multicast and SM-to-SM cluster transfers.
- **2026-05-20 — [TMA gather/scatter](https://github.com/tile-ai/tilelang/pull/2129):** added `tile::gather4` and `tile::scatter4` support.
- **2026-05-20 — [Native SM75 MMA GEMM](https://github.com/tile-ai/tilelang/pull/2198):** added FP16, INT8, and INT4 tensor-core paths for Turing GPUs.
- **2026-05-20 — [TIRX migration](https://github.com/tile-ai/tilelang/pull/2216):** moved TileLang IR usage to TVM's TIRX representation.
- **2026-05-11 — [Parallel autotuning](https://github.com/tile-ai/tilelang/pull/2159):** added pipelined compilation, grouped compilation, and multi-GPU benchmarking.
- **2026-05-07 — [DeepSeek V4 operators](https://github.com/tile-ai/tilelang/pull/2148):** added TileLang examples for DeepSeek V4 workloads.
- **2026-05-06 — [Windows support](https://github.com/tile-ai/tilelang/pull/2093):** added complete Windows build and runtime support with cross-platform fixes.
- **2026-04-28 — [MXFP8 grouped GEMM](https://github.com/tile-ai/tilelang/pull/2098):** added block-scaled grouped GEMM examples with transposed-B support on Blackwell.
- **2026-04-25 — [HISA sparse-attention indexer](https://github.com/tile-ai/tilelang/pull/2069):** added hierarchical sparse-attention indexing examples.
- **2026-04-24 — [Blackwell MXFP8 block-scaled GEMM](https://github.com/tile-ai/tilelang/pull/1945):** added MXFP8 block-scaled matrix multiplication on SM100.
- **2026-04-22 — [TileLang v0.1.9](https://github.com/tile-ai/tilelang/releases/tag/v0.1.9):** delivered CuTe DSL GEMM V2, Metal code generation improvements, and build-without-host-toolchain support.
- **2026-04-22 — [RDNA3/RDNA3.5 WMMA](https://github.com/tile-ai/tilelang/pull/2044):** added WMMA lowering for AMD gfx11 GPUs.
- **2026-04-20 — [INT4 `T.gemm`](https://github.com/tile-ai/tilelang/pull/2063):** added INT4 matrix multiplication to the CUDA GEMM path.
- **2026-04-17 — [CUDA source kernels](https://github.com/tile-ai/tilelang/pull/1970):** introduced `T.CUDASourceCodeKernel` for embedding custom CUDA source.
- **2026-04-15 — [AutoDD frozen regions](https://github.com/tile-ai/tilelang/pull/2045):** added `__freeze__` annotations to preserve selected code during automatic delta debugging.
- **2026-03-27 — [TMA stores](https://github.com/tile-ai/tilelang/pull/1981):** added store support to `T.tma_copy`.
- **2026-03-24 — [Two-SM Blackwell kernels](https://github.com/tile-ai/tilelang/pull/1882):** added two-SM TMA, TMEM, and TCGEN5 MMA support.
- **2026-03-23 — [AMD RDNA4](https://github.com/tile-ai/tilelang/pull/1951):** upgraded the ROCm path and added RDNA4 GPU support.
- **2026-03-22 — [FlashAttention on SM100](https://github.com/tile-ai/tilelang/pull/1910):** added Blackwell FlashAttention examples.
- **2026-03-18 — [Producer-consumer warp specialization](https://github.com/tile-ai/tilelang/pull/1909):** added automatic warp-specialized pipelines and the `T.tma_copy` API.
- **2026-03-12 — [Eager-mode autotuning](https://github.com/tile-ai/tilelang/pull/1906):** enabled the autotuner with eager JIT kernels.
- **2026-03-10 — [CPU `T.gemm`](https://github.com/tile-ai/tilelang/pull/1904):** added matrix multiplication support for the CPU target.
- **2026-03-05 — [IR dump configuration](https://github.com/tile-ai/tilelang/pull/1903):** added a TileLang pass configuration for dumping intermediate IR.
- **2026-02-28 — [CUDA cluster primitives](https://github.com/tile-ai/tilelang/pull/1874):** added cluster launch, query, synchronization, and barrier operations.
- **2026-02-28 — [TCGEN5 MMA tensor-shared path](https://github.com/tile-ai/tilelang/pull/1866):** added the tensor-memory/shared-memory Blackwell GEMM path.
- **2026-02-24 — [Host-toolchain-free builds](https://github.com/tile-ai/tilelang/pull/1833):** enabled installation without a host C/C++ toolchain when supported artifacts are available.
- **2026-02-23 — [CuTe DSL GEMM V2](https://github.com/tile-ai/tilelang/pull/1855):** added SM90 and SM100 GEMM V2 support to the CuTe DSL backend.
- **2026-02-16 — [TileLang v0.1.8](https://github.com/tile-ai/tilelang/releases/tag/v0.1.8):** shipped dynamic pipeline improvements, logging documentation, richer layout representations, and AMD fixes.
- **2026-02-14 — [Cross-CUDA release wheels](https://github.com/tile-ai/tilelang/pull/1826):** unified multiple CUDA versions behind a single wheel.
- **2026-02-14 — [Hierarchical reductions](https://github.com/tile-ai/tilelang/pull/1762):** added hierarchical and warp-level reduction intrinsics.
- **2026-02-09 — [CUDA runtime stubs](https://github.com/tile-ai/tilelang/pull/1821):** added lazy-loading CUDART and NVRTC stubs for CUDA 11, 12, and 13 compatible wheels.
- **2026-02-08 — [Layout visualization improvements](https://github.com/tile-ai/tilelang/pull/1811):** improved rendering and inspection of TileLang layouts.
- **2026-02-02 — [TileLang Puzzles](https://github.com/tile-ai/tilelang-puzzles):** published ten progressively harder exercises for learning TileLang interactively.

### 2025

- **2025-12-18 — [CuTe DSL backend](https://github.com/tile-ai/tilelang/pull/1421):** added compilation through NVIDIA CUTLASS CuTe DSL; follow ongoing work in [issue #1454](https://github.com/tile-ai/tilelang/issues/1454).
- **2025-12-17 — [Z3 integration](https://github.com/tile-ai/tilelang/pull/1367):** integrated SMT-based symbolic reasoning into the TVM arithmetic analyzer.
- **2025-10-31 — [Apache TVM FFI migration](https://github.com/tile-ai/tilelang/pull/1108):** moved the runtime interface to `apache-tvm-ffi` to reduce host-side overhead.
- **2025-10-30 — [TileLang v0.1.6.post2](https://github.com/tile-ai/tilelang/releases/tag/v0.1.6.post2):** published the final TileLang release compatible with Python 3.8.
- **2025-10-07 — [Apple Metal backend](https://github.com/tile-ai/tilelang/pull/799):** introduced Metal device support for Apple silicon.
- **2025-09-29 — [Huawei Ascend adapters](https://github.com/tile-ai/tilelang-ascend):** published AscendC and Ascend NPU IR backend work in the external TileLang Ascend project.
- **2025-07-04 — [2:4 sparse tensor cores](https://github.com/tile-ai/tilelang/pull/526):** introduced `T.gemm_sp` for structured-sparse matrix multiplication.
- **2025-06-05 — [NVRTC execution backend](https://github.com/tile-ai/tilelang/pull/461):** added an NVRTC path to reduce compilation time for generated CUDA templates.
- **2025-04-14 — [FlashMLA on AMD MI300X](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla/amd):** published the optimized AMD implementation and accompanying documentation.
- **2025-03-03 — [MLA decoding on H100](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla):** published the compact TileLang implementation, benchmarks, and optimization walkthrough.
- **2025-02-15 — [WebGPU code generation](https://github.com/tile-ai/tilelang/pull/86):** added the initial WebGPU backend.
- **2025-02-12 — [TileLang v0.1.0](https://github.com/tile-ai/tilelang/releases/tag/v0.1.0):** published the first v0.1 public release.
- **2025-02-10 — [Debugging and layout tools](https://tilelang.com/tutorials/debug_tools_for_tilelang.html):** added `T.print` and fragment-layout visualization workflows.
- **2025-01-20 — [TileLang open sourced](https://github.com/tile-ai/tilelang):** made the project publicly available.

</details>

See [all releases](https://github.com/tile-ai/tilelang/releases) for complete changelogs and compatibility notes.

## Platform and Backend Support

TileLang requires Python 3.10 or newer. The default `auto` target detects CUDA, HIP, and Metal devices; select an explicit target when compiling for another backend or architecture. Hardware-specific instructions and optimizations remain subject to the selected architecture.

| Backend | Target | Platforms and hardware | Support level | Notes |
| --- | --- | --- | --- | --- |
| NVIDIA CUDA | `cuda` | Linux x86-64/AArch64, Windows x86-64; code paths from SM70 through SM120 | Primary | Release wheels and CI coverage; TMA, WGMMA, and TMEM features require the corresponding GPU architecture. |
| AMD ROCm/HIP | `hip` | Linux; CDNA and RDNA GPUs, including gfx942/gfx950 paths | Supported | Included in Linux wheels; a ROCm runtime is required. CI runs on a self-hosted gfx942 (MI300X) runner; gfx950 is not yet covered. |
| Apple Metal | `metal` | macOS on Apple silicon | Supported | Release wheels and CI coverage; Metal 4 cooperative tensors are available on supported M5 systems. |
| LLVM CPU | `llvm` | Host CPUs | Experimental | Build from source with `USE_LLVM=ON`; LLVM 15 or newer is required. |
| NVIDIA CuTe DSL | `cutedsl` | NVIDIA GPUs | Experimental | Requires `nvidia-cutlass-dsl`. |
| WebGPU | `webgpu` | WebGPU runtimes | Experimental | Code generation and runtime integration are still evolving. |
| Tenstorrent | `tenstorrent` | Wormhole B0 and Blackhole | Experimental | Verified Device IR, in-memory TTL codegen, pinned TT-Lang compilation, and TTNN JIT integration. The current automatic path is ready for hardware testing but has not yet been validated on silicon. |
| Huawei Ascend | Ascend C / NPU IR | Ascend A2 and A3 | Ecosystem | Developed in [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) and the MLIR-based [tilelang-mlir-ascend](https://github.com/tile-ai/tilelang-mlir-ascend). |
| MetaX MACA | MACA adapter | MetaX C500 | Ecosystem | Developed in [tilelang-metax](https://github.com/tile-ai/tilelang-metax); requires the MACA software stack. |
| Moore Threads MUSA | `musa` | S5000, S4000, and M1000 | Ecosystem | Developed in [tilelang-musa](https://github.com/tile-ai/tilelang-musa) and released independently. |

Prebuilt wheels are published for Linux x86-64/AArch64, Windows x86-64, and macOS arm64. Ecosystem adapters live in separate repositories, are not included in the main TileLang release wheels, and may follow different compatibility schedules. See the [target guide](https://tilelang.com/get_started/targets.html) for target syntax, architecture options, and backend-specific notes, or the corresponding adapter repository for installation and tested-device details.

## Installation

Install the latest stable release from PyPI:

```bash
pip install tilelang
```

Verify the installation:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

Nightly wheels provide recent features and fixes before the next stable release:

```bash
pip install tilelang --find-links https://tile-ai.github.io/whl/nightly
```

Nightly builds may be less stable than official releases. For source builds, editable installs, Docker, ROCm setup, pip-provided CUDA toolchains, or a custom TVM checkout, follow the complete [installation guide](https://tilelang.com/get_started/Installation.html).

### Tenstorrent development installation

The Tenstorrent backend is currently a source-checkout feature. It requires an
explicit `wormhole_b0` or `blackhole` target and the pinned compiler/runtime
pair below:

| Component | Required version |
| --- | --- |
| Python | CPython 3.12 |
| TT-Lang | commit `4ca4b89de95242ccb9bca8ade71200ac81a59e38`, package `1.1.9.dev20260824+light` |
| TTNN | `0.75.0` |

On Linux x86-64, install Git, a C/C++ compiler, CMake 3.26 or newer, Ninja,
Python 3.12 development files, zlib, and libedit. Then initialize the checkout
and create an isolated environment:

```bash
git submodule update --init --recursive

python3.12 -m venv .venv-tt
source .venv-tt/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install "ttnn==0.75.0" pytest-timeout
```

Install the exact cp312 manylinux 2.34 TT-Lang wheel for the pinned commit. In
this development workspace it is stored at:

```bash
export TILELANG_TT_WHEEL=../code/build/ttlang/dist/tt_lang-1.1.9.dev20260824+light-cp312-cp312-manylinux_2_34_x86_64.whl
test -f "$TILELANG_TT_WHEEL"
sha256sum "$TILELANG_TT_WHEEL"
python -m pip install --no-deps "$TILELANG_TT_WHEEL"
```

The expected SHA256 is:

```text
0d562db2d8b80e6a12854b9f8c699d06c179cbd196b1453c70c96c530319d784
```

Do not substitute a nearby TT-Lang nightly. TileLang checks
`ttl.build_info()["ttlang"]` because the TTL dialect and Python compiler ABI are
revision-specific.

Build the source checkout without a CUDA dependency:

```bash
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_TENSTORRENT=ON"
python -m pip install -e . -v --no-build-isolation --no-deps

# Reconfigure and rebuild after native C++ changes.
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF \
  -DUSE_TENSTORRENT=ON
ninja -C build
```

Verify the pinned environment and hardware-free compiler path:

```bash
python - <<'PY'
from importlib.metadata import version
import tilelang
import ttl
import ttnn

assert ttl.build_info()["ttlang"] == "4ca4b89de95242ccb9bca8ade71200ac81a59e38"
assert version("ttnn") == "0.75.0"
assert callable(ttnn.generic_op)
print("tilelang", tilelang.__version__)
print("tt-lang", ttl.build_info())
print("ttnn", version("ttnn"))
PY

python -m pytest -q \
  testing/python/tenstorrent/codegen/test_tilelang_tenstorrent_codegen_v07.py
```

#### Additional hardware requirements

Python packages alone are not sufficient for device execution. A hardware host
must also provide:

- a compatible Tenstorrent kernel driver and firmware;
- readable and writable `/dev/tenstorrent/<id>` device nodes;
- configured hugepages, normally under `/dev/hugepages` and, on some systems,
  `/dev/hugepages-1G`;
- `tt-smi` or equivalent Tenstorrent device-health tooling;
- the SFPI runtime installed after TT-Lang with `tt-lang-setup-sfpi`;
- `pytest-timeout`, using the thread timeout method for device deadlocks.

Driver, firmware, hugepage, and device-permission setup is machine-specific and
is not installed by this repository. Keep those components compatible with the
pinned TTNN/TT-Metal runtime.

Before running TileLang kernels, ensure the target device is idle and verify
that TTNN can open it:

```bash
tt-lang-setup-sfpi
unset TTLANG_COMPILE_ONLY
export TT_DEVICE_ID=0

ls -l /dev/tenstorrent
tt-smi

python - <<'PY'
import os
import ttnn

print("available devices:", ttnn.GetNumAvailableDevices())
print("architecture:", ttnn.get_arch_name())
dispatch = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
device = ttnn.open_device(
    device_id=int(os.environ.get("TT_DEVICE_ID", "0")),
    dispatch_core_config=dispatch,
)
try:
    print("opened device:", device.id())
finally:
    ttnn.close_device(device)
PY
```

Run the hardware tests in stages. Start with TTNN DRAM/L1 transfers, then the
single-core TileLang kernels, and finally the complete default suite:

```bash
python -m pytest -v \
  testing/python/tenstorrent/hardware/test_device_smoke.py \
  --timeout-method=thread

python -m pytest -v \
  testing/python/tenstorrent/hardware/test_elementwise_add.py \
  testing/python/tenstorrent/hardware/test_reduction.py \
  testing/python/tenstorrent/hardware/test_elementwise_chain.py \
  testing/python/tenstorrent/hardware/test_repeated_launch.py \
  --timeout-method=thread

python -m pytest -v testing/python/tenstorrent/hardware \
  --timeout-method=thread
```

Without a device, all 13 hardware tests are expected to skip; that is not
hardware validation. The experimental multi-core SUMMA test is disabled by
default because the pinned TT-Lang EmitC path can currently terminate on a
native block-argument assertion. It runs in an isolated subprocess when
explicitly enabled:

```bash
TILELANG_TT_RUN_EXPERIMENTAL_SUMMA=1 \
python -m pytest -v \
  testing/python/tenstorrent/hardware/test_multicore_summa.py \
  --timeout-method=thread
```

See the [hardware test guide](testing/python/tenstorrent/hardware/README.md) for
the current test matrix and support boundaries. Required full-K FP32 matmul and
dynamic DFB family selection remain explicit capability gates.

## Quick Start

The following example defines, compiles, runs, and verifies an FP16 GEMM kernel with FP32 accumulation and a fused ReLU epilogue. It uses PyTorch CUDA tensors; PyTorch uses the same `cuda` device name on ROCm systems. TileLang selects the target automatically from the current environment.

```python
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def matmul_relu(A, B, block_M: int = 128, block_N: int = 128, block_K: int = 32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


M = N = K = 1024
a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)
c = matmul_relu(a, b)
torch.testing.assert_close(c, torch.relu(a @ b), rtol=1e-2, atol=1e-2)
print("GEMM + ReLU passed.")
```

`@tilelang.jit` specializes the kernel for the input shape and compile-time arguments on first use. `T.Pipelined` stages global-to-shared transfers, `T.gemm` maps the tile operation to the target backend, and `T.Parallel` expresses the elementwise ReLU epilogue. Continue with the [language basics](https://tilelang.com/programming_guides/language_basics.html), then explore the [GEMM examples](https://github.com/tile-ai/tilelang/tree/main/examples/gemm) for layouts, autotuning, and architecture-specific optimizations.

## Examples

- **Start here:** [quickstart](https://github.com/tile-ai/tilelang/blob/main/examples/quickstart.py) and [elementwise kernels](https://github.com/tile-ai/tilelang/tree/main/examples/elementwise)
- **GEMM and quantization:** [GEMM](https://github.com/tile-ai/tilelang/tree/main/examples/gemm), [grouped GEMM](https://github.com/tile-ai/tilelang/tree/main/examples/grouped_gemm), [FP8 GEMM](https://github.com/tile-ai/tilelang/tree/main/examples/gemm_fp8), [dequantization GEMM](https://github.com/tile-ai/tilelang/tree/main/examples/dequantize_gemm), and [block-scaled GEMM](https://github.com/tile-ai/tilelang/tree/main/examples/blockscaled_gemm_sm100)
- **Attention and sequence models:** [FlashAttention](https://github.com/tile-ai/tilelang/tree/main/examples/flash_attention), [Flash Decoding](https://github.com/tile-ai/tilelang/tree/main/examples/flash_decoding), [block-sparse attention](https://github.com/tile-ai/tilelang/tree/main/examples/blocksparse_attention), [linear attention](https://github.com/tile-ai/tilelang/tree/main/examples/linear_attention), and [GDN](https://github.com/tile-ai/tilelang/tree/main/examples/gdn)
- **Model workloads:** [DeepSeek MLA](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla), [DeepSeek V3.2](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32), [DeepSeek V4](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v4), and [DeepSeek mHC](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mhc)
- **Architecture-specific kernels:** [AMD](https://github.com/tile-ai/tilelang/tree/main/examples/amd), [TCGEN05](https://github.com/tile-ai/tilelang/tree/main/examples/gemm_tcgen05), and [SM120](https://github.com/tile-ai/tilelang/tree/main/examples/gemm_sm120)
- **Compiler and debugging tools:** [analyzer](https://github.com/tile-ai/tilelang/tree/main/examples/analyze), [layout visualization](https://github.com/tile-ai/tilelang/tree/main/examples/plot_layout), [AutoDD](https://github.com/tile-ai/tilelang/tree/main/examples/autodd), and [IKET](https://github.com/tile-ai/tilelang/tree/main/examples/iket)

Browse the [complete examples directory](https://github.com/tile-ai/tilelang/tree/main/examples) for additional operators, tests, and architecture-specific implementations.

## Benchmark Summary

TileLang achieves exceptional performance across a variety of computational patterns. Comprehensive benchmark scripts and settings are available at [tilelang-benchmark](https://github.com/tile-ai/tilelang-benchmark). Below are selected results showcasing its capabilities:

- MLA Decoding Performance on H100

  <div style="display: flex; gap: 10px; justify-content: center;">
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs64_float16.png" alt="mla decode performance bs64 on H100" width="100%" />
    </div>
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs128_float16.png" alt="mla decode performance bs128 on H100" width="100%" />
    </div>
  </div>

- Flash Attention Performance on H100

  <div align="center">    <img src="./images/mha_performance_h100.png" alt="operator performance on H100" width=80% />
  </div>

- Matmul Performance on GPUs (RTX 4090, A100, H100, MI300X)

  <div>
    <img src="./images/op_benchmark_consistent_gemm_fp16.png" alt="gemm fp16 performance on Gpus" />
  </div>

- Dequantize Matmul Performance on A100

  <div>
    <img src="./images/op_benchmark_a100_wq_gemv.png" alt="dequantize gemv performance on A100" />
  </div>

---

## Join the Discussion

Welcome to join our Discord community for discussions, support, and collaboration!

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/TUrHyJnKPG)

## Acknowledgments

We would like to express our gratitude to the [TVM](https://github.com/apache/tvm) community for their invaluable contributions. The initial version of this project was mainly developed by [LeiWang1999](https://github.com/LeiWang1999), [chengyupku](https://github.com/chengyupku) and [nox-410](https://github.com/nox-410) with supervision from Prof. [Zhi Yang](https://yangzhihome.github.io) at Peking University. Part of this work was carried out during an internship at Microsoft Research, where Dr. Lingxiao Ma, Dr. Yuqing Xia, Dr. Jilong Xue, and Dr. Fan Yang offered valuable advice and support. We deeply appreciate their mentorship and contributions.
