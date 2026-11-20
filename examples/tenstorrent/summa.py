"""Construct frontend IR for SUMMA using collective broadcast syntax.

``T.comm.broadcast`` constructs initial TIRX containing ``tl.tt.transfer``.
Executable TTL lowering and device execution are not implemented by this example.

The frontend captures the following contract for subsequent lowering:

* Core coordinates are ``(x, y)`` and are compile-time specializable.
* ``dst_cores`` as one Core coordinate selects a point-to-point contract.
* ``dst_cores`` as ``CoreRange`` selects a collective contract. A singleton
  ``CoreRange`` remains collective and is not rewritten as point-to-point.
* The collective destination range includes the source. Lowering keeps the
  source payload in a private send generation and delivers the multicast into
  the same logical compute buffer on every Core.
* The same symbolic shared buffer may appear as ``src`` and ``dst`` because it
  denotes a different Core-local instance on every Core. Source-inclusive
  delivery creates a new local generation instead of sharing one transaction
  between the send and compute consumers.
* The statement is logically synchronous. Transfer handles, waits, DFB
  lifecycle operations, Pipe records, and physical NoC details are generated
  by the backend.
* Repeated calls from ``T.serial`` are occurrences of one static communication
  pattern, not dynamically created topologies.

``make_summa`` preserves collective row/column distribution by passing a
``CoreRange`` to each transfer. The backend may lower that collective to target
multicast, but the frontend does not decompose it into independent unicasts.
"""

from tilelang.tenstorrent import language as T


_DFB_TILE = 32
_MAX_MVP_ACCUMULATOR_TILES = 4


def make_summa(
    core_m: int = 8,
    core_n: int = 8,
    k_stages: int = 16,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    output_dtype: str | None = None,
):
    """Return a fixed-source SUMMA PrimFunc using ``T.comm.broadcast``.

    Each Core owns one output block. Column zero loads and broadcasts one A
    panel per row; row zero loads and broadcasts one B panel per column.
    """

    for name, value in (
        ("core_m", core_m),
        ("core_n", core_n),
        ("k_stages", k_stages),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, block in (
        ("block_m", block_m),
        ("block_n", block_n),
        ("block_k", block_k),
    ):
        if not isinstance(block, int) or isinstance(block, bool) or block <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if block % _DFB_TILE:
            raise ValueError(f"{name} must be a multiple of {_DFB_TILE}; tail blocks are not supported")
    accumulator_tiles = (block_m // _DFB_TILE) * (block_n // _DFB_TILE)
    if accumulator_tiles > _MAX_MVP_ACCUMULATOR_TILES:
        raise ValueError(
            "SUMMA MVP supports at most "
            f"{_MAX_MVP_ACCUMULATOR_TILES} accumulator tiles, got "
            f"{accumulator_tiles} from block_m={block_m}, block_n={block_n}"
        )

    output_dtype = input_dtype if output_dtype is None else output_dtype
    dtype_config = (input_dtype, accum_dtype, output_dtype)
    supported_dtype_configs = {
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "float32", "bfloat16"),
        ("float32", "float32", "float32"),
    }
    if dtype_config not in supported_dtype_configs:
        raise ValueError(
            "unsupported Tenstorrent SUMMA dtype combination: "
            f"input={input_dtype}, accumulation={accum_dtype}, "
            f"output={output_dtype}; supported combinations are "
            "(bfloat16, bfloat16, bfloat16), "
            "(bfloat16, float32, bfloat16), and "
            "(float32, float32, float32)"
        )

    m = core_m * block_m
    n = core_n * block_n
    k = k_stages * block_k
    panel_annotations = {
        "tt.dfb_block_count": 2,
        "tt.tile_shape": (_DFB_TILE, _DFB_TILE),
    }

    @T.prim_func
    def summa(
        A: T.Tensor((m, k), input_dtype),
        B: T.Tensor((k, n), input_dtype),
        C: T.Tensor((m, n), output_dtype),
    ):
        with T.Kernel(core_n, core_m, threads=1) as (core_x, core_y):
            a_panel = T.alloc_shared((block_m, block_k), input_dtype, annotations=panel_annotations)
            b_panel = T.alloc_shared((block_k, block_n), input_dtype, annotations=panel_annotations)
            c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            row_begin = core_y * block_m
            col_begin = core_x * block_n

            T.clear(c_local)
            for stage in T.serial(k_stages):
                k_begin = stage * block_k

                # A: the leftmost Core loads and broadcasts along its complete
                # row, including loopback delivery to the source.
                if core_x == 0:
                    T.copy(A[row_begin : row_begin + block_m, k_begin : k_begin + block_k], a_panel)
                if core_n > 1:
                    T.comm.broadcast(src=a_panel, dst=a_panel, src_core=(0, core_y), dst_cores=T.comm.CoreRange(begin=(0, core_y), end=(core_n, core_y + 1)))

                # B: the topmost Core loads and broadcasts along its complete
                # column, including loopback delivery to the source.
                if core_y == 0:
                    T.copy(B[k_begin : k_begin + block_k, col_begin : col_begin + block_n], b_panel)
                if core_m > 1:
                    T.comm.broadcast(src=b_panel, dst=b_panel, src_core=(core_x, 0), dst_cores=T.comm.CoreRange(begin=(core_x, 0), end=(core_x + 1, core_m)))

                T.gemm(a_panel, b_panel, c_local, clear_accum=False)

            T.copy(c_local, C[row_begin : row_begin + block_m, col_begin : col_begin + block_n])

    return summa


def make_summa_2x2(
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
):
    """Return the 2x2 configuration with singleton collective destinations."""

    return make_summa(core_m=2, core_n=2, k_stages=2, block_m=block_m, block_n=block_n, block_k=block_k, input_dtype=input_dtype, accum_dtype=accum_dtype)


def make_summa_atom_2x2():
    """Return the all-BF16 128^3 collective-transfer configuration."""

    return make_summa(core_m=2, core_n=2, k_stages=2, block_m=64, block_n=64, block_k=64, input_dtype="bfloat16", accum_dtype="bfloat16", output_dtype="bfloat16")


if __name__ == "__main__":
    from pathlib import Path

    output_path = Path(__file__).resolve().with_suffix(".tirx")
    output_path.write_text(make_summa().script() + "\n", encoding="utf-8")
    print(f"TIRX written to {output_path}")
