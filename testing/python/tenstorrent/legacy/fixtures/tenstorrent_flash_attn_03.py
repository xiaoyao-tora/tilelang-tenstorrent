"""Build FlashAttention frontend IR; this is not yet an executable TT kernel.

Only aligned shapes are accepted. T.Tiles operates on rank-2 shared and fragment
buffers. Every GEMM has a uniquely initialized accumulator; the online output
state remains compute-local while padded row statistics stay in shared DFBs.
TTL lowering, compute materialization and hardware budgets still require backend
support and device validation.
"""

import math

from tilelang.tenstorrent import language as T


_DFB_TILE = 32
_NEG_LARGE = -1.0e30
_MAX_MVP_ACCUMULATOR_TILES = 4


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_causal_lengths(seq_q, seq_kv):
    if seq_q > seq_kv:
        raise ValueError(
            "Bottom-right causal attention requires seq_q <= seq_kv in this "
            f"version, got seq_q={seq_q}, seq_kv={seq_kv}; fully masked rows "
            "are not supported"
        )


def make_causal_mask(seq_q: int, seq_kv: int | None = None):
    """Return a NumPy FP32 bottom-right mask shared by all batches/heads.

    The causal kernel's fourth argument must contain this 0/-1e30 mask, converted
    to the configured kernel dtype before launch. The finite sentinel assumes
    finite inputs with attention scores well above -1e30; its accuracy on the
    target FPU/SFPU must be validated on hardware. This correctness-first
    representation costs O(seq_q * seq_kv) storage.
    """
    import numpy as np

    seq_kv = seq_q if seq_kv is None else seq_kv
    _positive_int("seq_q", seq_q)
    _positive_int("seq_kv", seq_kv)
    _validate_causal_lengths(seq_q, seq_kv)
    query = np.arange(seq_q)[:, None]
    key = np.arange(seq_kv)[None, :]
    allowed = key <= query + (seq_kv - seq_q)
    return np.where(allowed, np.float32(0), np.float32(_NEG_LARGE))


def _make_row_broadcast(core_q: int, core_bh: int):
    if core_q == 1:
        return None
    # Column zero forwards its locally loaded panel and also computes its Q tile.
    return T.comm.PipeNet(
        [T.comm.Pipe(src=(0, row), dst=T.comm.CoreRange(begin=(1, row), end=(core_q, row + 1))) for row in range(core_bh)]
    )


def make_flash_attention(
    batch: int = 2,
    heads: int = 8,
    seq_q: int = 1024,
    seq_kv: int | None = None,
    head_dim: int = 64,
    *,
    causal: bool = False,
    core_q: int = 8,
    core_bh: int = 4,
    block_q: int = 32,
    block_kv: int = 32,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "bfloat16",
    output_dtype: str | None = None,
):
    """Return IR for scaled dot-product attention in BHSD layout.

    Non-causal ABI: (Q, K, V, Output). Causal ABI: (Q, K, V, Mask, Output),
    where Mask is the [seq_q, seq_kv] array from make_causal_mask converted to
    input_dtype. The kernel computes QK / sqrt(D) + Mask. Arbitrary masks and
    fully masked rows are outside this example's contract.

    All dimensions must be positive; blocks/head_dim must be 32-aligned and
    sequence lengths plus batch/head/query waves must divide their blocks/grid.
    core_q=1 reads K/V directly. Otherwise column zero loads and forwards K/V
    before consuming the same panel in compute. Producer-side forwarding must
    remain in the load transaction so the DFB still has one consumer. The
    caller must choose a grid and buffer sizes fitting the target budgets.

    The frontend contract accepts all-BF16 or all-FP32. Mixed BF16-input/FP32-
    state attention remains deferred until the backend defines its intermediate
    result and compute-configuration contracts. This function constructs IR,
    not a compiled executable.
    """
    seq_kv = seq_q if seq_kv is None else seq_kv
    output_dtype = input_dtype if output_dtype is None else output_dtype
    dimensions = {
        "batch": batch,
        "heads": heads,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "head_dim": head_dim,
        "core_q": core_q,
        "core_bh": core_bh,
        "block_q": block_q,
        "block_kv": block_kv,
    }
    for name, value in dimensions.items():
        _positive_int(name, value)
    if not isinstance(causal, bool):
        raise ValueError(f"causal must be a bool, got {causal!r}")
    for name in ("block_q", "block_kv", "head_dim"):
        if dimensions[name] % _DFB_TILE:
            raise ValueError(f"{name}={dimensions[name]} must be divisible by {_DFB_TILE}")
    for name, value, divisor_name, divisor in (
        ("seq_q", seq_q, "block_q", block_q),
        ("seq_kv", seq_kv, "block_kv", block_kv),
        ("q_tiles", seq_q // block_q, "core_q", core_q),
        ("batch * heads", batch * heads, "core_bh", core_bh),
    ):
        if value % divisor:
            raise ValueError(f"{name}={value} must be divisible by {divisor_name}={divisor}")
    if causal:
        _validate_causal_lengths(seq_q, seq_kv)
    dtype_config = (input_dtype, accum_dtype, output_dtype)
    frontend_dtype_configs = {
        ("bfloat16", "bfloat16", "bfloat16"),
        ("float32", "float32", "float32"),
    }
    if dtype_config not in frontend_dtype_configs:
        raise ValueError(
            "unsupported Tenstorrent FlashAttention dtype combination: "
            f"input={input_dtype}, accumulation={accum_dtype}, output={output_dtype}; "
            "supported frontend combinations are BF16/BF16/BF16 and "
            "FP32/FP32/FP32. "
            "BF16 input with FP32 online state is deferred"
        )

    scores_fragment_tiles = (block_q // _DFB_TILE) * (block_kv // _DFB_TILE)
    output_fragment_tiles = (block_q // _DFB_TILE) * (head_dim // _DFB_TILE)
    row_fragment_tiles = block_q // _DFB_TILE
    fragment_tile_counts = {
        "scores accumulator": scores_fragment_tiles,
        "PV accumulator": output_fragment_tiles,
        "online output accumulator": output_fragment_tiles,
        "row reduction": row_fragment_tiles,
    }
    for name, tile_count in fragment_tile_counts.items():
        if tile_count > _MAX_MVP_ACCUMULATOR_TILES:
            raise ValueError(
                f"{name} requires {tile_count} tiles; the FlashAttention MVP supports at most {_MAX_MVP_ACCUMULATOR_TILES} tiles"
            )
    peak_fragment_tiles = output_fragment_tiles + max(
        scores_fragment_tiles + row_fragment_tiles,
        output_fragment_tiles,
    )
    if peak_fragment_tiles > _MAX_MVP_ACCUMULATOR_TILES:
        raise ValueError(
            f"FlashAttention MVP fragment live set requires {peak_fragment_tiles} tiles; at most {_MAX_MVP_ACCUMULATOR_TILES} are supported"
        )

    q_annotations = {"tt.dfb_block_count": 1, "tt.tile_shape": (32, 32)}
    dfb_annotations = {"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)}
    q_waves = (seq_q // block_q) // core_q
    batch_head_waves = (batch * heads) // core_bh
    kv_tiles = seq_kv // block_kv
    attention_scale = 1.0 / math.sqrt(head_dim)
    softmax_log2e = math.log2(math.e)
    # Separate topology identities keep K and V transfer payloads unambiguous.
    k_broadcast = _make_row_broadcast(core_q, core_bh)
    v_broadcast = _make_row_broadcast(core_q, core_bh)

    @T.macro
    def load_kv_tile(batch_index, head_index, kv_tile, K, V, k_buffer, v_buffer):
        kv_begin = kv_tile * block_kv
        T.copy(K[batch_index, head_index, kv_begin : kv_begin + block_kv, 0:head_dim], k_buffer)
        T.copy(V[batch_index, head_index, kv_begin : kv_begin + block_kv, 0:head_dim], v_buffer)

    @T.macro
    def attention_body(Q, K, V, Mask, Output):
        with T.Kernel(core_q, core_bh, threads=1) as (core_x, core_y):
            q_shared = T.alloc_shared((block_q, head_dim), input_dtype, annotations=q_annotations)
            k_compute_shared = T.alloc_shared((block_kv, head_dim), input_dtype, annotations=dfb_annotations)
            v_compute_shared = T.alloc_shared((block_kv, head_dim), input_dtype, annotations=dfb_annotations)
            if causal:
                mask_shared = T.alloc_shared((block_q, block_kv), accum_dtype, annotations=dfb_annotations)

            probabilities_shared = T.alloc_shared((block_q, block_kv), input_dtype, annotations=dfb_annotations)
            output_shared = T.alloc_shared((block_q, head_dim), output_dtype, annotations=dfb_annotations)

            # Physical row vectors occupy padded tiles. Reduction writes column
            # zero; Tiles broadcasts that column and writes a full padded block.
            row_max_shared = T.alloc_shared((block_q, 32), accum_dtype, annotations=dfb_annotations)
            row_max_previous_shared = T.alloc_shared((block_q, 32), accum_dtype, annotations=dfb_annotations)
            chunk_max_shared = T.alloc_shared((block_q, 32), accum_dtype, annotations=dfb_annotations)
            row_scale_shared = T.alloc_shared((block_q, 32), accum_dtype, annotations=dfb_annotations)
            tile_sum_shared = T.alloc_shared((block_q, 32), accum_dtype, annotations=dfb_annotations)
            row_sum_shared = T.alloc_shared((block_q, 32), accum_dtype, annotations=dfb_annotations)

            # Scores stay compute-local through scale/mask/exp/reductions. The
            # online output accumulator is an outer-loop-carried fragment value.
            scores_fragment = T.alloc_fragment((block_q, block_kv), accum_dtype)
            pv_fragment = T.alloc_fragment((block_q, head_dim), accum_dtype)
            output_acc_fragment = T.alloc_fragment((block_q, head_dim), accum_dtype)
            row_fragment = T.alloc_fragment((block_q, 1), accum_dtype)

            for batch_head_wave in T.serial(batch_head_waves):
                for query_wave in T.serial(q_waves):
                    batch_head_index = batch_head_wave * core_bh + core_y
                    batch_index = batch_head_index // heads
                    head_index = batch_head_index % heads
                    query_tile = query_wave * core_q + core_x
                    query_begin = query_tile * block_q
                    T.copy(Q[batch_index, head_index, query_begin : query_begin + block_q, 0:head_dim], q_shared)
                    T.clear(output_acc_fragment)
                    T.clear(row_sum_shared)
                    T.fill(row_max_shared, _NEG_LARGE)

                    # Keep a constant transfer count for the first PipeNet
                    # bring-up. The causal mask removes invisible KV positions.
                    for kv_tile in T.serial(kv_tiles):
                        if k_broadcast is None:
                            load_kv_tile(batch_index, head_index, kv_tile, K, V, k_compute_shared, v_compute_shared)
                        else:
                            if T.comm.is_src(k_broadcast):
                                load_kv_tile(batch_index, head_index, kv_tile, K, V, k_compute_shared, v_compute_shared)
                            for pipe in T.comm.foreach_src(k_broadcast):
                                T.copy(k_compute_shared, pipe)
                            for pipe in T.comm.foreach_dst(k_broadcast):
                                T.copy(pipe, k_compute_shared)
                            for pipe in T.comm.foreach_src(v_broadcast):
                                T.copy(v_compute_shared, pipe)
                            for pipe in T.comm.foreach_dst(v_broadcast):
                                T.copy(pipe, v_compute_shared)

                        T.gemm(q_shared, k_compute_shared, scores_fragment, transpose_B=True, clear_accum=True)

                        if causal:
                            kv_begin = kv_tile * block_kv
                            T.copy(Mask[query_begin : query_begin + block_q, kv_begin : kv_begin + block_kv], mask_shared)
                            for row, col in T.Tiles(scores_fragment):
                                scores_fragment[row, col] = scores_fragment[row, col] * attention_scale + mask_shared[row, col]
                        else:
                            for row, col in T.Tiles(scores_fragment):
                                scores_fragment[row, col] = scores_fragment[row, col] * attention_scale

                        T.reduce_max(scores_fragment, row_fragment, dim=1, clear=True)
                        T.copy(row_fragment, chunk_max_shared[0:block_q, 0:1])
                        T.copy(row_max_shared, row_max_previous_shared)
                        for row, col in T.Tiles(row_max_shared):
                            row_max_shared[row, col] = T.max(row_max_previous_shared[row, 0], chunk_max_shared[row, 0])
                        for row, col in T.Tiles(row_scale_shared):
                            row_scale_shared[row, col] = T.exp2((row_max_previous_shared[row, 0] - row_max_shared[row, 0]) * softmax_log2e)
                        for row, col in T.Tiles(scores_fragment):
                            scores_fragment[row, col] = T.exp2((scores_fragment[row, col] - row_max_shared[row, 0]) * softmax_log2e)

                        for row, col in T.Tiles(probabilities_shared):
                            probabilities_shared[row, col] = T.cast(scores_fragment[row, col], input_dtype)
                        T.reduce_sum(scores_fragment, row_fragment, dim=1, clear=True)
                        T.copy(row_fragment, tile_sum_shared[0:block_q, 0:1])
                        for row, col in T.Tiles(row_sum_shared):
                            row_sum_shared[row, col] = row_sum_shared[row, col] * row_scale_shared[row, col] + tile_sum_shared[row, 0]

                        T.gemm(probabilities_shared, v_compute_shared, pv_fragment, clear_accum=True)
                        for row, col in T.Tiles(output_acc_fragment):
                            output_acc_fragment[row, col] = output_acc_fragment[row, col] * row_scale_shared[row, 0] + pv_fragment[row, col]

                    for row, col in T.Tiles(output_shared):
                        output_shared[row, col] = T.cast(output_acc_fragment[row, col] / row_sum_shared[row, 0], output_dtype)
                    T.copy(output_shared, Output[batch_index, head_index, query_begin : query_begin + block_q, 0:head_dim])

    if causal:

        @T.prim_func
        def flash_attention(
            Q: T.Tensor((batch, heads, seq_q, head_dim), input_dtype),
            K: T.Tensor((batch, heads, seq_kv, head_dim), input_dtype),
            V: T.Tensor((batch, heads, seq_kv, head_dim), input_dtype),
            Mask: T.Tensor((seq_q, seq_kv), accum_dtype),
            Output: T.Tensor((batch, heads, seq_q, head_dim), output_dtype),
        ):
            attention_body(Q, K, V, Mask, Output)

    else:

        @T.prim_func
        def flash_attention(
            Q: T.Tensor((batch, heads, seq_q, head_dim), input_dtype),
            K: T.Tensor((batch, heads, seq_kv, head_dim), input_dtype),
            V: T.Tensor((batch, heads, seq_kv, head_dim), input_dtype),
            Output: T.Tensor((batch, heads, seq_q, head_dim), output_dtype),
        ):
            attention_body(Q, K, V, None, Output)

    return flash_attention


if __name__ == "__main__":
    print(make_flash_attention().script())
