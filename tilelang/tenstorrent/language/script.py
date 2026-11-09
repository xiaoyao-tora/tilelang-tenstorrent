"""TVMScript constructors for Tenstorrent TIR intrinsics."""

from __future__ import annotations

from tvm import ir, tirx
import tvm.tirx.script.parser as _tirx_parser


_FIXED_DTYPES = {
    "is_src": "bool",
    "is_dst": "bool",
    "is_active": "bool",
    "pipe_src": "int32",
    "pipe_dst": "int32",
    "pipe_dst_range": "int32",
    "core_x": "int32",
    "core_y": "int32",
}

_DYNAMIC_DTYPE = {
    "dfb_load",
    "compute_broadcast",
    "compute_value_load",
}

_VOID_INTRINSICS = {
    "pipe_send",
    "pipe_recv",
    "transfer",
    "all_gather",
    "all_reduce",
    "tile_add",
    "tile_compute",
    "dfb_compute",
    "compute_precision",
    "compute_value",
    "compute_value_store",
    "compute_value_gemm",
    "compute_value_state_init",
    "compute_value_state_input",
    "compute_value_state_yield",
    "compute_value_state_result",
    "tensor_to_dfb_nd",
    "dfb_to_tensor_nd",
    "dfb_reserve",
    "dfb_wait",
    "dfb_copy_wait",
    "dfb_release",
    "dfb_pipe_send",
    "dfb_pipe_recv",
    "dfb_pipe_wait",
    "dfb_pipe_send_record",
    "dfb_pipe_recv_record",
    "dfb_pipe_wait_record",
    "tensor_to_dfb",
    "dfb_to_tensor",
    "dfb_add",
    "dfb_local_copy",
    "dfb_gather_copy",
    "dfb_collective_reduce",
    "accumulator_init",
    "gemm_update",
    "accumulator_materialize",
}


class _TenstorrentTIRNamespace:
    def __getattr__(self, name: str):
        if name not in _FIXED_DTYPES and name not in _DYNAMIC_DTYPE and name not in _VOID_INTRINSICS:
            raise AttributeError(f"unknown Tenstorrent TIR intrinsic {name!r}")

        def build(*args, **annotations):
            if name in _DYNAMIC_DTYPE:
                if not args:
                    raise TypeError(f"T.tt.{name} requires its result dtype as the first argument")
                dtype, *operands = args
            else:
                dtype = _FIXED_DTYPES.get(name, "void")
                operands = args

            def normalize_annotation(value):
                if isinstance(value, str):
                    return tirx.StringImm(value)
                if isinstance(value, (int, bool)):
                    return tirx.const(value)
                if isinstance(value, (list, tuple)):
                    return [normalize_annotation(item) for item in value]
                return value

            normalized = {}
            for key, value in annotations.items():
                semantic_key = f"tt.{key[4:]}" if key.startswith("tt__") else key
                normalized[semantic_key] = normalize_annotation(value)
            return tirx.call_intrin(
                dtype,
                ir.Op.get(f"tl.tt.{name}"),
                *operands,
                annotations=normalized or None,
            )

        return build


# ``TScriptPrinterName = tt.<op>`` makes generated source use ``T.tt.<op>``.
# Register the namespace on the canonical tirx parser facade used by
# ``tvm.script.from_source``.
tt = _TenstorrentTIRNamespace()
_tirx_parser.tt = tt


__all__ = ["tt"]
