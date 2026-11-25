"""Tenstorrent physical storage annotations, independent of logical dtype."""

from collections.abc import Mapping

from tvm.tirx import Buffer
from tvm.tirx.script.builder.ir import func_attr


def annotate_storage_format(formats: Mapping[Buffer, str]) -> None:
    """Set physical storage for global Tensor arguments inside a PrimFunc.

    For example, ``T.annotate_storage_format({A: "bfp_bf8", C: "bfp_bf4"})``
    keeps both Tensor dtypes as ``bfloat16``. The empty string selects ordinary
    dtype storage. Shared buffers use ``alloc_shared``'s ``tt.storage_format``
    allocation annotation instead. Supply all Tensor formats in one call.
    """
    if not isinstance(formats, Mapping):
        raise TypeError("Storage formats must be a mapping from Tensor buffers to strings.")
    annotations = {}
    for buffer, storage_format in formats.items():
        if not isinstance(buffer, Buffer) or buffer.scope() != "global":
            raise ValueError("Storage format annotations require global Tensor buffers.")
        if not isinstance(storage_format, str):
            raise TypeError("`tt.storage_format` must be a string.")
        if storage_format not in ("", "bfp_bf8", "bfp_bf4"):
            raise ValueError(f"Unsupported Tenstorrent storage_format: {storage_format!r}.")
        if storage_format and str(buffer.dtype) != "bfloat16":
            raise ValueError("BFP storage_format requires logical bfloat16.")
        annotations[buffer.data] = storage_format
    func_attr({"tt.tensor_storage_formats": annotations})