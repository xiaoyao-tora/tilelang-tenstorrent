"""In-memory initial TTL artifact and runtime ABI metadata."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from tilelang import tvm

from ..contracts import TTLANG_REFERENCE_COMMIT


@dataclass(frozen=True, slots=True)
class TensorABI:
    global_arg_index: int
    shape: tuple[int, ...]
    dtype: str
    tile_shape: tuple[int, ...]
    memory_space: str
    memory_layout: str
    effect: str
    alias_group: int
    storage_format: str = ""


@dataclass(frozen=True, slots=True)
class TTLModuleMetadata:
    target_arch: str
    launch_grid: tuple[int, int]
    tensor_abi: tuple[TensorABI, ...]
    precision_policy: tuple[tuple[str, str, str], ...]
    contract_digest: str
    dependency_revisions: tuple[tuple[str, str], ...]
    source_map_summary: tuple[tuple[str, str, int, int], ...]
    build_digest: str


def _static_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise NotImplementedError(f"TTL codegen requires static {name}, got {value}") from error


def build_module_metadata(device_mod) -> TTLModuleMetadata:
    attrs = device_mod.attrs
    grid = attrs["tt.launch_grid"]
    tensors = tuple(
        TensorABI(
            global_arg_index=int(tensor.global_arg_index),
            shape=tuple(_static_int(value, "Tensor shape") for value in tensor.shape),
            dtype=str(tensor.dtype),
            tile_shape=tuple(_static_int(value, "Tensor tile shape") for value in tensor.tile_shape),
            memory_space=str(tensor.memory_space),
            memory_layout=str(tensor.memory_layout),
            effect=str(tensor.effect),
            alias_group=int(tensor.alias_group),
            storage_format=str(getattr(tensor, "storage_format", "")),
        )
        for tensor in attrs["tt.tensor_table"]
    )
    precision = []
    source_map = []
    for function in device_mod.functions.values():
        span = function.span
        source_map.append(
            (
                str(function.attrs["global_symbol"]),
                "" if span is None or span.source_name is None else str(span.source_name.name),
                0 if span is None else int(span.line),
                0 if span is None else int(span.column),
            )
        )
        requirements = function.attrs.get("tt.compute_requirements")
        if requirements is None:
            continue
        precision.append(
            (
                str(function.attrs["global_symbol"]),
                str(requirements.destination_width),
                str(requirements.matmul_full_fp32),
            )
        )
    build_digest = _codegen_build_digest()
    contract = {
        "device_ir": tvm.ir.save_json(device_mod),
        "ttlang": TTLANG_REFERENCE_COMMIT,
        "codegen": build_digest,
    }
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    return TTLModuleMetadata(
        target_arch=str(attrs["tt.target_arch"]),
        launch_grid=(int(grid.x), int(grid.y)),
        tensor_abi=tensors,
        precision_policy=tuple(sorted(precision)),
        contract_digest=digest,
        dependency_revisions=(("tt-lang", TTLANG_REFERENCE_COMMIT),),
        source_map_summary=tuple(sorted(source_map)),
        build_digest=build_digest,
    )


def _codegen_build_digest() -> str:
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "artifact.py",
        "attributes.py",
        "bindings.py",
        "expression.py",
        "function.py",
        "module.py",
        "records.py",
        "resources.py",
        "scalar.py",
        "stmt.py",
        "types.py",
    ):
        path = directory / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class TenstorrentTTLModuleNode:
    """Own one mutable MLIR module together with its originating context."""

    format = "ttl"
    eager_source_inspection = False

    def __init__(self, context, module, metadata: TTLModuleMetadata, bindings) -> None:
        self._context = context
        self._module = module
        self.metadata = metadata
        self.bindings = bindings
        self._mutation_lock = threading.Lock()
        self._compiled = False

    @property
    def context(self):
        return self._context

    @property
    def mlir_module(self):
        return self._module

    @property
    def operation(self):
        return self._module.operation

    @property
    def body(self):
        return self._module.body

    def inspect_source(self, _format: str = "") -> str:
        return str(self._module)

    @contextmanager
    def exclusive_mutation(self) -> Iterator[Any]:
        if not self._mutation_lock.acquire(blocking=False):
            raise RuntimeError("the same initial TTL Module cannot run concurrent TT-Lang pipelines")
        try:
            if self._compiled:
                raise RuntimeError("the initial TTL Module has already been consumed by TT-Lang")
            yield self._module
            self._compiled = True
        finally:
            self._mutation_lock.release()

    def __str__(self) -> str:
        return self.inspect_source()


__all__ = [
    "TTLModuleMetadata",
    "TensorABI",
    "TenstorrentTTLModuleNode",
    "build_module_metadata",
]
