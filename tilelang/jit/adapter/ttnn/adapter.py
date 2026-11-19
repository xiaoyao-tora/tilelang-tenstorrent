"""TileLang callable adapter for TT-Lang ``CompiledTTNNKernel`` objects."""

from __future__ import annotations

from functools import wraps
import importlib
from pathlib import Path
import threading

from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.tenstorrent.ttlang_adapter import compile_ttl_module
from tilelang.tenstorrent.ttl_codegen.artifact import TenstorrentTTLModuleNode, TensorABI


class TTNNKernelAdapter(BaseKernelAdapter):
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int] | int | None,
        func_or_mod,
        device_artifact,
        *,
        verbose: bool = False,
        compiler_options=None,
        compile_flags: list[str] | None = None,
    ) -> None:
        if not isinstance(device_artifact, TenstorrentTTLModuleNode):
            raise TypeError("TTNNKernelAdapter requires an in-memory TenstorrentTTLModuleNode")
        self.device_artifact = device_artifact
        self.tensor_abi = tuple(
            sorted(
                device_artifact.metadata.tensor_abi,
                key=lambda entry: entry.global_arg_index,
            )
        )
        metadata_outputs = [index for index, entry in enumerate(self.tensor_abi) if entry.effect in ("output", "inout")]
        if result_idx is None:
            result_idx = metadata_outputs
        normalized = [result_idx] if isinstance(result_idx, int) else list(result_idx)
        normalized = [index + len(params) if index < 0 else index for index in normalized]
        if normalized != metadata_outputs:
            raise ValueError(
                f"TileLang result_idx does not match the Tenstorrent Tensor ABI: expected {metadata_outputs}, got {normalized}"
            )
        self.verbose = verbose
        self._ttnn = importlib.import_module("ttnn")
        if compiler_options is None and compile_flags:
            compiler_options = importlib.import_module("ttl").CompilerOptions.from_string(" ".join(compile_flags))
        self.compiled_kernel = compile_ttl_module(
            device_artifact,
            compiler_options=compiler_options,
            verbose=verbose,
        )
        if self.compiled_kernel is None:
            raise RuntimeError("TT-Lang did not produce a CompiledTTNNKernel")
        self._call_lock = threading.Lock()
        super().__init__(func_or_mod, params=params, result_idx=normalized)

    def _dtype(self, dtype: str):
        mapping = {
            "bfloat16": self._ttnn.bfloat16,
            "float32": self._ttnn.float32,
        }
        try:
            return mapping[dtype]
        except KeyError as error:
            raise NotImplementedError(f"TTNN runtime dtype {dtype!r} is not supported") from error

    def _memory_config(self, memory_space: str):
        mapping = {
            "dram": self._ttnn.DRAM_MEMORY_CONFIG,
            "l1": self._ttnn.L1_MEMORY_CONFIG,
        }
        try:
            return mapping[memory_space]
        except KeyError as error:
            raise NotImplementedError(f"TTNN runtime memory space {memory_space!r} is not supported") from error

    def _validate_tensor(self, tensor, abi: TensorABI, index: int) -> None:
        tensor_type = getattr(self._ttnn, "Tensor", None)
        if tensor_type is None or not isinstance(tensor, tensor_type):
            raise TypeError(f"Tenstorrent argument {index} must be a device-resident ttnn.Tensor")
        if tensor.device() is None:
            raise ValueError(f"Tenstorrent argument {index} is not device-resident")
        if tuple(map(int, tensor.shape)) != abi.shape:
            raise ValueError(f"Tenstorrent argument {index} shape mismatch: expected {abi.shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != self._dtype(abi.dtype):
            raise ValueError(f"Tenstorrent argument {index} dtype mismatch: expected {abi.dtype}, got {tensor.dtype}")
        if tensor.layout != self._ttnn.TILE_LAYOUT:
            raise ValueError(f"Tenstorrent argument {index} must use ttnn.TILE_LAYOUT")
        get_tile = getattr(tensor, "get_tile", None)
        if callable(get_tile):
            actual_tile = tuple(map(int, get_tile().tile_shape))
            if actual_tile != abi.tile_shape:
                raise ValueError(f"Tenstorrent argument {index} tile mismatch: expected {abi.tile_shape}, got {actual_tile}")
        actual_memory = str(tensor.memory_config().buffer_type).lower()
        if abi.memory_space not in actual_memory:
            raise ValueError(
                f"Tenstorrent argument {index} memory mismatch: expected {abi.memory_space}, got {tensor.memory_config().buffer_type}"
            )
        actual_layout = str(tensor.memory_config().memory_layout).lower()
        if abi.memory_layout not in actual_layout:
            raise ValueError(
                f"Tenstorrent argument {index} memory layout mismatch: expected "
                f"{abi.memory_layout}, got {tensor.memory_config().memory_layout}"
            )

    @staticmethod
    def _storage_identity(tensor):
        buffer_address = getattr(tensor, "buffer_address", None)
        if callable(buffer_address):
            try:
                return (tensor.device(), int(buffer_address()))
            except (RuntimeError, TypeError, ValueError):
                pass
        return id(tensor)

    def _validate_aliases(self, tensors) -> None:
        for left_index, left_abi in enumerate(self.tensor_abi):
            for right_index in range(left_index + 1, len(self.tensor_abi)):
                right_abi = self.tensor_abi[right_index]
                same_group = left_abi.alias_group == right_abi.alias_group
                same_storage = self._storage_identity(tensors[left_index]) == self._storage_identity(tensors[right_index])
                if same_group != same_storage:
                    relation = "alias" if same_group else "not alias"
                    raise ValueError(f"Tenstorrent arguments {left_index} and {right_index} must {relation}")

    @staticmethod
    def _same_device(left, right) -> bool:
        if left is right:
            return True
        left_id = getattr(left, "id", None)
        right_id = getattr(right, "id", None)
        return callable(left_id) and callable(right_id) and left_id() == right_id()

    def _allocate_output(self, abi: TensorABI, device):
        if abi.memory_layout != "interleaved":
            raise NotImplementedError("TTNN output allocation currently requires interleaved Tensor layout")
        return self._ttnn.empty(
            abi.shape,
            dtype=self._dtype(abi.dtype),
            layout=self._ttnn.TILE_LAYOUT,
            device=device,
            memory_config=self._memory_config(abi.memory_space),
        )

    def _prepare_tensors(self, args):
        input_indices = [index for index, abi in enumerate(self.tensor_abi) if abi.effect != "output"]
        if len(args) not in (len(input_indices), len(self.tensor_abi)):
            raise ValueError(
                f"TTNN kernel expects {len(input_indices)} inputs or {len(self.tensor_abi)} preallocated tensors, got {len(args)}"
            )
        tensors = [None] * len(self.tensor_abi)
        if len(args) == len(self.tensor_abi):
            tensors[:] = args
        else:
            for index, tensor in zip(input_indices, args):
                tensors[index] = tensor
        device = None
        for index in input_indices:
            tensor = tensors[index]
            self._validate_tensor(tensor, self.tensor_abi[index], index)
            tensor_device = tensor.device()
            if device is None:
                device = tensor_device
            elif not self._same_device(device, tensor_device):
                raise ValueError("all TTNN kernel tensors must reside on the same device")
        if device is None:
            for tensor in tensors:
                if tensor is not None and isinstance(tensor, self._ttnn.Tensor):
                    device = tensor.device()
                    if device is not None:
                        break
        if device is None:
            raise ValueError("TTNN kernel requires a device-resident input or preallocated output")
        for index in self.result_idx:
            if tensors[index] is None:
                alias = next(
                    (
                        tensor
                        for candidate, tensor in enumerate(tensors)
                        if candidate != index
                        and tensor is not None
                        and self.tensor_abi[candidate].alias_group == self.tensor_abi[index].alias_group
                    ),
                    None,
                )
                tensors[index] = alias if alias is not None else self._allocate_output(self.tensor_abi[index], device)
            self._validate_tensor(tensors[index], self.tensor_abi[index], index)
            if not self._same_device(device, tensors[index].device()):
                raise ValueError("all TTNN kernel tensors must reside on the same device")
        self._validate_aliases(tensors)
        return tensors

    def _convert_torch_func(self):
        @wraps(self.compiled_kernel)
        def launcher(*args, **kwargs):
            if kwargs:
                raise TypeError("TTNNKernelAdapter accepts positional Tensor arguments only")
            tensors = self._prepare_tensors(args)
            with self._call_lock:
                self.compiled_kernel(*tensors)
            outputs = [tensors[index] for index in self.result_idx]
            if not outputs:
                return None
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

        return launcher

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        sources = []
        for path, _thread_type in self.compiled_kernel.kernel_paths:
            sources.append(Path(path).read_text(encoding="utf-8"))
        return "\n\n".join(sources)


__all__ = ["TTNNKernelAdapter"]
