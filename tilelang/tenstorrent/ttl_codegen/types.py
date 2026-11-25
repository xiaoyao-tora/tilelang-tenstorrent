"""Device descriptor to TTL type conversion."""

from __future__ import annotations


class TTLTypeConverter:
    def __init__(self, bindings, context) -> None:
        self.bindings = bindings
        self.context = context

    @staticmethod
    def static_shape(values, name: str) -> list[int]:
        try:
            result = [int(value) for value in values]
        except (TypeError, ValueError) as error:
            raise NotImplementedError(f"TTL codegen requires static {name}") from error
        if any(value <= 0 for value in result):
            raise ValueError(f"TTL codegen requires positive {name}, got {result}")
        return result

    def tile_type(self, dtype, tile_shape):
        data_types = {
            "bfloat16": self.bindings.ttcore.DataType.BFloat16,
            "float32": self.bindings.ttcore.DataType.Float32,
            "int32": self.bindings.ttcore.DataType.Int32,
            "uint32": self.bindings.ttcore.DataType.UInt32,
            "uint16": self.bindings.ttcore.DataType.UInt16,
        }
        dtype_name = str(dtype)
        if dtype_name not in data_types:
            raise NotImplementedError(f"TTL tile dtype {dtype_name!r} is not supported")
        height, width = self.static_shape(tile_shape, "tile shape")
        return self.bindings.ttcore.ir.TileType.get(
            self.context,
            height,
            width,
            int(data_types[dtype_name]),
        )

    def storage_tile_type(self, descriptor):
        storage_format = str(descriptor.storage_format)
        if not storage_format:
            dtype = descriptor.element_dtype if hasattr(descriptor, "element_dtype") else descriptor.dtype
            return self.tile_type(dtype, descriptor.tile_shape)
        formats = {
            "bfp_bf8": self.bindings.ttcore.DataType.BFP_BFloat8,
            "bfp_bf4": self.bindings.ttcore.DataType.BFP_BFloat4,
        }
        if storage_format not in formats:
            raise ValueError(f"Unknown Tenstorrent storage_format {storage_format!r}")
        height, width = self.static_shape(descriptor.tile_shape, "tile shape")
        if (height, width) != (32, 32):
            raise ValueError("BFP storage requires 32x32 tiles")
        return self.bindings.ttcore.ir.TileType.get(self.context, height, width, int(formats[storage_format]))

    def logical_block_type(self, descriptor):
        return self.bindings.ir.RankedTensorType.get(
            self.static_shape(descriptor.block_shape_in_tiles, "DFB block shape"),
            self.tile_type(descriptor.element_dtype, descriptor.tile_shape),
        )

    def block_type(self, descriptor):
        return self.bindings.ir.RankedTensorType.get(
            self.static_shape(descriptor.block_shape_in_tiles, "DFB block shape"),
            self.storage_tile_type(descriptor),
        )

    def circular_buffer_type(self, descriptor):
        return self.bindings.ttl.CircularBufferType.get(
            self.context,
            self.static_shape(descriptor.block_shape_in_tiles, "DFB block shape"),
            self.storage_tile_type(descriptor),
            int(descriptor.block_count),
        )

    def tensor_type(self, descriptor):
        memory_space = str(descriptor.memory_space)
        buffer_types = {
            "dram": self.bindings.ttl.BufferType.DRAM,
            "l1": self.bindings.ttl.BufferType.L1,
        }
        if memory_space not in buffer_types:
            raise NotImplementedError(f"TTL Tensor memory space {memory_space!r} is not supported")
        memory_layout = str(descriptor.memory_layout)
        layouts = {
            "interleaved": self.bindings.ttl.TensorMemoryLayout.Interleaved,
            "height_sharded": self.bindings.ttl.TensorMemoryLayout.HeightSharded,
            "width_sharded": self.bindings.ttl.TensorMemoryLayout.WidthSharded,
            "block_sharded": self.bindings.ttl.TensorMemoryLayout.BlockSharded,
        }
        if memory_layout not in layouts:
            raise NotImplementedError(f"TTL Tensor memory layout {memory_layout!r} is not supported")
        tile_type = self.storage_tile_type(descriptor)
        logical_shape = self.static_shape(descriptor.shape, "Tensor shape")
        if descriptor.shard_spec is None:
            grid = [1, 1]
        else:
            domain = descriptor.shard_spec.core_domain
            grid = [
                int(domain.end.x) - int(domain.begin.x),
                int(domain.end.y) - int(domain.begin.y),
            ]
        layout = self.bindings.ttl.ir.LayoutAttr.get(
            self.context,
            logical_shape,
            tile_type,
            int(buffer_types[memory_space]),
            grid,
            int(layouts[memory_layout]),
        )
        return self.bindings.ir.RankedTensorType.get(
            self.static_shape(descriptor.tile_grid_shape, "Tensor tile-grid shape"),
            tile_type,
            layout,
        )

    def value_type(self, descriptor):
        logical_shape = self.static_shape(descriptor.logical_shape, "compute-value logical shape")
        tile_shape = self.static_shape(descriptor.compute_tile_shape, "compute-value tile shape")
        if len(logical_shape) != len(tile_shape):
            raise ValueError("compute-value logical and tile shapes have different ranks")
        block_shape = []
        for logical, tile in zip(logical_shape, tile_shape):
            block_shape.append((logical + tile - 1) // tile)
        return self.bindings.ir.RankedTensorType.get(
            block_shape,
            self.tile_type(descriptor.dtype, tile_shape[-2:]),
        )


__all__ = ["TTLTypeConverter"]
