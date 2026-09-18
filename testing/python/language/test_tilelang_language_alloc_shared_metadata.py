import pytest

from tilelang import language as T
from tilelang import tvm


_ALLOC_BUFFER_ANNOTATIONS = "tl.alloc_buffer_annotations"


def _make_kernel(annotations, shape=(64, 128)):
    @T.prim_func
    def kernel(A: T.Tensor(shape, T.bfloat16)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared(shape, T.bfloat16, annotations=annotations)
            A[0, 0] = shared[0, 0]

    return kernel


def _find_metadata_block(func):
    blocks = []

    def collect(node):
        if isinstance(node, tvm.tirx.SBlock) and _ALLOC_BUFFER_ANNOTATIONS in node.annotations:
            blocks.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, collect)
    assert len(blocks) == 1
    return blocks[0]


def test_alloc_shared_without_annotations_keeps_existing_ir_shape():
    kernel = _make_kernel(None)
    blocks = []

    def collect(node):
        if isinstance(node, tvm.tirx.SBlock) and node.alloc_buffers:
            blocks.append(node)

    tvm.tirx.stmt_functor.post_order_visit(kernel.body, collect)
    assert len(blocks) == 1
    assert _ALLOC_BUFFER_ANNOTATIONS not in blocks[0].annotations


def test_alloc_shared_records_tt_metadata_by_buffer_identity():
    kernel = _make_kernel(
        {
            "tt.dfb_block_count": 2,
            "tt.tile_shape": (32, 32),
        }
    )

    block = _find_metadata_block(kernel)
    assert len(block.alloc_buffers) == 1
    buffer = block.alloc_buffers[0]
    metadata_by_buffer = block.annotations[_ALLOC_BUFFER_ANNOTATIONS]
    assert buffer.data in metadata_by_buffer

    metadata = metadata_by_buffer[buffer.data]
    assert metadata["tt.dfb_block_count"].value == 2
    assert [value.value for value in metadata["tt.tile_shape"]] == [32, 32]


def test_alloc_shared_keeps_metadata_separate_for_multiple_buffers():
    @T.prim_func
    def kernel(A: T.Tensor((64, 128), T.bfloat16)):
        with T.Kernel(1, threads=1):
            lhs = T.alloc_shared(
                (64, 128),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 1},
            )
            rhs = T.alloc_shared(
                (64, 128),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 3},
            )
            A[0, 0] = lhs[0, 0] + rhs[0, 0]

    block = _find_metadata_block(kernel)
    assert len(block.alloc_buffers) == 2
    metadata_by_buffer = block.annotations[_ALLOC_BUFFER_ANNOTATIONS]
    first, second = block.alloc_buffers
    assert metadata_by_buffer[first.data]["tt.dfb_block_count"].value == 1
    assert metadata_by_buffer[second.data]["tt.dfb_block_count"].value == 3


@pytest.mark.parametrize("block_count", [0, 33, 1.5, True])
def test_alloc_shared_rejects_invalid_dfb_block_count(block_count):
    with pytest.raises((TypeError, ValueError), match="tt.dfb_block_count"):
        _make_kernel({"tt.dfb_block_count": block_count})


@pytest.mark.parametrize("tile_shape", [(16, 32), (32,), (32, 0)])
def test_alloc_shared_rejects_unsupported_tile_shape(tile_shape):
    with pytest.raises((TypeError, ValueError), match="tt.tile_shape"):
        _make_kernel({"tt.tile_shape": tile_shape})


def test_alloc_shared_rejects_shape_not_divisible_by_tile():
    with pytest.raises(ValueError, match="must be divisible"):
        _make_kernel({"tt.tile_shape": (32, 32)}, shape=(48, 128))


def test_alloc_shared_rejects_tensor_backed_in_phase_1():
    with pytest.raises(NotImplementedError, match="tt.tensor_backed"):
        _make_kernel({"tt.tensor_backed": {"tensor": object(), "byte_offset": 0}})
