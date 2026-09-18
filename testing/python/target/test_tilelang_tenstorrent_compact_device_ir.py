"""Lossless v9 representation inside the existing Lower passes."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import device_ir as D, lower_tenstorrent_ir, transform
from tilelang.tenstorrent.pipeline import TENSTORRENT_LOWER_PASS_ORDER
from tvm import tirx

from testing.python.target.test_tilelang_tenstorrent_device_ir_size import attention_module as attention_module
from testing.python.target.test_tilelang_tenstorrent_flash_attention import kernel


@pytest.fixture(scope="module")
def compact_module(attention_module):
    return D.compact_device_ir(attention_module, force=True)


def test_compaction_is_lossless_and_uses_existing_passes(attention_module, compact_module):
    assert int(compact_module.attrs[D.DEVICE_IR_VERSION_ATTR]) == 9
    assert int(compact_module.attrs[D.COMPACT_ORIGINAL_VERSION_ATTR]) == 8
    assert D.DFB_TABLE_ATTR not in compact_module.attrs
    assert D.COMPUTE_VALUE_TABLE_ATTR not in compact_module.attrs
    assert all("Compact" not in name and "Share" not in name for name in TENSTORRENT_LOWER_PASS_ORDER)
    expanded = D.expand_device_ir(compact_module)
    tvm.ir.assert_structural_equal(attention_module, expanded)
    # Resource rows remain distinct, with the original fragment identities.
    originals = attention_module.attrs[D.COMPUTE_VALUE_TABLE_ATTR]
    values = expanded.attrs[D.COMPUTE_VALUE_TABLE_ATTR]
    assert len(values) == len(originals)
    assert len({hash(value) for value in values}) == len(values)
    assert all(value.buffer.same_as(original.buffer) for value, original in zip(values, originals))
    assert transform.VerifyTenstorrentDeviceIR()(compact_module).same_as(compact_module)
    assert D.compact_device_ir(compact_module, force=True).same_as(compact_module)
    assert D.compact_device_ir(attention_module).same_as(attention_module)
    assert D.expand_device_ir(attention_module).same_as(attention_module)
    tvm.ir.assert_structural_equal(compact_module, transform.InferTenstorrentComputeRequirements()(compact_module))


def test_compact_exports_roundtrip_and_reduce_size():
    # Exercise automatic compaction at a representative size. Tiny modules keep
    # v8 because descriptor-family metadata has a fixed serialization overhead.
    frontend = tvm.IRModule({"main": kernel.make_flash_attention(batch=1, heads=2, seq_q=256, seq_kv=512, core_q=2, core_bh=1)})
    compact_module = lower_tenstorrent_ir(frontend, tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"}))
    assert int(compact_module.attrs[D.DEVICE_IR_VERSION_ATTR]) == 9
    attention_module = D.expand_device_ir(compact_module)
    serialized = tvm.ir.save_json(compact_module)
    restored = tvm.ir.load_json(serialized)
    tvm.ir.assert_structural_equal(compact_module, restored)
    tvm.ir.assert_structural_equal(attention_module, D.expand_device_ir(restored))
    transform.VerifyTenstorrentDeviceIR()(restored)
    compact_text_size = len(compact_module.script(show_meta=True).encode())
    original_text_size = len(attention_module.script(show_meta=True).encode())
    assert compact_text_size < original_text_size * 0.75
    compact_json_size = len(serialized)
    original_json_size = len(tvm.ir.save_json(attention_module))
    assert compact_json_size < original_json_size * 0.75


@pytest.mark.parametrize("version", [1, 7, 8])
def test_compact_metadata_cannot_downgrade_version(compact_module, version):
    invalid = compact_module.with_attr(D.DEVICE_IR_VERSION_ATTR, version)
    with pytest.raises(ValueError, match="compact metadata requires Device IR v9"):
        transform.VerifyTenstorrentDeviceIR()(invalid)


@pytest.mark.parametrize("attribute", [D.COMPACT_DFB_FAMILIES_ATTR, D.COMPACT_VALUE_FAMILIES_ATTR, D.COMPACT_ORIGINAL_VERSION_ATTR])
def test_compact_metadata_is_required(compact_module, attribute):
    invalid = compact_module.without_attr(attribute)
    with pytest.raises(ValueError, match="v9 requires"):
        transform.VerifyTenstorrentDeviceIR()(invalid)


def test_compact_and_expanded_tables_cannot_coexist(attention_module, compact_module):
    invalid = compact_module.with_attr(D.DFB_TABLE_ATTR, attention_module.attrs[D.DFB_TABLE_ATTR])
    with pytest.raises(ValueError):
        transform.VerifyTenstorrentDeviceIR()(invalid)


def test_compact_verification_does_not_regenerate_requirements(compact_module):
    invalid = tvm.IRModule(compact_module.functions, attrs=compact_module.attrs)
    for global_var, func in invalid.functions.items():
        if str(func.attrs[D.KERNEL_SLOT_ATTR]) == "trisc":
            requirements = func.attrs[D.COMPUTE_REQUIREMENTS_ATTR]
            bad = D.ComputeRequirements("bits32_required", requirements.matmul_full_fp32, requirements.accumulators)
            invalid.update_func(global_var, func.with_attr(D.COMPUTE_REQUIREMENTS_ATTR, bad))
            break
    with pytest.raises(ValueError, match="compute_requirements disagrees"):
        transform.VerifyTenstorrentDeviceIR()(invalid)


def test_compact_descriptor_expansion_is_bounded(compact_module):
    families = list(compact_module.attrs[D.COMPACT_DFB_FAMILIES_ATTR])
    first = families[0]
    families[0] = D.DeviceIRDescriptorFamily(first.prototype, D.DeviceIRIntColumn(0, 1, 4_000_001), first.fields, first.source_parts)
    invalid = compact_module.with_attr(D.COMPACT_DFB_FAMILIES_ATTR, families)
    with pytest.raises(ValueError, match="budget"):
        transform.VerifyTenstorrentDeviceIR()(invalid)


def test_compact_statement_expansion_is_bounded(compact_module):
    invalid = tvm.IRModule(compact_module.functions, attrs=compact_module.attrs)
    global_var, func = next(iter(invalid.functions.items()))
    loop = tirx.For(tirx.Var("repeat", "int32"), 0, 4_000_001, tirx.ForKind.SERIAL, tirx.Evaluate(0))
    invalid.update_func(global_var, func.with_body(loop))
    with pytest.raises(ValueError, match="budget"):
        transform.VerifyTenstorrentDeviceIR()(invalid)
