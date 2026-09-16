"""Map verified logical requirements to concrete TTL kernel attributes."""


def fp32_destination(requirements):
    """Return None when Lower deliberately left destination width unconstrained."""
    width = str(requirements.destination_width)
    if width == "unconstrained":
        return None
    if width == "bits16_required":
        return False
    if width == "bits32_required":
        return True
    raise ValueError(f"Invalid verified Tenstorrent destination width {width!r}")


def attach_function_attributes(bindings, ctx, operation, function, module, dfb_count):
    ir, ttl = bindings.ir, bindings.ttl
    attrs = operation.attributes
    compute = str(function.attrs["tt.kernel_slot"]) == "trisc"
    attrs["ttl.kernel_thread"] = bindings.ttkernel.ir.ThreadTypeAttr.get(ctx, "compute" if compute else "noc")
    i32 = ir.IntegerType.get_signless(32, ctx)
    attrs["ttl.base_cta_index"] = ir.IntegerAttr.get(i32, dfb_count)
    attrs["ttl.crta_indices"] = ir.ArrayAttr.get(
        [ir.IntegerAttr.get(i32, int(index)) for index in function.attrs["tt.tensor_arg_indices"]], ctx
    )
    logical = function.attrs["tt.logical_kernel"]
    kind = ttl.ir.LogicalKernelKind.Compute if compute else ttl.ir.LogicalKernelKind.DataMovement
    attrs["ttl.logical_kernel"] = ttl.LogicalKernelAttr.get(
        ctx,
        kind,
        str(logical.kernel_id),
        str(module.attrs["tt.operation_identity"].operation_id),
        None,
    )
    if compute:
        requirements = function.attrs.get("tt.compute_requirements")
        if requirements is not None:
            enabled = fp32_destination(requirements)
            if enabled is not None:
                attrs["fp32_dest_acc_en"] = ir.BoolAttr.get(enabled, ctx)
    else:
        attrs["ttl.noc_index"] = ir.IntegerAttr.get(i32, int(function.attrs["tt.noc_index"]))
