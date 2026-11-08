"""Resolve current Device logical DFB streams to finite TTL bindings."""

from __future__ import annotations

from dataclasses import dataclass

from tvm import IRModule, arith, tirx


class ResourceResolutionError(NotImplementedError):
    """A Device resource cannot be resolved to one static logical DFB group."""


_DFB_OPERAND_POSITIONS = {
    "tl.tt.dfb_reserve": (0,),
    "tl.tt.dfb_wait": (0,),
    "tl.tt.dfb_copy_wait": (0,),
    "tl.tt.dfb_release": (0,),
    "tl.tt.dfb_load": (0,),
    "tl.tt.dfb_pipe_send_record": (3,),
    "tl.tt.dfb_pipe_recv_record": (3,),
    "tl.tt.dfb_pipe_wait_record": (3,),
    "tl.tt.tensor_to_dfb_nd": (1,),
    "tl.tt.dfb_to_tensor_nd": (0,),
    "tl.tt.gemm_update": (0, 1),
    "tl.tt.accumulator_materialize": (1,),
    "tl.tt.compute_value_store": (1,),
    "tl.tt.compute_value_gemm": (1, 2),
    "tl.tt.dfb_local_copy": (0, 1),
    "tl.tt.dfb_gather_copy": (0, 1),
    "tl.tt.dfb_collective_reduce": (0, 1, 2),
}


@dataclass(frozen=True)
class ResolvedDFB:
    storage_group_id: int


class CurrentDFBResolver:
    """Resolve static Device stream IDs without table expansion."""

    def __init__(self, mod: IRModule):
        attrs = mod.attrs
        self._groups = {int(group.storage_group_id): group for group in attrs["tt.dfb_storage_groups"]}

    def resolve(self, dfb_id: int) -> ResolvedDFB:
        if dfb_id not in self._groups:
            raise ResourceResolutionError(f"DFB stream {dfb_id} is missing from current storage groups")
        return ResolvedDFB(dfb_id)

    def groups_for_slot(self, slot: str) -> tuple[int, ...]:
        """Return groups whose typed producer/consumer contract uses ``slot``."""

        return tuple(
            sorted(
                group_id for group_id, group in self._groups.items() if str(group.producer_slot) == slot or str(group.consumer_slot) == slot
            )
        )


def resolve_current_function_groups(mod: IRModule) -> dict[str, tuple[int, ...]]:
    """Return the finite storage-group bindings required by each kernel slot.

    Device operations directly reference finite typed producer/consumer streams.
    """

    resolver = CurrentDFBResolver(mod)
    analyzer = arith.Analyzer()
    result: dict[str, tuple[int, ...]] = {}
    for global_var, function in mod.functions.items():
        slot = str(function.attrs["tt.kernel_slot"])
        groups = set(resolver.groups_for_slot(slot))
        failure: list[ResourceResolutionError] = []

        def visit(node, failure=failure, groups=groups, global_var=global_var, slot=slot):
            if failure:
                return
            if not isinstance(node, tirx.Call) or not hasattr(node.op, "name"):
                return
            name = str(node.op.name)
            positions = _DFB_OPERAND_POSITIONS.get(name, ())
            if name == "tl.tt.dfb_compute":
                positions = range(len(node.args))
            elif name == "tl.tt.compute_value":
                positions = range(1, len(node.args))
            for position in positions:
                value = analyzer.simplify(node.args[position])
                if not isinstance(value, tirx.IntImm):
                    continue
                try:
                    resolved = resolver.resolve(int(value)).storage_group_id
                    if resolved not in groups:
                        failure.append(
                            ResourceResolutionError(
                                f"function {global_var.name_hint} operation {name} uses "
                                f"storage group {resolved} outside slot {slot} ownership"
                            )
                        )
                        return
                except ResourceResolutionError as error:
                    failure.append(error)
                    return

        tirx.stmt_functor.post_order_visit(function.body, visit)
        if failure:
            raise failure[0]
        result[str(global_var.name_hint)] = tuple(sorted(groups))
    return result


__all__ = [
    "CurrentDFBResolver",
    "ResolvedDFB",
    "ResourceResolutionError",
    "resolve_current_function_groups",
]
