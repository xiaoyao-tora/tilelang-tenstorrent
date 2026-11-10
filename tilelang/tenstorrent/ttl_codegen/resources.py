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
    "tl.tt.dfb_pipe_send_record": (2,),
    "tl.tt.dfb_pipe_recv_record": (2,),
    "tl.tt.dfb_pipe_wait_record": (2,),
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

_RESOURCE_REF_NAMES = frozenset(
    {
        "tl.tt.tensor_ref",
        "tl.tt.dfb_stream_ref",
        "tl.tt.pipe_net_ref",
        "tl.tt.accumulator_ref",
    }
)


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

    def resolve_operand(self, operand, aliases) -> ResolvedDFB:
        if isinstance(operand, tirx.Var):
            if operand not in aliases:
                raise ResourceResolutionError(f"unbound DFB resource ref {operand.name}")
            kind, identifier = aliases[operand]
            if kind != "tl.tt.dfb_stream_ref":
                raise ResourceResolutionError(f"resource ref {operand.name} is not a DFB stream")
            return self.resolve(identifier)
        if isinstance(operand, tirx.Call) and str(operand.op.name) == "tl.tt.dfb_stream_ref":
            if len(operand.args) != 1 or not isinstance(operand.args[0], tirx.IntImm):
                raise ResourceResolutionError("DFB stream ref does not contain one static ID")
            return self.resolve(int(operand.args[0]))
        value = arith.Analyzer().simplify(operand)
        if isinstance(value, tirx.IntImm):
            raise ResourceResolutionError("current Device DFB operand is a raw integer instead of a stream ref")
        raise ResourceResolutionError("Device DFB operand is not a static stream ref")

    def groups_for_slot(self, slot: str) -> tuple[int, ...]:
        """Return groups whose typed producer/consumer contract uses ``slot``."""

        return tuple(
            sorted(
                group_id for group_id, group in self._groups.items() if str(group.producer_slot) == slot or str(group.consumer_slot) == slot
            )
        )


def _resource_alias_prologue(function):
    aliases = {}
    identities = set()
    prologue_vars = set()
    failure = []
    body = function.body
    if isinstance(body, tirx.SBlockRealize) and str(body.block.name_hint) == "root":
        body = body.block.body
    statements = list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]
    prologue_open = True
    for statement in statements:
        value = statement.value if isinstance(statement, tirx.Bind) else None
        name = str(value.op.name) if isinstance(value, tirx.Call) and hasattr(value.op, "name") else ""
        if name not in _RESOURCE_REF_NAMES:
            prologue_open = False
            continue
        if not prologue_open:
            failure.append(ResourceResolutionError("Device resource aliases must form the PrimFunc prologue"))
            continue
        if len(value.args) != 1 or not isinstance(value.args[0], tirx.IntImm):
            failure.append(ResourceResolutionError(f"resource alias {statement.var.name} has no static ID"))
            continue
        identifier = int(value.args[0])
        identity = (name, identifier)
        if identity in identities:
            failure.append(ResourceResolutionError(f"resource {name}({identifier}) has more than one function alias"))
            continue
        identities.add(identity)
        prologue_vars.add(statement.var)
        aliases[statement.var] = identity

    def reject_non_prologue_alias(node):
        if not isinstance(node, tirx.Bind) or not isinstance(node.value, tirx.Call):
            return
        if str(node.value.op.name) in _RESOURCE_REF_NAMES and node.var not in prologue_vars:
            failure.append(ResourceResolutionError(f"resource alias {node.var.name} is outside the PrimFunc prologue"))

    tirx.stmt_functor.post_order_visit(function.body, reject_non_prologue_alias)
    if failure:
        raise failure[0]
    return aliases


def resolve_current_function_groups(mod: IRModule) -> dict[str, tuple[int, ...]]:
    """Return the finite storage-group bindings required by each kernel slot.

    Device operations directly reference finite typed producer/consumer streams.
    """

    resolver = CurrentDFBResolver(mod)
    result: dict[str, tuple[int, ...]] = {}
    for global_var, function in mod.functions.items():
        slot = str(function.attrs["tt.kernel_slot"])
        groups = set(resolver.groups_for_slot(slot))
        failure: list[ResourceResolutionError] = []
        aliases = _resource_alias_prologue(function)

        def visit(
            node,
            aliases=aliases,
            failure=failure,
            groups=groups,
            global_var=global_var,
            slot=slot,
        ):
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
                try:
                    resolved = resolver.resolve_operand(node.args[position], aliases).storage_group_id
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
