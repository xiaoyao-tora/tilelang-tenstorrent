"""Structured Device statement and control-flow emission."""

from __future__ import annotations

from copy import copy

from tvm import tirx


class TTLStatementEmitterMixin:
    def emit_statement(self, statement) -> None:
        if isinstance(statement, tirx.SeqStmt):
            for child in statement.seq:
                self.emit_statement(child)
            return
        if isinstance(statement, tirx.Bind):
            if isinstance(statement.value, tirx.Call) and str(statement.value.op.name) in {
                "tl.tt.tensor_ref",
                "tl.tt.dfb_stream_ref",
                "tl.tt.pipe_net_ref",
                "tl.tt.accumulator_ref",
                "tl.tt.compute_value_ref",
                "tl.tt.compute_state_ref",
            }:
                return
            raise NotImplementedError(f"TTL statement emitter has no mapping for Device Bind {statement.var.name}")
        if isinstance(statement, tirx.Evaluate):
            if isinstance(statement.value, tirx.IntImm) and int(statement.value) == 0:
                return
            if not isinstance(statement.value, tirx.Call):
                raise NotImplementedError(f"TTL statement emitter requires a Device call, got {statement.value}")
            with self.location(statement.value.span):
                self.emit_call(statement.value)
            return
        if isinstance(statement, tirx.SBlockRealize):
            self.emit_sblock(statement)
            return
        if isinstance(statement, tirx.For):
            self.emit_for(statement)
            return
        if isinstance(statement, tirx.IfThenElse):
            self.emit_if(statement)
            return
        raise NotImplementedError(f"TTL statement emitter has no mapping for {type(statement).__name__}")

    def emit_sblock(self, realize) -> None:
        block = realize.block
        name = str(block.name_hint)
        if name in ("root", "tt_device_core_work"):
            self.emit_statement(block.body)
            return
        if name != "tt_device_stream_work":
            raise NotImplementedError(f"TTL statement emitter has no mapping for SBlock {name!r}")
        role_attr = block.annotations.get("tt.communication_role")
        if role_attr is None:
            self.emit_statement(block.body)
            return
        role = str(role_attr)
        pipe_net_ids = tuple(map(int, block.annotations["tt.pipe_net_ids"]))
        if len(pipe_net_ids) != 1:
            raise NotImplementedError("TTL PipeNet stream regions currently require exactly one static PipeNet")
        pipe_net_id = pipe_net_ids[0]
        records = self.pipenet_attributes[pipe_net_id]
        if role == "source":
            operation = self.bindings.ttl.pipenet_foreach_src(records)
            pipe_type = self.bindings.ttl.SelectedPipeSrcType.get(self.context)
        elif role == "destination":
            operation = self.bindings.ttl.pipenet_foreach_dst(records)
            pipe_type = self.bindings.ttl.SelectedPipeDstType.get(self.context)
        else:
            raise ValueError(f"invalid Device communication role {role!r}")
        region_block = self.bindings.ir.Block.create_at_start(operation.body, [pipe_type])
        previous_pipe, previous_net = self.selected_pipe, self.selected_pipe_net
        with self.bindings.ir.InsertionPoint(region_block):
            self.selected_pipe = region_block.arguments[0]
            self.selected_pipe_net = pipe_net_id
            self.emit_statement(block.body)
            self.bindings.ttl.yield_([])
        self.selected_pipe, self.selected_pipe_net = previous_pipe, previous_net

    def _snapshot_transient_state(self):
        return (
            copy(self.reserved),
            copy(self.values),
            copy(self.pending_copies),
            copy(self.pending_pipe_copies),
            copy(self.completed_pipe_transfers),
            copy(self.assembled),
            copy(self.compute_values),
            copy(self.loop_states),
            copy(self.accumulators),
        )

    def _restore_transient_state(self, snapshot) -> None:
        (
            self.reserved,
            self.values,
            self.pending_copies,
            self.pending_pipe_copies,
            self.completed_pipe_transfers,
            self.assembled,
            self.compute_values,
            self.loop_states,
            self.accumulators,
        ) = snapshot

    def emit_if(self, statement) -> None:
        condition = self.scalar.emit(statement.condition)
        operation = self.bindings.scf.IfOp(
            cond=condition,
            results_=[],
            has_else=statement.else_case is not None,
        )
        snapshot = self._snapshot_transient_state()
        with self.bindings.ir.InsertionPoint(operation.then_block):
            self.emit_statement(statement.then_case)
            self.bindings.scf.YieldOp([])
        self._restore_transient_state(snapshot)
        if statement.else_case is not None:
            with self.bindings.ir.InsertionPoint(operation.else_block):
                self.emit_statement(statement.else_case)
                self.bindings.scf.YieldOp([])
            self._restore_transient_state(snapshot)

    def _loop_state_ids(self, body) -> tuple[int, ...]:
        result = set()
        initialized = set()

        def visit(node):
            if not isinstance(node, tirx.Call) or not hasattr(node.op, "name"):
                return
            if str(node.op.name) == "tl.tt.compute_value_state_init":
                initialized.add(self.resource_id(node.args[0], "tl.tt.compute_state_ref"))
                return
            if str(node.op.name) not in (
                "tl.tt.compute_value_state_input",
                "tl.tt.compute_value_state_yield",
            ):
                return
            position = 1 if str(node.op.name).endswith("_input") else 0
            result.add(
                self.resource_id(
                    node.args[position],
                    "tl.tt.compute_state_ref",
                )
            )

        tirx.stmt_functor.post_order_visit(body, visit)
        # Nested loops may update an outer state, but a state initialized inside
        # this body is local to this iteration, not an incoming iter_arg.
        return tuple(sorted(result - initialized))

    def _loop_accumulator_ids(self, body) -> tuple[int, ...]:
        result = set()
        initialized = set()

        def visit(node):
            if isinstance(node, tirx.Call) and str(node.op.name) == "tl.tt.accumulator_init":
                initialized.add(self.resource_id(node.args[0], "tl.tt.accumulator_ref"))
            if isinstance(node, tirx.Call) and str(node.op.name) == "tl.tt.gemm_update":
                result.add(
                    self.resource_id(
                        node.args[2],
                        "tl.tt.accumulator_ref",
                    )
                )

        tirx.stmt_functor.post_order_visit(body, visit)
        return tuple(sorted(result - initialized))

    def emit_for(self, statement) -> None:
        lower = self.scalar.emit(statement.min)
        upper = self.bindings.arith.addi(lower, self.scalar.emit(statement.extent))
        step = self.scalar.index_constant(1)
        state_ids = self._loop_state_ids(statement.body)
        accumulator_ids = self._loop_accumulator_ids(statement.body)
        carried = []
        for identifier in state_ids:
            if identifier not in self.loop_states:
                raise RuntimeError(f"compute loop state {identifier} has no initializer")
            carried.append(self.loop_states[identifier])
        for identifier in accumulator_ids:
            if identifier not in self.accumulators:
                raise RuntimeError(f"accumulator {identifier} has no initializer")
            carried.append(self.accumulators[identifier])
        operation = self.bindings.scf.ForOp(lower, upper, step, carried)
        transient = self._snapshot_transient_state()
        previous_scalar = self.scalar_values.get(statement.loop_var)
        with self.bindings.ir.InsertionPoint(operation.body):
            self.scalar_values[statement.loop_var] = operation.induction_variable
            inner = iter(operation.inner_iter_args)
            for identifier in state_ids:
                self.loop_states[identifier] = next(inner)
            for identifier in accumulator_ids:
                self.accumulators[identifier] = next(inner)
            self.emit_statement(statement.body)
            yields = [self.loop_states[identifier] for identifier in state_ids]
            yields.extend(self.accumulators[identifier] for identifier in accumulator_ids)
            self.bindings.scf.YieldOp(yields)
        self._restore_transient_state(transient)
        if previous_scalar is None:
            self.scalar_values.pop(statement.loop_var, None)
        else:
            self.scalar_values[statement.loop_var] = previous_scalar
        results = iter(operation.results)
        for identifier in state_ids:
            self.loop_states[identifier] = next(results)
        for identifier in accumulator_ids:
            self.accumulators[identifier] = next(results)


__all__ = ["TTLStatementEmitterMixin"]
