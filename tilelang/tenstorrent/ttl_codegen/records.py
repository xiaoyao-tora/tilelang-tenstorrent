"""Resolve current Device transport families to ordered TTL PipeNet records."""

from __future__ import annotations

from dataclasses import dataclass

from tvm import IRModule


class RecordResolutionError(NotImplementedError):
    """A current Device transport family cannot be mapped to TTL records."""


@dataclass(frozen=True)
class ResolvedPipeRecord:
    communication_id: int
    record_index: int
    source: tuple[int, int]
    destination_begin: tuple[int, int]
    destination_end: tuple[int, int]
    contract: str


def _column_values(column) -> tuple[int, ...]:
    count = int(column.count)
    values = tuple(map(int, column.values))
    period = int(column.period)
    if count < 0 or period < 0 or period > count:
        raise RecordResolutionError("transport family has an invalid integer column")
    if period:
        if len(values) != period:
            raise RecordResolutionError("periodic transport column has the wrong value count")
        return tuple(values[index % period] + int(column.step) * (index // period) for index in range(count))
    if values:
        if len(values) != count:
            raise RecordResolutionError("explicit transport column has the wrong value count")
        return values
    return tuple(int(column.start) + int(column.step) * index for index in range(count))


def resolve_transport_records(mod: IRModule) -> dict[int, tuple[ResolvedPipeRecord, ...]]:
    """Decode only transport-family rows required by ``PipeNetRecordsAttr``."""

    if mod.attrs is None or "tt.transport_families" not in mod.attrs:
        raise RecordResolutionError("missing tt.transport_families")
    records: dict[int, list[ResolvedPipeRecord]] = {}
    required = (
        "pipe_net_id",
        "event_index",
        "source_x",
        "source_y",
        "destination_begin_x",
        "destination_begin_y",
        "destination_end_x",
        "destination_end_y",
    )
    for family in mod.attrs["tt.transport_families"]:
        fields = family.fields
        missing = [name for name in required if name not in fields]
        if missing:
            raise RecordResolutionError("transport family is missing fields: " + ", ".join(missing))
        columns = {name: _column_values(fields[name]) for name in required}
        count = int(family.positions.count)
        if any(len(column) != count for column in columns.values()):
            raise RecordResolutionError("transport family columns have inconsistent lengths")
        contract = str(family.prototype.contract)
        if contract not in ("point_to_point", "collective"):
            raise RecordResolutionError(f"unsupported transport contract {contract!r}")
        for row in range(count):
            communication = columns["pipe_net_id"][row]
            record = ResolvedPipeRecord(
                communication_id=communication,
                record_index=columns["event_index"][row],
                source=(columns["source_x"][row], columns["source_y"][row]),
                destination_begin=(columns["destination_begin_x"][row], columns["destination_begin_y"][row]),
                destination_end=(columns["destination_end_x"][row], columns["destination_end_y"][row]),
                contract=contract,
            )
            if record.destination_begin[0] >= record.destination_end[0] or record.destination_begin[1] >= record.destination_end[1]:
                raise RecordResolutionError("transport record has an empty destination range")
            if (
                record.destination_begin[0] <= record.source[0] < record.destination_end[0]
                and record.destination_begin[1] <= record.source[1] < record.destination_end[1]
            ):
                raise RecordResolutionError("source-exclusive transport record includes its source")
            records.setdefault(communication, []).append(record)
    result: dict[int, tuple[ResolvedPipeRecord, ...]] = {}
    for communication, entries in records.items():
        ordered = sorted(entries, key=lambda entry: entry.record_index)
        indices = [entry.record_index for entry in ordered]
        if len(indices) != len(set(indices)):
            raise RecordResolutionError(f"PipeNet {communication} has duplicate record indices")
        result[communication] = tuple(ordered)
    return result


__all__ = ["RecordResolutionError", "ResolvedPipeRecord", "resolve_transport_records"]
