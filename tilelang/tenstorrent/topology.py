"""Strict decoding of the frontend's frozen PipeNet descriptor.

Topology expansion and endpoint validation remain native lowering operations.
"""

import json

import tvm_ffi
from tvm.runtime import convert


@tvm_ffi.register_global_func("tl.tenstorrent.DecodePipeNet")
def _decode_pipenet(text):
    def fail(message):
        raise ValueError(f"[NormalizeTenstorrentTopology] {message}")

    def integer(value):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**31 - 1:
            fail("PipeNet IDs and coordinates must be nonnegative int32 integers")
        return value

    def coord(value):
        if not isinstance(value, list) or len(value) != 2:
            fail("PipeNet coordinate must contain exactly two integers")
        return [integer(v) for v in value]

    try:
        data = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        fail(f"invalid frozen PipeNet JSON: {exc}")
    if not isinstance(data, dict) or set(data) != {"id", "kind", "pipes"}:
        fail("PipeNet descriptor requires id, kind and pipes")
    net = integer(data["id"])
    if data["kind"] not in ("point_to_point", "collective"):
        fail("unknown PipeNet kind")
    collective = data["kind"] == "collective"
    if not isinstance(data["pipes"], list) or not 1 <= len(data["pipes"]) <= 4096:
        fail("PipeNet requires between 1 and 4096 ordered records")
    records = []
    for index, pipe in enumerate(data["pipes"]):
        if not isinstance(pipe, dict) or set(pipe) != {"src", "dst"}:
            fail("Pipe record requires src and dst")
        source = coord(pipe["src"])
        if collective:
            destination = pipe["dst"]
            if not isinstance(destination, dict) or set(destination) != {"begin", "end"}:
                fail("collective destination requires begin and end")
            begin, end = coord(destination["begin"]), coord(destination["end"])
            if any(b >= e for b, e in zip(begin, end)):
                fail("collective destination range must be nonempty")
        else:
            begin = coord(pipe["dst"])
            end = [v + 1 for v in begin]
        records.append([net, index, int(collective), *source, *begin, *end])
    return convert(records)
