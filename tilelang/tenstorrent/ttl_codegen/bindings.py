"""Lazy, revision-checked access to the optional TT-Lang compiler bindings."""

from dataclasses import dataclass
import importlib

from ..contracts import TTLANG_REFERENCE_COMMIT

# API inspected against the production frontend, ODS definitions and C++ bindings.
TTLANG_REVISION = TTLANG_REFERENCE_COMMIT


@dataclass(frozen=True)
class Bindings:
    ir: object
    ttl: object
    ttcore: object
    ttkernel: object
    func: object
    arith: object


def load_bindings() -> Bindings:
    try:
        package = importlib.import_module("ttl")
        modules = [
            importlib.import_module(name)
            for name in (
                "ttl.ir",
                "ttl.dialects.ttl",
                "ttl.dialects.ttcore",
                "ttl.dialects.ttkernel",
                "ttl.dialects.func",
                "ttl.dialects.arith",
            )
        ]
    except (ImportError, OSError) as error:
        raise ImportError(
            "Tenstorrent TTL codegen requires the TT-Lang compiler Python bindings "
            f"built from revision {TTLANG_REVISION}. Install the compiler wheel or "
            "activate its build environment; the simulator-only package is insufficient. "
            f"Original import error: {error}"
        ) from error
    revision = package.build_info().get("ttlang", "unknown")
    if revision != TTLANG_REVISION:
        raise RuntimeError(
            f"TT-Lang compiler revision mismatch: expected {TTLANG_REVISION}, got {revision!r}. "
            "Rebuild the pinned compiler; do not bypass the revision check for an unverified API."
        )
    return Bindings(*modules)
