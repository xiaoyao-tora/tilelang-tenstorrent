"""Initial TTL emission. Importing this package does not load TT-Lang."""

from .artifact import TensorABI, TenstorrentTTLModuleNode
from .bindings import TTLANG_REVISION

__all__ = ["TTLANG_REVISION", "TensorABI", "TenstorrentTTLModuleNode"]
