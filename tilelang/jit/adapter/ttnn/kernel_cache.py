"""TTNN cache policy for in-memory TT-Lang executable artifacts."""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata

from tilelang.cache.kernel_cache import KernelCache


class TTNNKernelCache(KernelCache):
    """Reuse compiled TTNN kernels in process without serializing live objects."""

    @staticmethod
    def _package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "source"

    def _generate_key(self, *args, **kwargs):
        base = super()._generate_key(*args, **kwargs)
        dependencies = (
            self._package_version("tt-lang"),
            self._package_version("ttnn"),
        )
        return sha256(f"{base}|{dependencies!r}".encode()).hexdigest()

    def _load_kernel_from_disk(self, *_args, **_kwargs):
        return None

    def _save_kernel_to_disk(self, *_args, **_kwargs):
        return None


__all__ = ["TTNNKernelCache"]
