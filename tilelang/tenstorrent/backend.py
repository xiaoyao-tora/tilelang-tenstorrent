"""Tenstorrent backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline


BACKEND = register_backend(
    BackendModule(
        name="tenstorrent",
        target_kinds=("tenstorrent",),
        pipelines={"tenstorrent": pipeline.TENSTORRENT_PIPELINE},
        device_codegens={
            "tenstorrent": DeviceCodegen(
                "tenstorrent",
                build_without_compile=codegen.build_ttl_without_compile,
            )
        },
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
