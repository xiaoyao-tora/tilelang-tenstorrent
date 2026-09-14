"""FFI APIs for Tenstorrent-owned compiler transformations."""

import tvm_ffi

tvm_ffi.init_ffi_api("tl.tenstorrent.transform", __name__)
