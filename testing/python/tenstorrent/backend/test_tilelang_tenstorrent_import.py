from __future__ import annotations

import subprocess
import sys


def test_tilelang_import_does_not_load_tenstorrent_dependencies():
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'ttl', 'ttnn'}:
        raise AssertionError(f'unexpected optional Tenstorrent import: {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import tilelang
from tilelang.backend import get_backend

assert get_backend('tenstorrent').target_kinds == ('tenstorrent',)
"""

    subprocess.run([sys.executable, "-c", script], check=True)
