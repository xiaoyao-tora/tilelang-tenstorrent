"""Tenstorrent language dialect: common TileLang plus communication primitives."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.loop import Tiles as Tiles

from . import comm as comm
from .copy import copy as copy

__tilelang_dialect__ = "tenstorrent"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, "Tiles", "comm")))

del _COMMON_ALL
