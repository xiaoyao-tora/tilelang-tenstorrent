# Tenstorrent target and frontend intrinsic registrations are hardware-independent.
file(GLOB TILE_LANG_TENSTORRENT_SRCS
  src/tenstorrent/op/*.cc
)
list(APPEND TILE_LANG_TENSTORRENT_SRCS
  src/tenstorrent/target.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_TENSTORRENT_SRCS})

# USE_TENSTORRENT is reserved for future TT-Lang/TTNN-dependent native sources.
# TargetKind and intrinsic registration above must remain available when it is OFF.
