# Frontend intrinsic registrations do not require a device SDK or runtime.
file(GLOB TILE_LANG_TENSTORRENT_SRCS
  src/tenstorrent/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_TENSTORRENT_SRCS})
