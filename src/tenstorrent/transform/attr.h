#ifndef TVM_TL_TENSTORRENT_TRANSFORM_ATTR_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_ATTR_H_

namespace tvm::tl::tenstorrent {
constexpr const char *kTTTilesScope = "tl.tt.tiles_scope";
constexpr const char *kTTTilesDomain = "tl.tt.tiles_domain";
constexpr const char *kTTTilesParallel = "tl.tt.tiles_parallel";
constexpr const char *kTTTilesStage = "tl.tt.tiles_stage";
constexpr const char *kTTComputeKind = "tl.tt.compute_kind";
constexpr const char *kTTLogicalDomain = "tl.tt.logical_domain";
constexpr const char *kTTPhysicalTileShape = "tl.tt.physical_tile_shape";
constexpr const char *kTTBlockShape = "tl.tt.block_shape";
constexpr const char *kTTIteratorTypes = "tl.tt.iterator_types";
constexpr const char *kTTAccessMaps = "tl.tt.access_maps";
constexpr const char *kTTBroadcastRecipes = "tl.tt.broadcast_recipes";
enum class TTTilesStage : int {
  kFrontend = 0,
  kStructured = 1,
  kPartitioned = 2,
  kLowered = 3,
};
} // namespace tvm::tl::tenstorrent
#endif // TVM_TL_TENSTORRENT_TRANSFORM_ATTR_H_
