/*!
 * \file tenstorrent/target.cc
 * \brief Tenstorrent target kind registration and validation.
 */

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>

namespace tvm {
namespace {

ffi::Map<ffi::String, ffi::Any>
CanonicalizeTenstorrentTarget(ffi::Map<ffi::String, ffi::Any> target) {
  if (!target.count("arch")) {
    TVM_FFI_THROW(ValueError) << "Tenstorrent target requires an explicit "
                                 "'arch'. Supported architectures: "
                                 "wormhole_b0, blackhole.";
  }

  ffi::String arch = Downcast<ffi::String>(target.at("arch"));
  if (arch != "wormhole_b0" && arch != "blackhole") {
    TVM_FFI_THROW(ValueError)
        << "Unsupported Tenstorrent architecture '" << arch
        << "'. Supported architectures: wormhole_b0, blackhole.";
  }

  if (target.count("keys")) {
    ffi::Array<ffi::String> keys =
        Downcast<ffi::Array<ffi::String>>(target.at("keys"));
    if (keys.size() != 1 || keys[0] != "tenstorrent") {
      TVM_FFI_THROW(ValueError) << "Tenstorrent target keys must be exactly "
                                   "['tenstorrent']; the 'gpu' key is "
                                   "not supported.";
    }
  }

  target.Set("keys", ffi::Array<ffi::String>({"tenstorrent"}));
  return target;
}

} // namespace

TVM_REGISTER_TARGET_KIND("tenstorrent", kDLExtDev)
    .add_attr_option<ffi::String>("arch")
    .set_default_keys({"tenstorrent"})
    .set_target_canonicalizer(CanonicalizeTenstorrentTarget);

} // namespace tvm
