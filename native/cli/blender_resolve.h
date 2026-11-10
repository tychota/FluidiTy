#pragma once

#include <filesystem>
#include <span>

#include "absl/status/status.h"

namespace fluidity::blender {
// Encode a bounded Blender input packet as canonical M1 resolved settings and frame-zero
// state. Paths are supplied explicitly; the output paths must be fresh. The pair is
// intended for a private caller-owned staging directory and must be discarded on failure.
absl::Status ResolveBlenderScene(std::span<const std::filesystem::path> args);
}  // namespace fluidity::blender
