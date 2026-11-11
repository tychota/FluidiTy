#pragma once

#include <filesystem>
#include <span>

#include "absl/status/status.h"

namespace fluidity::blender {
// Encode a bounded Blender input packet as canonical M1 settings and frame-zero
// state, with at most 1024 seeded particles. Version one preserves the original
// translating paddle defaults. Version two carries density (kg/m^3), surface
// tension (N/m), friction, energy blend, step duration (s), pressure budget,
// initial velocity (m/s), and one free-space/plane/slab/open-tank collision model.
// Rejects malformed, nonfinite or unsupported inputs with kInvalidArgument.
// Paths are explicit and outputs must be fresh. The pair belongs to a private
// caller-owned staging directory and must be discarded on any failure.
absl::Status ResolveBlenderScene(std::span<const std::filesystem::path> args);
}  // namespace fluidity::blender
