#pragma once

#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/surface_mesh_3d.h"

namespace fluidity {
struct SurfaceVolume3D {
  std::uint64_t component;  // Nonzero material identity, ascending unique order.
  double material_m3;
  double raw_m3;
  double corrected_m3;  // Must match measured geometry, within 0.5% of material.
};
struct SurfaceFrame3D {
  LiquidClock3D clock{};
  double cell_width_m = 0;
  SurfaceMesh3D mesh{};
  std::vector<SurfaceVolume3D> volumes{};
};
struct SurfaceCacheLimits3D {
  std::size_t maximum_vertices = 100000;
  std::size_t maximum_triangles = 200000;
  LiquidCacheLimits3D cache{};
};

// Serializes all persistent Mesh attributes at a completed clock. Checks finite
// fields, unique live IDs below next_id, same-component triangles, closed safe
// geometry and measured per-component volume against declared corrected volume
// and the 0.5% material-volume gate. Invalid records return InvalidArgument,
// configured or hard (2^20) record bounds ResourceExhausted. Empty mesh/volumes
// is valid. The input is borrowed and never changed; failures return no bytes.
absl::StatusOr<LiquidCacheBytes3D> EncodeSurfaceFrame3D(const SurfaceFrame3D& frame, SurfaceCacheLimits3D limits = {});
// Verifies bounded FlatBuffers before allocation, then repeats encoder semantic
// checks. Corrupt or inconsistent content returns DataLoss, exceeded budgets
// ResourceExhausted. Owns decoded arrays; retains no references to input bytes.
absl::StatusOr<SurfaceFrame3D>
DecodeSurfaceFrame3D(std::span<const std::uint8_t> bytes, SurfaceCacheLimits3D limits = {});
}  // namespace fluidity
