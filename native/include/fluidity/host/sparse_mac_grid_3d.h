#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

// All 27 samples for one face normal. Each sample links a quadratic basis
// weight and its derivative to the unique face slot that owns the value.
using MacFaceStencil3D = std::array<MacFaceStencilSample3D, 27>;

// Owns the uniform staggered layout. Each active block has 64 cells and 64
// low faces per axis. A cell's high face belongs to the adjacent cell's owner.
class SparseMacGrid3D {
 public:
  // Copies and sorts active block indices in (z,y,x) order. Returns kInvalidArgument
  // for invalid geometry or duplicate keys, kOutOfRange for face counts or
  // keys whose faces leave the signed int32 lattice, and kResourceExhausted
  // if keys exceed block_capacity.
  // An empty layout is valid with zero capacity.
  static absl::StatusOr<SparseMacGrid3D>
  Create(UniformMacGeometry3D geometry, std::span<const BlockIndex3D> active_block_indices, std::size_t block_capacity);

  // Returns the validated world origin and uniform cell width in meters.
  UniformMacGeometry3D geometry() const;

  // Returns the total number of owned face slots across all active blocks.
  std::size_t face_count() const;

  // Views the active keys in (z,y,x) order for the lifetime of this layout.
  std::span<const BlockIndex3D> active_block_indices() const;

  // Returns the owned slot, with blocks in (z,y,x) order, then x/y/z normals,
  // then local (z,y,x) order. A cell's high face uses the next lattice index
  // along its normal. Returns kInvalidArgument for an invalid axis and
  // kFailedPrecondition if the face's owner is inactive.
  absl::StatusOr<std::size_t> FaceSlot(FaceIndex3D face_index) const;

  // Returns the face anchor in world meters, with a half-cell shift on each
  // transverse axis. Allocation is immaterial to this geometry query. Returns
  // kInvalidArgument for an invalid axis or kOutOfRange for a nonfinite result.
  absl::StatusOr<Vec3> FaceWorldAnchor(FaceIndex3D face_index) const;

  // Returns the cell center in world meters, regardless of allocation.
  // Returns kOutOfRange for a nonfinite result.
  absl::StatusOr<Vec3> CellWorldCenter(CellIndex3D cell_index) const;

 private:
  SparseMacGrid3D(UniformMacGeometry3D geometry, std::vector<BlockIndex3D> active_block_indices);

  UniformMacGeometry3D geometry_;
  std::vector<BlockIndex3D> active_block_indices_;
};

// Computes floor((X - 1.5)/4) per axis, with X the sorted world position in
// cell widths relative to the origin. The result is independent of allocated
// blocks. Returns kInvalidArgument for invalid geometry or nonfinite positions
// and kOutOfRange for nonfinite normalized coordinates or bins outside int32.
absl::StatusOr<ParticleBinIndex3D>
ComputeParticleSortBin3D(UniformMacGeometry3D geometry, Vec3 sorted_particle_world_position);

// Returns all 27 owned quadratic samples in (z,y,x) order, including nodes of
// zero weight. Positions are in world meters. Per-axis trial motion is at most
// half a cell, and every node must be in the sorted particle's tile [4B,4B+8).
// Returns kInvalidArgument for an invalid axis or nonfinite input position,
// kFailedPrecondition for excess motion or unsupported ownership or tile,
// and kOutOfRange for unrepresentable indices or arithmetic. Errors carry no
// partial stencil.
absl::StatusOr<MacFaceStencil3D> BuildOwnedMacFaceStencil3D(
    const SparseMacGrid3D& mac_grid,
    FaceNormalAxis axis,
    Vec3 sorted_particle_world_position,
    Vec3 trial_particle_world_position
);

}  // namespace fluidity
