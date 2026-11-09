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

// Borrowed canonical layout. Creation checks geometry, strictly ordered unique keys,
// signed lattice bounds and slot arithmetic without allocation. Errors are code-only.
// The source keys must remain unchanged and alive for every query; copying the view
// does not extend that lifetime. An empty default view has unit geometry.
class SparseMacGridView3D {
 public:
  SparseMacGridView3D() = default;
  static absl::StatusOr<SparseMacGridView3D>
  Create(UniformMacGeometry3D geometry, std::span<const BlockIndex3D> active_block_indices);
  // Validated world origin/cell width in meters, slot count, and borrowed canonical keys.
  UniformMacGeometry3D geometry() const;
  std::size_t face_count() const;
  std::span<const BlockIndex3D> active_block_indices() const;
  // Same ownership, units and status categories as the owning layout below.
  absl::StatusOr<std::size_t> FaceSlot(FaceIndex3D face_index) const;
  // Inverse owned-slot mapping; OutOfRange when face_slot >= face_count().
  absl::StatusOr<FaceIndex3D> FaceIndex(std::size_t face_slot) const;
  absl::StatusOr<Vec3> FaceWorldAnchor(FaceIndex3D face_index) const;
  absl::StatusOr<Vec3> CellWorldCenter(CellIndex3D cell_index) const;

 private:
  friend class SparseMacGrid3D;
  SparseMacGridView3D(UniformMacGeometry3D geometry, std::span<const BlockIndex3D> active_block_indices);
  UniformMacGeometry3D geometry_{{0.0, 0.0, 0.0}, 1.0};
  std::span<const BlockIndex3D> active_block_indices_;
};

// Mathematical floor ownership on the signed lattice; no allocation or failure.
BlockIndex3D MacOwnerBlock3D(FaceLatticeIndex3D lattice_index);

struct MacFaceGeometrySample3D {
  FaceIndex3D face_index;  // Global signed staggered lattice identity, independent of activation.
  double weight;  // Dimensionless quadratic tensor product, including exact zero edges.
  Vec3 face_offset_from_particle;  // Face anchor minus trial particle position, meters.
  Vec3 weight_gradient;  // Derivative with respect to trial position, inverse meters.
};
using MacFaceGeometryStencil3D = std::array<MacFaceGeometrySample3D, 27>;

// Computes the exact same checked quadratic geometry as BuildOwnedMacFaceStencil3D
// before consulting ownership. Includes all zero-weight nodes. Geometry must be valid;
// invalid geometry/input/axis is InvalidArgument, motion/tile failure FailedPrecondition,
// and unrepresentable indices/arithmetic OutOfRange. No partial output or allocation.
absl::StatusOr<MacFaceGeometryStencil3D> BuildMacFaceGeometryStencil3D(
    UniformMacGeometry3D geometry,
    FaceNormalAxis axis,
    Vec3 sorted_particle_world_position,
    Vec3 trial_particle_world_position
);

// Owns the uniform staggered layout. Each active block has 64 cells and 64
// low faces per axis. A cell's high face belongs to the adjacent cell's owner.
class SparseMacGrid3D {
 public:
  // Copies and sorts active block indices in (z,y,x) order. Returns kInvalidArgument
  // for invalid geometry or duplicate keys, kOutOfRange for face counts or
  // keys whose faces leave the signed int32 lattice, and kResourceExhausted
  // if keys exceed block_capacity.
  // An empty layout is valid with zero capacity. All errors have code-only empty messages.
  static absl::StatusOr<SparseMacGrid3D>
  Create(UniformMacGeometry3D geometry, std::span<const BlockIndex3D> active_block_indices, std::size_t block_capacity);

  // Borrows canonical keys until this owner is moved, assigned or destroyed.
  SparseMacGridView3D view() const;

  // Returns the validated world origin and uniform cell width in meters.
  UniformMacGeometry3D geometry() const;

  // Returns the total number of owned face slots across all active blocks.
  std::size_t face_count() const;

  // Views the active block indices in (z,y,x) order. Obtain a new view after moving or
  // assigning the layout. The view must not outlive the layout.
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

// Borrowed overload with identical numerics and failure categories; no allocation,
// including error construction/destruction. The view's keys must remain alive.
absl::StatusOr<MacFaceStencil3D> BuildOwnedMacFaceStencil3D(
    SparseMacGridView3D mac_grid,
    FaceNormalAxis axis,
    Vec3 sorted_particle_world_position,
    Vec3 trial_particle_world_position
);

}  // namespace fluidity
