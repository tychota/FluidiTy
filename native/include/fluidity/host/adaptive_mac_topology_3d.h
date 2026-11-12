#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

// The signed lower corner uses the common finest cell lattice. A leaf spans
// 2^width_exponent finest cells along each axis and is aligned to that width.
struct AdaptiveMacCellKey3D {
  CellIndex3D lower_corner;
  std::uint8_t width_exponent;
};

struct AdaptiveMacLeaf3D {
  AdaptiveMacCellKey3D key;
  double cell_width;  // m.
  double cell_volume;  // m³.
};

// One face shared by two leaves. The lower corner lies on the face plane
// and names the low corner in both tangential axes. Its width exponent is
// the smaller of the two incident cell widths' exponents.
struct AdaptiveMacInteriorFace3D {
  FaceIndex3D face_index;
  std::uint8_t width_exponent;
  std::size_t tail_cell_ordinal;  // Low side along the positive normal.
  std::size_t head_cell_ordinal;  // High side along the positive normal.
  double face_area;  // m².
  double cell_center_distance;  // Normal spacing in m.
};

class AdaptiveMacTopology3D {
 public:
  // Build from 1..4096 dyadically aligned, disjoint leaves and a finite
  // positive finest width in m. Touching leaves, including edge and corner
  // contacts, may differ by at most one exponent. Width exponents are 0..30.
  // Input order does not affect cell or face ordinals. Exterior faces are
  // omitted. InvalidArgument denotes invalid geometry or overlapping leaves,
  // FailedPrecondition denotes a grading violation, OutOfRange denotes
  // unrepresentable indices or measures, and ResourceExhausted denotes
  // capacity overflow. Failure returns no topology.
  static absl::StatusOr<AdaptiveMacTopology3D>
  Create(std::span<const AdaptiveMacCellKey3D> leaf_keys, double finest_cell_width);

  // Return the minimal canonical set of committed parent ordinals to split
  // once so the resulting leaves stay 2:1 across faces, edges and corners.
  // Requests may be in any order or repeated. The proposed leaf count must
  // fit maximum_candidate_cells and the topology limit. Invalid ordinals
  // return InvalidArgument, a finest-level request returns FailedPrecondition
  // and excess capacity returns ResourceExhausted. The topology stays frozen.
  absl::StatusOr<std::vector<std::size_t>>
  RefinementClosure(std::span<const std::size_t> requested_parents, std::size_t maximum_candidate_cells) const;

  // Views borrow this object. Cells are ordered by signed (z,y,x) lower
  // corner. Face slots are the order (normal axis,z,y,x,width exponent).
  // Width of the common finest lattice in meters.
  double finest_cell_width() const {
    return finest_cell_width_;
  }
  // Bytes reserved by the owned leaf and interior face arrays.
  std::size_t reserved_bytes() const {
    return cells_.capacity() * sizeof(AdaptiveMacLeaf3D)
        + interior_faces_.capacity() * sizeof(AdaptiveMacInteriorFace3D);
  }
  std::span<const AdaptiveMacLeaf3D> cells() const;
  std::span<const AdaptiveMacInteriorFace3D> interior_faces() const;

 private:
  AdaptiveMacTopology3D() = default;
  double finest_cell_width_ = 0;
  std::vector<AdaptiveMacLeaf3D> cells_;
  std::vector<AdaptiveMacInteriorFace3D> interior_faces_;
};

// One uncovered geometric subface. The owned leaf is on the low side of
// the positive normal when cell_is_tail is true. face_area is in m².
struct AdaptiveMacExteriorFace3D {
  FaceIndex3D face_index;
  std::uint8_t width_exponent;
  std::size_t cell_ordinal;
  bool cell_is_tail;
  double face_area;
  std::size_t face_slot;
};

struct AdaptiveMacFaceOwnership3D {
  // Slots parallel AdaptiveMacTopology3D::interior_faces().
  std::vector<std::size_t> interior_face_slots;
  // Exterior subfaces in increasing global slot order.
  std::vector<AdaptiveMacExteriorFace3D> exterior_faces;
};

// Complete each uncovered leaf side using a full face or the unoccupied
// quarters of a partially covered coarse face. Assign both classes of faces
// global slots ordered by (normal axis,z,y,x,width exponent). The input
// topology supplies all interior contacts. Geometric area does not include
// physical aperture clipping. A moved-from empty topology is invalid.
absl::StatusOr<AdaptiveMacFaceOwnership3D> BuildAdaptiveMacFaceOwnership3D(const AdaptiveMacTopology3D& topology);

}  // namespace fluidity
