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

  // Views borrow this object. Cells are ordered by signed (z,y,x) lower
  // corner. Face slots are the order (normal axis,z,y,x,width exponent).
  std::span<const AdaptiveMacLeaf3D> cells() const;
  std::span<const AdaptiveMacInteriorFace3D> interior_faces() const;

 private:
  AdaptiveMacTopology3D() = default;
  std::vector<AdaptiveMacLeaf3D> cells_;
  std::vector<AdaptiveMacInteriorFace3D> interior_faces_;
};

}  // namespace fluidity
