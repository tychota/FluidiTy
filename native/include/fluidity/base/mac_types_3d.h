#pragma once

#include <cstddef>
#include <cstdint>

namespace fluidity {

// Three components in Cartesian order.
struct Vec3 {
  double x;
  double y;
  double z;
};

// Row-major matrix with Cartesian rows and columns.
struct Mat3 {
  Vec3 x_row;
  Vec3 y_row;
  Vec3 z_row;
};

// Signed cell lattice indices, with the origin at the lower corner of cell (0,0,0).
struct CellIndex3D {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};

// Signed coordinates of an owned 4x4x4 cell block.
struct BlockIndex3D {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};

// Signed particle bin coordinates at the last sort, distinct from block ownership.
struct ParticleBinIndex3D {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};

// Signed coordinates on one component's staggered face lattice.
struct FaceLatticeIndex3D {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};

enum class FaceNormalAxis : std::uint8_t {
  kX,
  kY,
  kZ,
};

// Axis is the face normal. A cell owns the low face at its own lattice index.
struct FaceIndex3D {
  FaceNormalAxis axis;
  FaceLatticeIndex3D lattice_index;
};

struct UniformMacGeometry3D {
  Vec3 world_origin;  // Lower corner of cell (0,0,0), in meters.
  double cell_width;  // Positive normal spacing in meters.
};

struct MacFaceStencilSample3D {
  std::size_t face_slot;  // Zero-based owned slot in the sparse layout.
  double weight;  // Dimensionless quadratic tensor product.
  Vec3 face_offset_from_particle;  // Face anchor minus trial particle position in meters.
  Vec3 weight_gradient;  // Derivative with respect to the trial position, in inverse meters.
};

}  // namespace fluidity
