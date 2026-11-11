#pragma once

#include "fluidity/host/surface_mesh_3d.h"

namespace fluidity {
struct SurfaceClosestSample3D {
  Vec3 position;  // Closest point on the selected material surface, world metres.
  Vec3 outward_normal;  // Unit normal of the deterministically selected triangle.
  double distance_m;  // Nonnegative Euclidean distance; does not classify phase.
  std::uint64_t component;
  std::uint32_t triangle;
};

// Owning deterministic median-split triangle BVH. Phase/occupancy remains the
// Simulate producer's authority; a geometric nearest query does not invent it.
class SurfaceClosestPoint3D {
 public:
  // Copies safe closed labelled geometry, with nonzero same-component triangle
  // labels. Enforces the supplied triangle budget and a 2^20 hard ceiling.
  // Invalid coordinates/indices/labels return InvalidArgument, unsafe geometry
  // FailedPrecondition, counts ResourceExhausted. Nonzero edge components must
  // be within [1e-50,1e50] metres for representable closest-point arithmetic;
  // violations return OutOfRange. No input aliases survive construction.
  static absl::StatusOr<SurfaceClosestPoint3D> Create(const SurfaceMesh3D& mesh, std::size_t maximum_triangles);
  // Finds only triangles of expected_component. Face/edge/vertex ties use the
  // lowest original triangle index. Finite queries must stay within 1e50 metres
  // of the geometry. Invalid arguments return InvalidArgument, absent component
  // FailedPrecondition, arithmetic OutOfRange. No partial result on failure.
  absl::StatusOr<SurfaceClosestSample3D> Sample(Vec3 world_position, std::uint64_t expected_component) const;

 private:
  struct Face {
    std::array<Vec3, 3> vertices;
    Vec3 normal;
    Vec3 lower;
    Vec3 upper;
    std::uint64_t component;
    std::uint32_t index;
  };
  struct Node {
    Vec3 lower;
    Vec3 upper;
    std::uint32_t left = 0;
    std::uint32_t right = 0;
    std::uint32_t begin = 0;
    std::uint32_t count = 0;
  };
  SurfaceClosestPoint3D() = default;
  std::uint32_t Build(std::uint32_t begin, std::uint32_t end);
  std::vector<Face> faces_;
  std::vector<std::uint32_t> order_;
  std::vector<Node> nodes_;
};
}  // namespace fluidity
