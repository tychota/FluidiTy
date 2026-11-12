#pragma once

#include <array>
#include <vector>

#include "fluidity/host/motion_interval_3d.h"

namespace fluidity {

struct SurfaceVertex3D {
  std::uint64_t id = 0;  // Nonzero persistent identity, unique and below next_id.
  Vec3 position{};  // World metres.
  Vec3 geometric_velocity{};  // World m/s from actual geometry displacement.
  Vec3 fluid_velocity{};  // Independently sampled carrier velocity, world m/s.
  Vec3 normal{};  // Unit liquid-to-air world normal.
  double age_seconds = 0;
  double wetness = 0;  // Dimensionless fraction in [0,1].
  double thickness_m = 0;
  std::uint64_t component = 0;  // Stable nonzero material/sheet identity at publication.
  std::uint64_t parent_id = 0;  // Repair lineage, zero if no parent.
  bool contact = false;  // Excluded from unconstrained volume offsets.
  bool unresolved_sheet = false;  // Excluded from volume offsets until sheet resolution.
  std::uint64_t phase_id = 0;  // Stable phase when the active composition prefix is nonempty.
  std::uint8_t constituent_count = 0;
  std::array<LiquidConstituentFraction3D, kMaximumLiquidParticleConstituents3D> constituents{};
};
struct SurfaceMesh3D {
  std::vector<SurfaceVertex3D> vertices{};
  std::vector<std::array<std::uint32_t, 3>> triangles{};  // Outward winding from liquid to air.
  std::uint64_t next_id = 1;
};
struct SurfaceLattice3D {
  Vec3 origin{};
  double spacing = 0;
  std::array<std::size_t, 3> extent{};
  std::vector<double> values{};  // Negative liquid, x-major then y then z.
};
enum class SurfacePartitionFace3D : std::uint8_t {
  kPhaseAToAir,
  kPhaseBToAir,
  kPhaseAToPhaseB,
};

struct SurfacePartition3D {
  std::vector<Vec3> positions;  // Shared world-metre vertices for all three face classes.
  std::vector<std::array<std::uint32_t, 3>> triangles;
  std::vector<SurfacePartitionFace3D> face_types;  // One orientation label per triangle.
};

// Partitions the common PL exterior zero set and extracts the phase zero set
// inside it. Negative phase samples label A. The three sheets share seam
// vertex indices. Near-zero lattice nodes use a positive symbolic tie;
// coincident planes on an intersected tetrahedral face fail explicitly.
// The exterior requires a positive boundary shell. Both sample arrays are
// finite and use the lattice's x-major ordering. Invalid geometry or extents
// return InvalidArgument, budget overflow ResourceExhausted, and unresolved
// or unrepresentable intersections FailedPrecondition or OutOfRange.
absl::StatusOr<SurfacePartition3D> ExtractSurfacePartition3D(
    const SurfaceLattice3D& exterior,
    std::span<const double> phase_distance,
    std::size_t maximum_vertices
);
struct SurfaceMetrics3D {
  double signed_volume = 0;  // m³, local-origin tetrahedron sum.
  double area = 0;  // m².
  std::size_t components = 0;
  std::size_t positive_volume_components = 0;  // Outward-oriented closed boundaries.
  std::size_t negative_volume_components = 0;  // Inward-oriented closed boundaries.
  std::int64_t euler_characteristic = 0;
  std::size_t boundary_edges = 0;
  std::size_t nonmanifold_edges = 0;
  std::size_t nonmanifold_vertices = 0;
  std::size_t orientation_defects = 0;
  std::size_t degenerate_triangles = 0;
  std::size_t self_intersections = 0;  // Triangle intersections beyond their shared indexed boundary.
};

// Extracts the PL zero set with one globally consistent six-tetrahedron cube
// subdivision and shared lattice-edge IDs. Requires finite samples, positive
// spacing, matching extents and a positive boundary shell. Nodes within
// 1e-8*spacing of zero are assigned to air with that positive tie offset.
// Crossings within 1e-7 of an edge endpoint share the lattice node's vertex ID;
// resulting repeated-index triangles are omitted and topology remains checked.
// Vertex normals interpolate centred lattice-node gradients when they agree
// with the outward winding, otherwise use area-weighted geometry normals.
// Invalid input returns
// InvalidArgument, an unclosed lattice FailedPrecondition, excessive vertices
// ResourceExhausted and nonfinite geometry OutOfRange. No input is modified.
absl::StatusOr<SurfaceMesh3D> ExtractSurfaceMesh3D(const SurfaceLattice3D& lattice, std::size_t maximum_vertices);

// Measures indexed geometry and performs an AABB-sweep triangle intersection
// check and closed vertex-link check. Invalid indices/coordinates return
// InvalidArgument. Topology defects are measurements, not suppressed failures.
absl::StatusOr<SurfaceMetrics3D> MeasureSurfaceMesh3D(const SurfaceMesh3D& mesh);
// As above, also records up to 256 intersecting triangle index pairs in sweep
// order. The total count in the returned metrics is not truncated.
absl::StatusOr<SurfaceMetrics3D>
MeasureSurfaceMesh3D(const SurfaceMesh3D& mesh, std::vector<std::array<std::size_t, 2>>* crossings);

// Adjusts mobile vertices along area-weighted normals with a step capped at
// ±0.25 cell_width_m in total from the input, excluding contact and unresolved-sheet vertices. Remeasures
// actual volume, topology and intersections and backtracks unsafe steps. Publishes
// vertices/normals only after the relative volume tolerance is met. Invalid
// controls return InvalidArgument and infeasible constraints FailedPrecondition.
// All failures leave mesh unchanged.
absl::StatusOr<SurfaceMetrics3D> CorrectSurfaceVolume3D(
    SurfaceMesh3D& mesh,
    double target_volume_m3,
    double cell_width_m,
    double relative_tolerance,
    std::size_t maximum_iterations
);

// Uses frozen unit outward normals already stored on the mesh (for example the
// interpolated field normals produced by ExtractSurfaceMesh3D). This avoids
// unrelated area-normal offsets at near-coincident crossing vertices. The same
// total displacement, backtracking, topology, volume and failure-atomicity rules
// as CorrectSurfaceVolume3D apply. Nonunit or inward normals return InvalidArgument.
absl::StatusOr<SurfaceMetrics3D> CorrectSurfaceVolumeWithNormals3D(
    SurfaceMesh3D& mesh,
    double target_volume_m3,
    double cell_width_m,
    double relative_tolerance,
    std::size_t maximum_iterations
);
}  // namespace fluidity
