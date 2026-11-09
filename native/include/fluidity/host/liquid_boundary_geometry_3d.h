#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "fluidity/base/liquid_boundary_math.h"
#include "fluidity/host/mac_pressure_3d.h"

namespace fluidity {

// Signed lattice corner coordinates, without the cell-center half shift.
struct CellCornerIndex3D {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};
struct LiquidCenterDistance3D {
  CellIndex3D cell_index;
  double signed_distance;  // Physical liquid distance in meters, negative inside.
};
struct LiquidCornerDistance3D {
  CellCornerIndex3D corner_index;
  double signed_distance;  // Same physical field and version as center samples.
};
struct TranslatingHalfspace3D {
  Vec3 unit_normal;  // Outward from solid; solid is n dot x <= start_plane_coordinate.
  double start_plane_coordinate;  // World meters at substep start.
  Vec3 wall_velocity;  // Constant world velocity in m/s.
};
// Solid is |unit_normal dot world_position - start_center_coordinate| <= half_width.
struct TranslatingSlab3D {
  Vec3 unit_normal;  // Unit slab axis; both plane normals follow from this equation.
  double start_center_coordinate;  // World meters at substep start.
  double half_width;  // Positive finite thickness / 2 in meters.
  Vec3 wall_velocity;  // Constant world velocity in m/s.
};
using LiquidObstacle3D =
    std::variant<std::monostate, TranslatingHalfspace3D, TranslatingSlab3D, AxisAlignedLiquidTank3D>;

struct LiquidBoundaryGeometryInputs3D {
  const SparseMacGrid3D& mac_grid;  // Borrowed only during freeze; result owns a copy.
  // Candidate cells, distinct and sorted (z,y,x), separate from the sample halo.
  std::span<const CellIndex3D> cell_indices;
  // Distinct (z,y,x) ordered samples. Include corners of all candidates and centers
  // of every air neighbor; positive-sigma open air links need the centered gradient/Hessian halo
  // at their adjacent liquid centers (axis neighbors and mixed-derivative diagonals).
  std::span<const LiquidCenterDistance3D> center_distances;  // Candidate centers plus the documented neighbor halo.
  std::span<const LiquidCornerDistance3D> corner_distances;  // All eight corners of each candidate.
  LiquidObstacle3D obstacle;
  double step_duration;  // Positive finite seconds; geometry freezes at start.
  double liquid_density;  // Positive finite kg/m³.
  double surface_tension;  // Finite nonnegative N/m.
};
struct FrozenLiquidCell3D {
  CellIndex3D cell_index;
  double fluid_volume;  // Liquid intersect nonsolid, m³.
  double center_distance;  // Physical signed distance, m.
  double curvature;  // Centered outward curvature, 1/m; zero unless an open air link needs positive-sigma curvature.
};
struct FrozenLiquidFace3D {
  FaceIndex3D face_index;
  std::size_t face_slot;
  double open_area;  // Full face minus solid, m²; not clipped by liquid.
  double blocked_area;  // Exactly h² - open_area, m².
  double wall_normal_velocity;  // Positive Cartesian face axis, m/s, on both low and high faces.
  // Positive Cartesian orientation, m³/s. For tanks this is the piston-only
  // geometric flux on this pressure face, without liquid-area clipping. Other
  // obstacles use blocked_area * wall_normal_velocity.
  std::optional<double> prescribed_wall_flux = std::nullopt;
  // Freeze identifies a closed low-x piston rectangle independently of its speed.
  bool is_low_x_piston_face = false;
};
struct LiquidPressureCellFields3D {
  double target_divergence;  // s⁻¹; independent of geometric volume.
  double bulk_modulus;  // Positive Pa, or +infinity for incompressibility.
  double previous_pressure;  // Pa.
};
struct LiquidPressureFields3D {
  std::span<const LiquidPressureCellFields3D> cells;  // Frozen cell order.
  std::span<const double> predicted_face_velocities;  // Frozen face order, m/s.
  double ambient_pressure;  // Finite gauge pressure, Pa.
  // Empty preserves frozen wall flux. Otherwise use exact frozen face order,
  // with a finite signed flux in m³/s on each closed piston face with a single
  // selected head. Zero is explicit. Other entries must be absent.
  std::span<const std::optional<double>> wetted_piston_face_fluxes{};
};
struct LiquidPressureProblem3D {
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  std::size_t lower_theta_regularization_count = 0;  // Number of air links clamped upward to 0.05h.
};

class FrozenLiquidBoundaryGeometry3D;
namespace boundary_internal {
struct GeometryBackend;
absl::StatusOr<FrozenLiquidBoundaryGeometry3D> FreezeGeometry(const LiquidBoundaryGeometryInputs3D&, GeometryBackend);
absl::StatusOr<LiquidPressureProblem3D>
BuildLinks(const FrozenLiquidBoundaryGeometry3D&, const LiquidPressureFields3D&, GeometryBackend);
}  // namespace boundary_internal

// Owns validated start-time geometry and samples. Views remain valid until move
// or destruction. One face per owner, in slot order; cells use (z,y,x) order.
class FrozenLiquidBoundaryGeometry3D {
 public:
  // Returns the owned grid, including its unique face-slot layout.
  const SparseMacGrid3D& mac_grid() const {
    return mac_grid_;
  }
  // Returns selected pressure cells in (z,y,x) order.
  std::span<const FrozenLiquidCell3D> cells() const {
    return cells_;
  }
  // Returns the exact union of incident faces in owned slot order.
  std::span<const FrozenLiquidFace3D> faces() const {
    return faces_;
  }
  // Returns the frozen positive step duration in seconds.
  double step_duration() const {
    return step_duration_;
  }
  // Seconds; +infinity means no active positive-sigma air link and no capillary restriction.
  double capillary_step_bound() const {
    return capillary_step_bound_;
  }

 private:
  explicit FrozenLiquidBoundaryGeometry3D(const SparseMacGrid3D& grid) : mac_grid_(grid) {
  }
  friend absl::StatusOr<FrozenLiquidBoundaryGeometry3D>
  FreezeLiquidBoundaryGeometry3D(const LiquidBoundaryGeometryInputs3D&);
  friend absl::StatusOr<LiquidPressureProblem3D>
  BuildLiquidPressureLinks3D(const FrozenLiquidBoundaryGeometry3D&, const LiquidPressureFields3D&);
  friend absl::StatusOr<FrozenLiquidBoundaryGeometry3D>
  boundary_internal::FreezeGeometry(const LiquidBoundaryGeometryInputs3D&, boundary_internal::GeometryBackend);
  friend absl::StatusOr<LiquidPressureProblem3D> boundary_internal::BuildLinks(
      const FrozenLiquidBoundaryGeometry3D&,
      const LiquidPressureFields3D&,
      boundary_internal::GeometryBackend
  );
  SparseMacGrid3D mac_grid_;
  std::vector<LiquidCenterDistance3D> center_distances_;
  std::vector<FrozenLiquidCell3D> cells_;
  std::vector<FrozenLiquidFace3D> faces_;
  double step_duration_ = 0.0;
  double liquid_density_ = 0.0;
  double surface_tension_ = 0.0;
  double capillary_step_bound_ = 0.0;
};

// Clips fixed matching face triangles and six body-diagonal tetrahedra at the
// start pose. Selects strictly negative centers with positive fluid volume.
// Returns InvalidArgument for malformed fields/order/obstacle, FailedPrecondition
// for missing samples/owners, disconnected slab liquid, degenerate interface
// gradient (|g| < 1e-8), excessive capillary dt, or a tank closing within the step,
// ResourceExhausted above 4096 candidates/24576 faces, OutOfRange for arithmetic
// or lattice overflow and tank cells whose world corners cannot resolve the cell width.
// Errors carry no partial geometry; inputs are unmodified.
absl::StatusOr<FrozenLiquidBoundaryGeometry3D> FreezeLiquidBoundaryGeometry3D(
    const LiquidBoundaryGeometryInputs3D& inputs
);

// Builds the exact incident face union with full frozen apertures and wall flux.
// Air links use theta in [0.05,1] and ambient + sigma*curvature once; internal
// jumps are zero. Curvature and capillary dt were checked during geometry freeze.
// Errors carry no problem: InvalidArgument for invalid field extents/values or
// overrides inconsistent with the frozen piston incidence or speed,
// FailedPrecondition for missing air samples or omitted negative-distance neighbors,
// OutOfRange for unrepresentable arithmetic. No input storage is modified.
absl::StatusOr<LiquidPressureProblem3D>
BuildLiquidPressureLinks3D(const FrozenLiquidBoundaryGeometry3D& geometry, const LiquidPressureFields3D& fields);

}  // namespace fluidity
