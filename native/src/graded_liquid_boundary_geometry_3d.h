#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "mac_pressure_3d_system.h"

namespace fluidity {

struct LiquidAdaptiveCellState3D;
struct GradedLiquidCornerSample3D;

struct GradedLiquidPlanarSurface3D {
  Vec3 unit_normal;  // Outward from liquid; liquid is n dot x < plane_coordinate_m.
  double plane_coordinate_m;  // World meters at the freeze time.
};

struct FrozenGradedLiquidLeaf3D {
  Vec3 world_lower_m;
  Vec3 world_upper_m;
  double fluid_volume_m3;  // Intersection of liquid and nonsolid.
  bool known_air;  // Entire leaf is strictly nonsolid and on the dry side of the liquid plane.
};

struct FrozenGradedLiquidSubface3D {
  FaceIndex3D face_index;
  std::uint8_t width_exponent;
  bool exterior;
  Vec3 world_lower_m;
  Vec3 world_upper_m;  // Same coordinate as lower along the normal.
  double full_area_m2;
  double wet_open_area_m2;  // Liquid intersect nonsolid; excludes dry nonsolid area.
  std::optional<Vec3> wet_open_centroid_m;  // World-space area centroid, absent for zero wet-open area.
  double wet_solid_area_m2;  // Liquid intersect solid; excludes dry solid area.
  std::optional<Vec3> wet_solid_centroid_m;  // World-space area centroid, absent for zero wet-solid area.
  double wall_normal_flux_m3_per_s;  // Wet solid area times wall speed along the positive MAC axis.
};

class FrozenGradedPlanarLiquidGeometry3D {
 public:
  // Views borrow this owner and are in canonical leaf and global subface order.
  std::span<const FrozenGradedLiquidLeaf3D> leaves() const {
    return leaves_;
  }
  std::span<const FrozenGradedLiquidSubface3D> subfaces() const {
    return subfaces_;
  }
  const AdaptiveMacTopology3D& topology() const {
    return topology_;
  }
  // World-space lower corner of finest lattice cell zero, in meters.
  Vec3 world_origin_m() const {
    return world_origin_m_;
  }
  double step_duration_s() const {
    return step_duration_s_;
  }
  // Reserved dynamic bytes owned by this frozen layout and its measured records.
  std::size_t reserved_bytes() const {
    return topology_.reserved_bytes() + leaves_.capacity() * sizeof(FrozenGradedLiquidLeaf3D)
        + subfaces_.capacity() * sizeof(FrozenGradedLiquidSubface3D);
  }

 private:
  friend absl::StatusOr<FrozenGradedPlanarLiquidGeometry3D> FreezeGradedPlanarLiquidGeometry3D(
      const AdaptiveMacTopology3D&,
      Vec3,
      const TranslatingHalfspace3D&,
      const GradedLiquidPlanarSurface3D&,
      double
  );

  friend absl::StatusOr<FrozenGradedPlanarLiquidGeometry3D>
  FreezeGradedSampledLiquidGeometry3D(const AdaptiveMacTopology3D&, Vec3, double, std::span<const GradedLiquidCornerSample3D>, std::span<const std::uint8_t>);

  FrozenGradedPlanarLiquidGeometry3D(const AdaptiveMacTopology3D& topology, Vec3 world_origin_m, double step_duration_s)
      : topology_(topology), world_origin_m_(world_origin_m), step_duration_s_(step_duration_s) {
  }

  AdaptiveMacTopology3D topology_;
  Vec3 world_origin_m_;
  double step_duration_s_;
  std::vector<FrozenGradedLiquidLeaf3D> leaves_;
  std::vector<FrozenGradedLiquidSubface3D> subfaces_;
};

// Freeze the halfspace and liquid plane at the start of the interval. The
// origin is the world lower corner of finest lattice cell zero. Canonical
// endpoints define the physical axis-aligned bounds. Clip the two plane
// fields at these corners with the shared tetrahedra and face diagonals.
// A face on the solid plane is blocked; a face on the liquid plane is dry.
// The owner includes dry and solid leaves and every canonical global subface.
// Invalid planes, origin or duration return InvalidArgument. Unresolved
// world bounds and unrepresentable measures return OutOfRange. Topology and
// clipping errors propagate. Inputs are read-only, with no result on failure.
absl::StatusOr<FrozenGradedPlanarLiquidGeometry3D> FreezeGradedPlanarLiquidGeometry3D(
    const AdaptiveMacTopology3D& topology,
    Vec3 world_origin_m,
    const TranslatingHalfspace3D& solid,
    const GradedLiquidPlanarSurface3D& liquid_surface,
    double step_duration_s
);

struct GradedLiquidPressureCellFields3D {
  double target_divergence_s_inv = 0;  // External source per fluid volume, independent of wall flux.
  double bulk_modulus_pa = std::numeric_limits<double>::infinity();
  double previous_pressure_pa = 0;
};

struct GradedLiquidPressureExterior3D {
  double pressure_pa;  // Gauge pressure including any caller-supplied lift.
  double distance_m;  // Positive distance from the incident pressure cell.
};

struct GradedLiquidPressureSubfaceFields3D {
  double inverse_density_m3_per_kg = 0;
  double predicted_normal_velocity_m_per_s = 0;
  double pressure_jump_pa = 0;  // Equilibrium head minus tail, interior only.
  // Absent on interior faces. On exterior faces, absence selects sealing
  // of the wet aperture while retaining the measured prescribed wall flux.
  std::optional<GradedLiquidPressureExterior3D> exterior_pressure = std::nullopt;
};

struct GradedLiquidPressureInputs3D {
  std::vector<MacPressureCell3D> cells;
  std::vector<pressure_internal::AdaptiveMacPressureSubface3D> subfaces;
};

enum class GradedLiquidPressureDomain3D {
  kAllWet,
  kKnownAir
};

// Compose canonical pressure input rows without solving. This lets a scheduler
// freeze aperture and wall terms before adding its trial predictor velocities.
// kKnownAir allows geometrically certified zero-volume leaves in the staging
// arrays. Pass them to the sparse graph preparer for occupancy admission.
absl::StatusOr<GradedLiquidPressureInputs3D> ComposeGradedPlanarLiquidPressureInputs3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const GradedLiquidPressureCellFields3D> cell_fields,
    std::span<const GradedLiquidPressureSubfaceFields3D> subface_fields,
    GradedLiquidPressureDomain3D domain = GradedLiquidPressureDomain3D::kAllWet
);

// Compose explicit pressure fields with frozen physical measures and canonical
// tail/head incidence. Fields cover every frozen leaf and global face slot.
// A zero-volume leaf returns FailedPrecondition. World extents incompatible
// with the nominal pressure metrics return OutOfRange. Tiny upper-measure
// roundoff is capped at the nominal metric. Malformed extents return
// InvalidArgument and existing graph errors propagate. The graph does not
// infer an embedded free-surface pressure or a separate interior wall term.
// The frozen geometry and inputs are unchanged, with no graph on failure.
absl::StatusOr<pressure_internal::PreparedMacPressureSystem3D> PrepareGradedPlanarLiquidPressure3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const GradedLiquidPressureCellFields3D> cell_fields,
    std::span<const GradedLiquidPressureSubfaceFields3D> subface_fields
);

struct GradedWetPressureSubset3D {
  pressure_internal::PreparedMacPressureSystem3D system;
  // Maps compact pressure row ordinals to canonical accepted leaf ordinals.
  std::vector<std::size_t> leaf_ordinals;
};

// Prepare a positive-volume pressure subset on the fixed accepted leaf roster.
// Supply validated accepted leaf records whose inventory accounts for all
// resident material. Field spans retain canonical leaf and global subface
// order, and returned links retain global face slots. A dry leaf must already
// be accepted empty air and be geometrically certified. Wetting, drying,
// unsupported occupancy, an open wet/air interface, and nonempty fields for
// omitted rows or faces return FailedPrecondition. Inconsistent extents or
// keys return InvalidArgument.
// Geometry and graph errors propagate. Inputs remain unchanged on failure.
absl::StatusOr<GradedWetPressureSubset3D> PrepareGradedWetPressureSubset3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const LiquidAdaptiveCellState3D> accepted_leaves,
    std::span<const GradedLiquidPressureCellFields3D> cell_fields,
    std::span<const GradedLiquidPressureSubfaceFields3D> subface_fields
);

// Prepare the same subset from canonical inputs produced by Compose for this
// geometry. The caller may stage predictor velocities and warm-start pressure
// on the wet domain before assembly. The sparse admission and graph checks
// apply to the completed fields, with no change to the borrowed inputs.
absl::StatusOr<GradedWetPressureSubset3D> PrepareGradedWetPressureSubsetFromInputs3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const LiquidAdaptiveCellState3D> accepted_leaves,
    std::span<const MacPressureCell3D> cell_inputs,
    std::span<const pressure_internal::AdaptiveMacPressureSubface3D> subface_inputs
);

}  // namespace fluidity
