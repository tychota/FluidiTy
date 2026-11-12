#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/mac_pressure_3d.h"

namespace fluidity {
class AdaptiveMacTopology3D;
}

namespace fluidity::pressure_internal {

struct PreparedMacPressureCell3D {
  CellIndex3D cell_index;
  double fluid_volume;  // m³.
  double integrated_target_flux;  // V times target divergence, m³/s.
  double integrated_inward_wall_flux;  // Interior boundary flux, m³/s.
  double compliance;  // V / (K dt), in (m³/s)/Pa, or zero for incompressibility.
  double previous_pressure;  // Pa.
};

struct FrozenMacPressureLink3D {
  std::size_t face_slot;
  // Missing endpoints are exterior to the pressure domain. Tail is +1 and head is −1 in outward incidence.
  std::optional<std::size_t> tail_cell_ordinal;
  std::optional<std::size_t> head_cell_ordinal;
  double open_area;  // m².
  double blocked_area;  // h² minus open area, m².
  double pressure_to_velocity_scale;  // dt times inverse density divided by distance, in (m/s)/Pa.
  double conductance;  // Open area times pressure-to-velocity scale, in (m³/s)/Pa.
  double predicted_normal_velocity;  // m/s.
  double prescribed_wall_flux;  // Checked positive-normal wall flux, m³/s.
  std::variant<InternalMacPressureLink3D, DirichletMacPressureLink3D, SealedMacPressureLink3D> pressure_link;
};

// Explicitly owned subface between two pressure cells. Cell ordinals
// index the supplied cell span. Normal velocity points from tail to head.
struct ExplicitMacPressureFace3D {
  std::size_t face_slot;
  std::size_t tail_cell_ordinal;
  std::size_t head_cell_ordinal;
  double open_area;  // m².
  double inverse_density;  // m³/kg.
  double pressure_gradient_distance;  // m.
  double predicted_normal_velocity;  // m/s.
  std::optional<double> full_face_area = std::nullopt;  // m². Omission denotes a fully open face.
  double prescribed_wall_flux = 0;  // m³/s along the positive normal.
  double pressure_jump_pa = 0;  // Equilibrium head pressure minus tail pressure on an open link.
};

// An exterior subface names the one incident cell. Missing exterior pressure
// denotes a sealed face. Areas are in m², mobility in m³/kg, distance in m,
// velocity in m/s, and prescribed wall flux in m³/s along the positive normal.
struct ExplicitMacPressureBoundaryFace3D {
  std::size_t face_slot;
  std::size_t cell_ordinal;
  bool cell_is_tail;
  double full_face_area;
  double open_area;
  double inverse_density;
  double pressure_gradient_distance;
  double predicted_normal_velocity;
  double prescribed_wall_flux;
  std::optional<double> exterior_pressure;  // Pa.
};

// Owns validated coefficients and topology independently of the original input
// storage. The following views borrow from this object and follow input order.
class PreparedMacPressureSystem3D {
 public:
  // True for a checked graph with explicit subface incidence.
  bool is_explicit_graph() const {
    return is_explicit_graph_;
  }
  std::span<const PreparedMacPressureCell3D> cells() const;
  std::span<const FrozenMacPressureLink3D> links() const;
  std::span<const double> integrated_rhs() const;
  // Absolute assembly terms before cancellation, used in the residual/defect roundoff bound.
  std::span<const double> rhs_absolute_term_sums() const;
  // Each entry identifies a positive-conductance component by its lowest cell ordinal.
  std::span<const std::size_t> component_ordinals() const;
  // One zero-pressure pin per compatible unanchored component, in increasing order.
  std::span<const std::size_t> gauge_cell_ordinals() const;

 private:
  PreparedMacPressureSystem3D() = default;
  friend absl::StatusOr<PreparedMacPressureSystem3D> PrepareMacPressureSystem3D(const MacPressureInputs3D&);

  friend absl::StatusOr<PreparedMacPressureSystem3D>
  PrepareExplicitMacPressureSystem3D(double, std::span<const MacPressureCell3D>, std::span<const ExplicitMacPressureFace3D>, std::span<const ExplicitMacPressureBoundaryFace3D>);

  absl::Status Finalize();

  bool is_explicit_graph_ = false;
  std::vector<PreparedMacPressureCell3D> cells_;
  std::vector<FrozenMacPressureLink3D> links_;
  std::vector<double> integrated_rhs_;
  std::vector<double> rhs_absolute_term_sums_;
  std::vector<std::size_t> component_ordinals_;
  std::vector<std::size_t> gauge_cell_ordinals_;
};

// Freeze the original integrated pressure operator and RHS. Returns
// kInvalidArgument for invalid physical data, ordering or face inventory,
// kFailedPrecondition for missing ownership or an incompatible unanchored
// component, kResourceExhausted for excessive extents, and kOutOfRange for
// unrepresentable lattice indices or arithmetic, including inexact flux
// products that round to zero or a subnormal. Errors carry no system.
absl::StatusOr<PreparedMacPressureSystem3D> PrepareMacPressureSystem3D(const MacPressureInputs3D& inputs);

// Freeze a graph with explicit subface ownership. Supply positive finite
// duration, cells ordered by signed (z,y,x), and each face span in increasing
// slot order. Slots must be distinct across both spans. The caller certifies
// geometric coverage and measures. Interior subfaces have two distinct
// in-range endpoints. Exterior subfaces have one in-range endpoint and
// supply a positive aperture with finite exterior pressure, or a zero
// aperture when sealed. Nonzero wall flux requires blocked area.
// Invalid data or ownership returns kInvalidArgument, excessive extents
// kResourceExhausted, unrepresentable arithmetic kOutOfRange, and an
// incompatible unanchored component kFailedPrecondition. The pressure to
// velocity scale must be normal. Errors carry no system.
absl::StatusOr<PreparedMacPressureSystem3D> PrepareExplicitMacPressureSystem3D(
    double step_duration,
    std::span<const MacPressureCell3D> pressure_cells,
    std::span<const ExplicitMacPressureFace3D> pressure_faces,
    std::span<const ExplicitMacPressureBoundaryFace3D> boundary_faces = {}
);

// Step-start pressure input indexed by canonical global subface slot.
struct AdaptiveMacPressureSubface3D {
  double open_area_m2 = 0;
  double inverse_density_m3_per_kg = 0;
  double predicted_normal_velocity_m_per_s = 0;
  double prescribed_wall_flux_m3_per_s = 0;  // Positive Cartesian normal.
  double exterior_pressure_distance_m = 0;  // Positive on open exterior faces only.
  std::optional<double> exterior_pressure_pa = std::nullopt;
  double pressure_jump_pa = 0;  // Equilibrium head pressure minus tail pressure on an open interior link.
};

// Derive incidence, full area and interior center distance from the frozen
// topology. Cells match the canonical leaves and supply liquid volumes in m³,
// source divergence in 1/s and warm-start pressure in Pa. Subfaces cover every
// global slot, including sealed faces. Invalid extents, mismatched cells and
// malformed coefficients return InvalidArgument. Explicit graph errors
// propagate. The returned graph owns its coefficients and never borrows inputs.
absl::StatusOr<PreparedMacPressureSystem3D> PrepareAdaptiveMacPressureSystem3D(
    const AdaptiveMacTopology3D& topology,
    double step_duration,
    std::span<const MacPressureCell3D> cells,
    std::span<const AdaptiveMacPressureSubface3D> subfaces
);

// Evaluate all original rows for the supplied finite pressures in Pa, in
// prepared cell order. Reports residual, independently accumulated physical
// defect and corrected normal velocities without applying stopping tolerances.
// Returns kInvalidArgument for malformed candidates, kOutOfRange for
// unrepresentable arithmetic, including inexact flux and velocity correction
// products that round to zero or a subnormal. Returns kInternal for a
// residual/defect identity mismatch. Errors carry no projection.
absl::StatusOr<MacPressureProjection3D>
EvaluateMacPressure3D(const PreparedMacPressureSystem3D& system, std::span<const double> candidate_cell_pressures);

}  // namespace fluidity::pressure_internal
