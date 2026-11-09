#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/mac_pressure_3d.h"

namespace fluidity::pressure_internal {

struct PreparedMacPressureCell3D {
  CellIndex3D cell_index;
  double fluid_volume;  // m³.
  double integrated_target_flux;  // V times target divergence, m³/s.
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

// Owns validated coefficients and topology independently of the original input
// storage. The following views borrow from this object and follow input order.
class PreparedMacPressureSystem3D {
 public:
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
