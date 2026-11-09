#include "fluidity/host/mac_pressure_3d.h"

#include "fluidity/host/mac_pressure_workspace_3d.h"

namespace fluidity {
absl::StatusOr<MacPressureProjection3D>
ProjectMacPressure3D(const MacPressureInputs3D& inputs, MacPressureSolverOptions3D options) {
  if (options.max_iterations == 0 || options.max_iterations > 256) {
    return absl::InvalidArgumentError("pressure iteration budget must be in [1,256]");
  }
  if (inputs.pressure_cells.size() > 4096 || inputs.pressure_faces.size() > 24576) {
    return absl::ResourceExhaustedError("3D pressure system exceeds cell or face capacity");
  }
  auto workspace = MacPressureWorkspace3D::Create(inputs.pressure_cells.size(), inputs.pressure_faces.size());
  if (!workspace.ok()) {
    return workspace.status();
  }
  const auto projection = ProjectMacPressure3D(
      inputs.mac_grid.view(),
      inputs.step_duration,
      {inputs.pressure_cells, inputs.pressure_faces},
      options,
      *workspace
  );
  if (!projection.ok()) {
    return projection.status();
  }
  return MacPressureProjection3D{
      .cell_pressures = {projection->cell_pressures.begin(), projection->cell_pressures.end()},
      .corrected_face_velocities =
          {projection->corrected_face_velocities.begin(), projection->corrected_face_velocities.end()},
      .integrated_rhs = {projection->integrated_rhs.begin(), projection->integrated_rhs.end()},
      .true_residuals = {projection->true_residuals.begin(), projection->true_residuals.end()},
      .cell_constraint_defects =
          {projection->cell_constraint_defects.begin(), projection->cell_constraint_defects.end()},
      .gauge_cell_ordinals = {projection->gauge_cell_ordinals.begin(), projection->gauge_cell_ordinals.end()},
      .diagnostics = projection->diagnostics,
  };
}
}  // namespace fluidity
