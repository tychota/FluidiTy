#pragma once

#include "fluidity/base/solid_acoustic_3d.h"
#include "fluidity/base/solid_deformation_3d.h"
#include "m5_constitutive_3d_cases.h"

namespace fluidity::m5_test {

FLUIDITY_HD inline int CheckDeformationCases() {
  const SnowParameters3D
      snow{.shear_modulus = 2, .lame_lambda = 3, .compression_limit = 0.1, .stretch_limit = 0.05, .hardening = 5};
  // U swaps the x and y axes with one sign reversal. V rotates about z by 45 degrees.
  constexpr double q = 0.7071067811865475244;
  const Mat3 trial{{1.2 * q, -1.2 * q, 0}, {0.7 * q, 0.7 * q, 0}, {0, 0, 1}};
  SolidDeformationState3D state{};
  if (!ReturnSnowDeformation3D(trial, 0, snow, state) || !Near(state.elastic_deformation.x_row.x, 1.05 * q)
      || !Near(state.elastic_deformation.x_row.y, -1.05 * q) || !Near(state.elastic_deformation.y_row.x, 0.9 * q)
      || !Near(state.elastic_deformation.y_row.y, 0.9 * q) || !Near(state.elastic_deformation.z_row.z, 1)) {
    return 21;
  }
  const double jacobian = 0.9 * 1.05;
  const double hardening = ::exp(5 * (1 - 0.84 / jacobian));
  if (!Near(state.kirchhoff_stress.x_row.x, hardening * (4 * 0.05 * 1.05 + 3 * (jacobian - 1) * jacobian))
      || !Near(state.kirchhoff_stress.y_row.y, hardening * (-4 * 0.1 * 0.9 + 3 * (jacobian - 1) * jacobian))
      || !Near(state.kirchhoff_stress.x_row.y, 0) || !Near(state.log_plastic_jacobian, ::log(0.84 / jacobian))) {
    return 22;
  }
  const Mat3 reflection{{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const Mat3 singular{{1, 0, 0}, {0, 0, 0}, {0, 0, 1}};
  if (ReturnSnowDeformation3D(reflection, 0, snow, state) || ReturnSnowDeformation3D(singular, 0, snow, state)
      || !Near(state.log_plastic_jacobian, ::log(0.84 / jacobian))) {
    return 23;
  }
  const Mat3 dependent_rows{{-2, 1, -5}, {2, 2, -3}, {2, 5, -11}};
  if (ReturnSnowDeformation3D(dependent_rows, 0, snow, state)
      || !Near(state.log_plastic_jacobian, ::log(0.84 / jacobian))) {
    return 29;
  }
  const GranularParameters3D sand{
      .shear_modulus = 2,
      .lame_lambda = 3,
      .friction_angle = 0.5,
      .cohesion_log_strain = 0
  };
  const Mat3 rotated{{0, -0.9, 0}, {0.9 + 1e-14, 0, 0}, {0, 0, 0.9}};
  if (!ReturnGranularDeformation3D(rotated, 0, sand, state) || !Near(state.elastic_deformation.x_row.y, -0.9)
      || !Near(state.elastic_deformation.y_row.x, 0.9) || !Near(state.kirchhoff_stress.y_row.y, 13 * ::log(0.9))) {
    return 24;
  }
  double acoustic_bound = 0;
  if (!GranularAcousticBoundSquared3D({1, 1 + 1e-14, 1}, sand, 2, acoustic_bound) || !Near(acoustic_bound, 6.5)) {
    return 25;
  }
  if (!SnowAcousticBoundSquared3D({1, 1, 1}, ::log(0.8), snow, 2, acoustic_bound)
      || !Near(acoustic_bound, 6.5 * ::exp(1.0))) {
    return 26;
  }
  if (GranularAcousticBoundSquared3D({2, 2, 2}, sand, 2, acoustic_bound) || !Near(acoustic_bound, 6.5 * ::exp(1.0))) {
    return 27;
  }
  // All axial modes are positive, but the x-z acoustic minor at
  // n=(sqrt(0.75),0,0.5) has negative determinant.
  const GranularParameters3D oblique_sand{1, 1, 1.5, 0};
  if (GranularAcousticBoundSquared3D({0.2, 0.1, 2.5}, oblique_sand, 1, acoustic_bound)
      || !Near(acoustic_bound, 6.5 * ::exp(1.0))) {
    return 28;
  }
  return 0;
}

}  // namespace fluidity::m5_test
