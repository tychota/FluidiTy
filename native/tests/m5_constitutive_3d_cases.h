#pragma once

#include <cmath>
#if defined(__CUDACC__)
#include <cuda/std/limits>
#else
#include <limits>
#endif

#include "fluidity/base/m5_constitutive_3d.h"

namespace fluidity::m5_test {

using fluidity::GranularParameters3D;
using fluidity::PrincipalSolidState3D;
using fluidity::SnowParameters3D;
using fluidity::Vec3;

FLUIDITY_HD inline bool Near(double actual, double expected) {
  return ::fabs(actual - expected) < 1e-11;
}

// A nonzero return identifies the first failed physical invariant.
FLUIDITY_HD inline int CheckConstitutiveCases() {
  const GranularParameters3D sand{
      .shear_modulus = 2,
      .lame_lambda = 3,
      .friction_angle = 0.5235987755982989,
      .cohesion_log_strain = 0.03
  };
  PrincipalSolidState3D state{};
  if (!fluidity::ReturnGranularPrincipal3D({0.9, 0.9, 0.9}, 0.2, sand, state) || !Near(state.elastic_stretches.x, 0.9)
      || !Near(state.log_plastic_jacobian, 0.2) || !Near(state.kirchhoff_stress.x, 13 * ::log(0.9))) {
    return 1;
  }
  if (!fluidity::ReturnGranularPrincipal3D({1.2, 1.1, 1.0}, 0, sand, state)
      || !Near(state.elastic_stretches.x, ::exp(0.01)) || !Near(state.elastic_stretches.y, ::exp(0.01))
      || !Near(state.elastic_stretches.z, ::exp(0.01)) || !Near(state.log_plastic_jacobian, ::log(1.32) - 0.03)) {
    return 2;
  }
  if (!fluidity::ReturnGranularPrincipal3D({::exp(0.4), ::exp(-0.1), ::exp(-0.6)}, 0.2, sand, state)) {
    return 3;
  }
  const double compression = 0.33;
  const double cone_radius = (13.0 / 4) * compression * ::sqrt(2.0 / 3) * 0.4;
  const double shear_offset = cone_radius / ::sqrt(2.0);
  if (!Near(::log(state.elastic_stretches.x), -0.1 + shear_offset) || !Near(::log(state.elastic_stretches.y), -0.1)
      || !Near(::log(state.elastic_stretches.z), -0.1 - shear_offset) || !Near(state.log_plastic_jacobian, 0.2)) {
    return 4;
  }
  const SnowParameters3D
      snow{.shear_modulus = 2, .lame_lambda = 3, .compression_limit = 0.1, .stretch_limit = 0.05, .hardening = 5};
  if (!fluidity::ReturnSnowPrincipal3D({0.7, 1.2, 0.98}, 0, snow, state) || !Near(state.elastic_stretches.x, 0.9)
      || !Near(state.elastic_stretches.y, 1.05) || !Near(state.elastic_stretches.z, 0.98)) {
    return 5;
  }
  const double elastic_jacobian = 0.9 * 1.05 * 0.98;
  const double plastic_jacobian = (0.7 * 1.2 * 0.98) / elastic_jacobian;
  const double stress_x =
      (4 * (0.9 - 1) * 0.9 + 3 * (elastic_jacobian - 1) * elastic_jacobian) * ::exp(5 * (1 - plastic_jacobian));
  if (!Near(::exp(state.log_plastic_jacobian) * elastic_jacobian, 0.7 * 1.2 * 0.98)
      || !Near(state.kirchhoff_stress.x, stress_x)) {
    return 6;
  }
  const double first_compaction = state.log_plastic_jacobian;
  if (!fluidity::ReturnSnowPrincipal3D({0.8, 1, 1}, first_compaction, snow, state)
      || !Near(state.log_plastic_jacobian, first_compaction + ::log(0.8 / 0.9))
      || !(state.log_plastic_jacobian < first_compaction)) {
    return 7;
  }
  const PrincipalSolidState3D before = state;
#if defined(__CUDACC__)
  constexpr double kInfinity = cuda::std::numeric_limits<double>::infinity();
#else
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
#endif
  if (fluidity::ReturnGranularPrincipal3D({-1, 1, 1}, 0, sand, state)
      || fluidity::ReturnSnowPrincipal3D({1, 1, 1}, kInfinity, snow, state)
      || fluidity::ReturnSnowPrincipal3D({1, 1, 1}, 1000, snow, state)
      || !Near(state.log_plastic_jacobian, before.log_plastic_jacobian)
      || !Near(state.kirchhoff_stress.x, before.kirchhoff_stress.x)) {
    return 8;
  }
  return 0;
}

}  // namespace fluidity::m5_test
