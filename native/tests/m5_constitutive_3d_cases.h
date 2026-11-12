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
  // The cohesionless 30-degree law matches the triaxial compression ratio
  // (1 + sin(30 degrees)) / (1 - sin(30 degrees)) = 3. These trial
  // stretches give stresses (-5, -1, -1) Pa with the moduli above.
  auto cohesionless_sand = sand;
  cohesionless_sand.cohesion_log_strain = 0;
  const Vec3 triaxial_trial{::exp(-11.0 / 13), ::exp(2.0 / 13), ::exp(2.0 / 13)};
  if (!ReturnGranularPrincipal3D(triaxial_trial, 0, cohesionless_sand, state) || !Near(state.kirchhoff_stress.x, -4.2)
      || !Near(state.kirchhoff_stress.y, -1.4) || !Near(state.kirchhoff_stress.z, -1.4)
      || !Near(state.log_plastic_jacobian, 0)) {
    return 9;
  }
  // Plane strain with only normal compression and shear has lateral stress
  // set by lambda. The principal log strains are the eigenvalues of its
  // symmetric 2x2 compression/shear block, plus a zero out-of-plane strain.
  constexpr double kCompression = 0.01;
  const double alpha = ::sqrt(2.0 / 3) * 0.4;
  constexpr double kLambdaValues[]{700, 1280};
  constexpr double kShearFactors[]{0.999, 1.001};
  for (const double lambda : kLambdaValues) {
    const double shear_modulus = (1600 - lambda) / 2;
    const GranularParameters3D plane_strain{shear_modulus, lambda, 0.5235987755982989, 0};
    const double cone_scale = (3 * lambda + 2 * shear_modulus) * alpha / shear_modulus;
    const double yield_shear = kCompression * ::sqrt(0.5 * (cone_scale * cone_scale - 8.0 / 3));
    for (const double factor : kShearFactors) {
      const double radius = ::hypot(kCompression, factor * yield_shear);
      const Vec3 trial{
          ::exp(0.5 * (-kCompression + radius)),
          1,
          ::exp(0.5 * (-kCompression - radius)),
      };
      if (!ReturnGranularPrincipal3D(trial, 0, plane_strain, state)
          || Near(state.elastic_stretches.x, trial.x) != (factor < 1) || !Near(state.log_plastic_jacobian, 0)) {
        return 10;
      }
    }
  }
  return 0;
}

}  // namespace fluidity::m5_test
