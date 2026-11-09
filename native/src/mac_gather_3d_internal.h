#pragma once
#include <cmath>

#include "fluidity/host/transfers_3d.h"
namespace fluidity::internal {
// Complete-basis gather shared by owning and bounded callers; inputs cover every stencil slot.
struct GatheredMacVelocityComponent {
  double new_pic_velocity;
  double grid_velocity_increment;
  Vec3 fitted_velocity_slope_row;
  Vec3 velocity_gradient_row;
};

inline absl::StatusOr<GatheredMacVelocityComponent> GatherMacVelocityComponent(
    const MacFaceStencil3D& stencil,
    double cell_width,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    bool bounded = false
) {
  const auto IsFinite = [](Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
  };
  const auto AddScaled = [](Vec3& sum, Vec3 value, double scale) {
    sum.x += value.x * scale;
    sum.y += value.y * scale;
    sum.z += value.z * scale;
  };
  double old_pic_velocity = 0.0;
  double new_pic_velocity = 0.0;
  Vec3 normalized_offset_velocity_moment{};
  Vec3 velocity_gradient_row{};
  for (const auto& sample : stencil) {
    const double old_face_velocity = old_face_velocities[sample.face_slot];
    const double new_face_velocity = new_face_velocities[sample.face_slot];
    const double weighted_new_face_velocity = sample.weight * new_face_velocity;
    old_pic_velocity += sample.weight * old_face_velocity;
    new_pic_velocity += weighted_new_face_velocity;
    const Vec3 normalized_face_displacement{
        .x = sample.face_offset_from_particle.x / cell_width,
        .y = sample.face_offset_from_particle.y / cell_width,
        .z = sample.face_offset_from_particle.z / cell_width,
    };
    AddScaled(normalized_offset_velocity_moment, normalized_face_displacement, weighted_new_face_velocity);
    AddScaled(velocity_gradient_row, sample.weight_gradient, new_face_velocity);
  }
  // Complete quadratic support makes the fit's normal matrix I/4 in cell
  // coordinates. Apply its inverse before converting the slope to inverse
  // seconds. Forming cell_width squared would narrow the usable scale range.
  const Vec3 fitted_velocity_slope_row{
      .x = (4.0 * normalized_offset_velocity_moment.x) / cell_width,
      .y = (4.0 * normalized_offset_velocity_moment.y) / cell_width,
      .z = (4.0 * normalized_offset_velocity_moment.z) / cell_width,
  };
  const double grid_velocity_increment = new_pic_velocity - old_pic_velocity;
  if (!std::isfinite(new_pic_velocity) || !std::isfinite(grid_velocity_increment)
      || !IsFinite(fitted_velocity_slope_row) || !IsFinite(velocity_gradient_row)) {
    return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "gathered velocity or slope is nonfinite");
  }
  return GatheredMacVelocityComponent{
      .new_pic_velocity = new_pic_velocity,
      .grid_velocity_increment = grid_velocity_increment,
      .fitted_velocity_slope_row = fitted_velocity_slope_row,
      .velocity_gradient_row = velocity_gradient_row,
  };
}

}  // namespace fluidity::internal
