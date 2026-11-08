#include "fluidity/host/liquid_wall_contact_3d.h"

#include "fluidity/base/liquid_wall_contact_math_3d.h"
#include "liquid_boundary_backend.h"
namespace fluidity {
absl::StatusOr<LiquidWallContactResult3D> ResolveLiquidWallContact3D(const LiquidWallContactInputs3D& inputs) {
  const auto result = boundary_internal::ResolveContact(boundary_internal::PackContact(inputs));
  if (!result.ok()) {
    return boundary_internal::ArithmeticStatus(result.status());
  }
  return boundary_internal::UnpackContact(*result);
}
}  // namespace fluidity
