#pragma once

#include <array>
#include <cstdint>

#include "absl/status/statusor.h"
#include "fluidity/base/types.h"

namespace fluidity {

// Signed coordinates on the lattice, including nodes outside a physical grid.
struct Node2D {
  std::int64_t x;
  std::int64_t y;
};

struct Stencil2D {
  // The nine entries share the same order, with y varying fastest within each x slot.
  std::array<Node2D, 9> nodes;
  // Dimensionless tensor-product weights, including zero weights at support endpoints.
  std::array<double, 9> weights;
  // Derivatives with respect to physical position, in inverse length.
  std::array<Vec2, 9> gradients;
};

// Builds the complete 3-by-3 stencil without clipping to a grid. Measure relative_position
// from the unshifted lattice origin in physical units. Spacing is in physical units and
// offset is in lattice units. Gradients differentiate with respect to physical position.
// Half-cell ties select the higher stencil.
// Returns kInvalidArgument for nonfinite inputs or nonpositive spacing, and kOutOfRange if
// any node or gradient cannot be represented. Errors carry no stencil.
absl::StatusOr<Stencil2D> QuadraticStencil2D(Vec2 relative_position, double spacing, Vec2 offset);

}  // namespace fluidity
