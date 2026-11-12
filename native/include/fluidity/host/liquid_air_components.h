#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

enum class LiquidAirPassage3D : std::uint8_t {
  kClosed,
  kOpen,  // Connects to a represented neighbor or an unclassified frontier.
  kAtmosphere,  // Certified outlet to atmosphere beyond the supplied cells.
};

struct LiquidAirVolumeCell3D {
  CellIndex3D cell_index;
  double gas_volume;  // Positive geometric gas volume in m³.
  std::array<LiquidAirPassage3D, 6> passages{};  // Low/high x, low/high y, low/high z.
};

enum class LiquidAirConnectivity3D : std::uint8_t {
  kEnclosed,
  kUnclassified,
  kVented,
};

struct LiquidAirComponent3D {
  CellIndex3D first_cell;  // First member in canonical (z,y,x) order.
  double gas_volume;  // m³, summed over the component.
  LiquidAirConnectivity3D connectivity;
};

// Labels gas cells in first-member order using face connectivity. Cells must
// be unique and sorted by (z,y,x). Shared faces must agree, and atmosphere
// outlets must face outside the supplied cells. An open missing neighbor
// without a certified outlet makes its component unclassified. A known
// atmosphere path takes precedence over an unclassified frontier.
// Label and queue scratch must hold one entry per cell. Component scratch
// holds one entry per detected component, at most the number of cells. All
// spans must be disjoint. Returns the number of populated components. Scratch
// may change on error. Bad records or extents return InvalidArgument, short
// component scratch returns ResourceExhausted, and volume overflow returns
// OutOfRange.
absl::StatusOr<std::size_t> LabelLiquidAirComponents3D(
    std::span<const LiquidAirVolumeCell3D> cells,
    std::span<std::size_t> labels,
    std::span<std::size_t> queue,
    std::span<LiquidAirComponent3D> components
);

}  // namespace fluidity
