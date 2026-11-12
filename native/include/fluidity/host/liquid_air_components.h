#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"

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

// Measures the sampled cells and emits positive-volume gas cells in the
// same order. Supply unique (z,y,x) cells and corners, including all eight
// corners per cell. Atmosphere outlets are optional certified outer faces,
// ordered by (axis,z,y,x). Each must have a positive gas opening and exactly
// one incident sampled cell. An unlisted opening into an unsampled cell
// remains an unknown frontier. Output scratch must be disjoint from inputs.
// Returns the number of populated gas records. Scratch may change on failure.
// Invalid input returns InvalidArgument, insufficient output returns
// ResourceExhausted, and clipping/sample failures propagate.
absl::StatusOr<std::size_t> BuildLiquidAirDomain3D(
    UniformMacGeometry3D geometry,
    const LiquidObstacle3D& obstacle,
    std::span<const CellIndex3D> sampled_cells,
    std::span<const LiquidCornerDistance3D> corners,
    std::span<const FaceIndex3D> atmosphere_outlets,
    std::span<LiquidAirVolumeCell3D> gas_cells
);

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

struct LiquidAirComponentSnapshot3D {
  std::span<const LiquidAirVolumeCell3D> cells;  // Unique and sorted by (z,y,x).
  std::span<const std::size_t> labels;  // One component index per cell.
  std::size_t component_count;
};

struct LiquidAirComponentOverlap3D {
  std::size_t previous_component;
  std::size_t current_component;
  double shared_cell_volume;  // m³, sum of the smaller gas volume in each coincident cell.
};

// Estimates lineage weights between two snapshots on the same MAC lattice.
// Coincident cell gas measures give an occupancy weight, with subcell motion
// unresolved. All component classes participate, including vented and unknown.
// Scratch can hold one entry per coincident cell and must be disjoint from
// both snapshots. Returns the compacted edge count in (previous,current)
// order. Scratch may change on failure. InvalidArgument covers malformed
// snapshots or aliasing, ResourceExhausted covers scratch capacity, and
// OutOfRange covers accumulated volume overflow.
absl::StatusOr<std::size_t> MeasureLiquidAirComponentOverlaps3D(
    LiquidAirComponentSnapshot3D previous,
    LiquidAirComponentSnapshot3D current,
    std::span<LiquidAirComponentOverlap3D> overlaps
);

}  // namespace fluidity
