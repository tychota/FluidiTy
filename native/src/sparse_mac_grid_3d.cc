#include "fluidity/host/sparse_mac_grid_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace fluidity {
namespace {

constexpr std::int64_t kCellsPerBlockAxis = 4;
constexpr std::int64_t kTransferTileWidthInCells = 2 * kCellsPerBlockAxis;
constexpr std::size_t kFacesPerNormalInBlock = kCellsPerBlockAxis * kCellsPerBlockAxis * kCellsPerBlockAxis;
constexpr std::size_t kFacesPerBlock = 3 * kFacesPerNormalInBlock;
constexpr std::int64_t kMinSignedIndex = std::numeric_limits<std::int32_t>::min();
constexpr std::int64_t kMaxSignedIndex = std::numeric_limits<std::int32_t>::max();
constexpr std::int64_t kMinBlockIndex = kMinSignedIndex / kCellsPerBlockAxis;
constexpr std::int64_t kMaxBlockIndex = kMaxSignedIndex / kCellsPerBlockAxis;
constexpr double kParticleBinShiftInCells = 1.5;

bool IsFinite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

absl::Status ValidateGeometry(UniformMacGeometry3D geometry) {
  if (!IsFinite(geometry.world_origin)) {
    return absl::InvalidArgumentError("grid origin must be finite");
  }
  const bool cell_width_is_positive_normal = geometry.cell_width > 0.0 && std::isnormal(geometry.cell_width);
  if (!cell_width_is_positive_normal) {
    return absl::InvalidArgumentError("cell width must be positive and normal");
  }
  return absl::OkStatus();
}

auto BlockSortKey(BlockIndex3D block_index) {
  return std::tuple{block_index.z, block_index.y, block_index.x};
}

// C++ division truncates toward zero; a negative remainder belongs to the previous block.
std::int64_t FloorDivideByBlockWidth(std::int64_t index) {
  const std::int64_t quotient = index / kCellsPerBlockAxis;
  return index % kCellsPerBlockAxis < 0 ? quotient - 1 : quotient;
}

bool IsValidAxis(FaceNormalAxis axis) {
  return axis == FaceNormalAxis::kX || axis == FaceNormalAxis::kY || axis == FaceNormalAxis::kZ;
}

std::array<double, 3> FaceLatticeShifts(FaceNormalAxis axis) {
  return {
      axis == FaceNormalAxis::kX ? 0.0 : 0.5,
      axis == FaceNormalAxis::kY ? 0.0 : 0.5,
      axis == FaceNormalAxis::kZ ? 0.0 : 0.5
  };
}

absl::StatusOr<Vec3> ReconstructWorldPosition(UniformMacGeometry3D geometry, Vec3 normalized_position) {
  // Preserve the affine residual through translation, as the stencil edge comparisons do.
  const Vec3 world_position{
      .x = std::fma(geometry.cell_width, normalized_position.x, geometry.world_origin.x),
      .y = std::fma(geometry.cell_width, normalized_position.y, geometry.world_origin.y),
      .z = std::fma(geometry.cell_width, normalized_position.z, geometry.world_origin.z),
  };
  if (!IsFinite(world_position)) {
    return absl::OutOfRangeError("world position is outside the finite range");
  }
  return world_position;
}

// Recover the roundoff from the finite subtraction. Callers use its sign
// to distinguish coordinates that round onto the same lattice edge.
double SubtractionRoundoff(double minuend, double subtrahend, double difference) {
  // FastTwoSum recovers the smaller operand's contribution after subtraction.
  if (std::abs(minuend) >= std::abs(subtrahend)) {
    return (minuend - difference) - subtrahend;
  }
  return minuend - (difference + subtrahend);
}

// Retain the scaled world subtraction so lattice decisions can recover
// position lost when the quotient rounds to a half-cell or bin edge.
struct NormalizedCoordinate {
  double rounded_coordinate_in_cells;
  double scaled_world_difference;
  double scaled_world_difference_error;
  double scaled_cell_width;
  int world_scale_exponent;

  double ScaledOffsetFrom(double edge) const {
    return std::fma(-edge, scaled_cell_width, scaled_world_difference) + scaled_world_difference_error;
  }

  bool IsBelow(double edge) const {
    const double offset = rounded_coordinate_in_cells - edge;
    // Subtraction and division together can round the coordinate across a nearby edge.
    const double error_bound = std::abs(rounded_coordinate_in_cells) * (2.0 * std::numeric_limits<double>::epsilon());
    if (std::abs(offset) > error_bound) {
      return offset < 0.0;
    }
    return ScaledOffsetFrom(edge) < 0.0;
  }

  double OffsetFrom(double edge) const {
    const double offset = rounded_coordinate_in_cells - edge;
    if (std::abs(offset) >= std::abs(rounded_coordinate_in_cells) && offset != 0.0) {
      return offset;
    }
    // Recompute local differences where subtraction cancels leading digits of the coordinate.
    const double scaled_offset = ScaledOffsetFrom(edge);
    return std::isfinite(scaled_offset) ? scaled_offset / scaled_cell_width : offset;
  }

  double FaceDisplacement(double anchor, double cell_width) const {
    const double normalized_displacement = -OffsetFrom(anchor);
    const double displacement = normalized_displacement * cell_width;
    if (std::isnormal(normalized_displacement) && std::isfinite(displacement)) {
      return displacement;
    }
    // A normalized offset may underflow even when the world displacement is representable.
    const double scaled_offset = ScaledOffsetFrom(anchor);
    return scaled_offset == 0.0 ? 0.0 : -std::scalbn(scaled_offset, -world_scale_exponent);
  }
};

NormalizedCoordinate NormalizeCoordinate(double world_coordinate, double world_origin, double cell_width) {
  double scaled_world_difference = world_coordinate - world_origin;
  int world_scale_exponent = 0;
  if (!std::isfinite(scaled_world_difference)) {
    // If the subtraction overflowed, both operands are large enough to halve exactly.
    world_coordinate *= 0.5;
    world_origin *= 0.5;
    scaled_world_difference = world_coordinate - world_origin;
    world_scale_exponent = -1;
  }
  double scaled_cell_width = std::scalbn(cell_width, world_scale_exponent);
  const double rounded_coordinate_in_cells = scaled_world_difference / scaled_cell_width;
  double scaled_world_difference_error = SubtractionRoundoff(world_coordinate, world_origin, scaled_world_difference);
  if (std::isfinite(rounded_coordinate_in_cells) && scaled_cell_width < 0x1p-960) {
    // Leave room for the exact product of a normal cell width and a half-integer edge.
    world_scale_exponent += 64;
    scaled_world_difference = std::scalbn(scaled_world_difference, 64);
    scaled_world_difference_error = std::scalbn(scaled_world_difference_error, 64);
    scaled_cell_width = std::scalbn(scaled_cell_width, 64);
  }
  return {
      rounded_coordinate_in_cells,
      scaled_world_difference,
      scaled_world_difference_error,
      scaled_cell_width,
      world_scale_exponent
  };
}

using NormalizedPosition = std::array<NormalizedCoordinate, 3>;

absl::StatusOr<NormalizedPosition> NormalizePosition(UniformMacGeometry3D geometry, Vec3 world_position) {
  if (!IsFinite(world_position)) {
    return absl::InvalidArgumentError("particle world position must be finite");
  }
  const NormalizedPosition normalized_position{
      NormalizeCoordinate(world_position.x, geometry.world_origin.x, geometry.cell_width),
      NormalizeCoordinate(world_position.y, geometry.world_origin.y, geometry.cell_width),
      NormalizeCoordinate(world_position.z, geometry.world_origin.z, geometry.cell_width),
  };
  for (const auto& coordinate : normalized_position) {
    if (!std::isfinite(coordinate.rounded_coordinate_in_cells)) {
      return absl::OutOfRangeError("normalized particle position is outside the finite range");
    }
  }
  return normalized_position;
}

// Range checks at the call sites leave room for either correction in the widened index.
std::int64_t FloorShiftedCoordinate(const NormalizedCoordinate& coordinate, double shift) {
  const auto candidate_floor_index =
      static_cast<std::int64_t>(std::floor(coordinate.rounded_coordinate_in_cells - shift));
  const double edge = static_cast<double>(candidate_floor_index) + shift;
  if (coordinate.IsBelow(edge)) {
    return candidate_floor_index - 1;
  }
  if (coordinate.IsBelow(edge + 1.0)) {
    return candidate_floor_index;
  }
  return candidate_floor_index + 1;
}

absl::StatusOr<ParticleBinIndex3D> CheckedParticleSortBin(const NormalizedPosition& sorted_normalized_position) {
  constexpr double first_bin_edge =
      static_cast<double>(kMinSignedIndex * kCellsPerBlockAxis) + kParticleBinShiftInCells;
  constexpr double past_last_bin_edge =
      static_cast<double>((kMaxSignedIndex + 1) * kCellsPerBlockAxis) + kParticleBinShiftInCells;
  std::array<std::int32_t, 3> bin_indices;
  for (std::size_t axis_index = 0; axis_index < sorted_normalized_position.size(); ++axis_index) {
    const auto& coordinate = sorted_normalized_position[axis_index];
    if (coordinate.IsBelow(first_bin_edge) || !coordinate.IsBelow(past_last_bin_edge)) {
      return absl::OutOfRangeError("particle bin index is outside the int32 range");
    }
    const std::int64_t bin_index =
        FloorDivideByBlockWidth(FloorShiftedCoordinate(coordinate, kParticleBinShiftInCells));
    bin_indices[axis_index] = static_cast<std::int32_t>(bin_index);
  }
  return ParticleBinIndex3D{
      .x = bin_indices[0],
      .y = bin_indices[1],
      .z = bin_indices[2],
  };
}

bool WithinHalfCell(double sorted_world_coordinate, double trial_world_coordinate, double cell_width) {
  const double greater_world_coordinate = std::max(sorted_world_coordinate, trial_world_coordinate);
  const double lesser_world_coordinate = std::min(sorted_world_coordinate, trial_world_coordinate);
  const double displacement = greater_world_coordinate - lesser_world_coordinate;
  // Avoid rounding a subnormal half-width.
  const double doubled_displacement = 2.0 * displacement;
  if (doubled_displacement != cell_width) {
    return doubled_displacement < cell_width;
  }

  // Recover the subtraction residual when rounding lands on the inclusive motion limit.
  return SubtractionRoundoff(greater_world_coordinate, lesser_world_coordinate, displacement) <= 0.0;
}

struct QuadraticSplineFactor {
  double rounded_factor;
  // An outer weight is half this squared distance, even if the factor underflowed.
  double outer_weight_edge_distance = 0.0;

  bool IsZero() const {
    return rounded_factor == 0.0 && outer_weight_edge_distance == 0.0;
  }

  double Mantissa(int& exponent) const {
    if (std::isnormal(rounded_factor) || outer_weight_edge_distance == 0.0) {
      return std::frexp(rounded_factor, &exponent);
    }
    int distance_exponent;
    const double distance_mantissa = std::frexp(outer_weight_edge_distance, &distance_exponent);
    exponent = 2 * distance_exponent;
    return 0.5 * distance_mantissa * distance_mantissa;
  }
};

struct AxisStencilSample {
  std::int32_t lattice_index;
  QuadraticSplineFactor weight;
  double face_offset_from_particle;
  double normalized_weight_derivative;
};

using AxisStencil = std::array<AxisStencilSample, 3>;

// Checks all three indices, including the zero-weight support edge at an exact tie.
absl::StatusOr<AxisStencil> BuildCheckedAxisStencil(
    const NormalizedCoordinate& coordinate,
    double lattice_shift,
    std::int32_t bin_index,
    double cell_width
) {
  const double start_shift = lattice_shift + 0.5;
  const double first_support_edge = static_cast<double>(kMinSignedIndex) + start_shift;
  const double past_last_support_edge = static_cast<double>(kMaxSignedIndex - 1) + start_shift;
  if (coordinate.IsBelow(first_support_edge) || !coordinate.IsBelow(past_last_support_edge)) {
    return absl::OutOfRangeError("quadratic stencil face index is outside the int32 range");
  }
  const std::int64_t first_lattice_index = FloorShiftedCoordinate(coordinate, start_shift);
  const std::int64_t tile_start = static_cast<std::int64_t>(bin_index) * kCellsPerBlockAxis;
  const std::int64_t last_lattice_index = first_lattice_index + 2;
  if (first_lattice_index < tile_start || last_lattice_index >= tile_start + kTransferTileWidthInCells) {
    return absl::FailedPreconditionError("quadratic stencil leaves the sorted particle tile");
  }

  // Evaluate the outer weights from the distances to their exact support edges.
  // Forming these distances through the center offset would round away small weights near a tie.
  const double first_anchor = static_cast<double>(first_lattice_index) + lattice_shift;
  const double distance_to_left_edge = coordinate.OffsetFrom(first_anchor + 0.5);
  const double distance_to_right_edge = std::abs(coordinate.OffsetFrom(first_anchor + 1.5));
  const double particle_offset_from_center = coordinate.OffsetFrom(first_anchor + 1.0);
  const std::array weights{
      QuadraticSplineFactor{0.5 * distance_to_right_edge * distance_to_right_edge, distance_to_right_edge},
      QuadraticSplineFactor{0.75 - particle_offset_from_center * particle_offset_from_center},
      QuadraticSplineFactor{0.5 * distance_to_left_edge * distance_to_left_edge, distance_to_left_edge},
  };
  const std::array derivatives{-distance_to_right_edge, -2.0 * particle_offset_from_center, distance_to_left_edge};

  AxisStencil stencil;
  for (std::size_t sample_index = 0; sample_index < stencil.size(); ++sample_index) {
    const std::int64_t lattice_index = first_lattice_index + static_cast<std::int64_t>(sample_index);
    const double anchor = static_cast<double>(lattice_index) + lattice_shift;
    stencil[sample_index] = {
        .lattice_index = static_cast<std::int32_t>(lattice_index),
        .weight = weights[sample_index],
        .face_offset_from_particle = coordinate.FaceDisplacement(anchor, cell_width),
        .normalized_weight_derivative = derivatives[sample_index],
    };
  }
  return stencil;
}

double PhysicalWeightGradient(
    QuadraticSplineFactor x_factor,
    QuadraticSplineFactor y_factor,
    QuadraticSplineFactor z_factor,
    double cell_width
) {
  const double normalized_gradient = x_factor.rounded_factor * y_factor.rounded_factor * z_factor.rounded_factor;
  if (std::isnormal(normalized_gradient) || x_factor.IsZero() || y_factor.IsZero() || z_factor.IsZero()) {
    return normalized_gradient / cell_width;
  }

  // Keep small outer weights and products in exponent form until applying the physical scale.
  int x_exponent;
  int y_exponent;
  int z_exponent;
  int width_exponent;
  const double x_mantissa = x_factor.Mantissa(x_exponent);
  const double y_mantissa = y_factor.Mantissa(y_exponent);
  const double z_mantissa = z_factor.Mantissa(z_exponent);
  const double width_mantissa = std::frexp(cell_width, &width_exponent);
  return std::scalbn(
      (x_mantissa * y_mantissa * z_mantissa) / width_mantissa,
      x_exponent + y_exponent + z_exponent - width_exponent
  );
}

}  // namespace

SparseMacGrid3D::SparseMacGrid3D(UniformMacGeometry3D geometry, std::vector<BlockIndex3D> active_block_indices)
    : geometry_(geometry), active_block_indices_(std::move(active_block_indices)) {
}

absl::StatusOr<SparseMacGrid3D> SparseMacGrid3D::Create(
    UniformMacGeometry3D geometry,
    std::span<const BlockIndex3D> active_block_indices,
    std::size_t block_capacity
) {
  const absl::Status geometry_status = ValidateGeometry(geometry);
  if (!geometry_status.ok()) {
    return geometry_status;
  }
  if (active_block_indices.size() > block_capacity) {
    return absl::ResourceExhaustedError("active block count exceeds block capacity");
  }
  if (active_block_indices.size() > std::numeric_limits<std::size_t>::max() / kFacesPerBlock) {
    return absl::OutOfRangeError("owned face count is outside the size_t range");
  }
  for (BlockIndex3D block_index : active_block_indices) {
    for (std::int32_t coordinate : {block_index.x, block_index.y, block_index.z}) {
      if (coordinate < kMinBlockIndex || coordinate > kMaxBlockIndex) {
        return absl::OutOfRangeError("active block faces are outside the int32 lattice");
      }
    }
  }

  std::vector<BlockIndex3D> sorted_block_indices(active_block_indices.begin(), active_block_indices.end());
  std::ranges::sort(sorted_block_indices, {}, BlockSortKey);
  const auto duplicate = std::ranges::adjacent_find(sorted_block_indices, {}, BlockSortKey);
  if (duplicate != sorted_block_indices.end()) {
    return absl::InvalidArgumentError("active block keys must be unique");
  }
  return SparseMacGrid3D(geometry, std::move(sorted_block_indices));
}

UniformMacGeometry3D SparseMacGrid3D::geometry() const {
  return geometry_;
}

std::size_t SparseMacGrid3D::face_count() const {
  return kFacesPerBlock * active_block_indices_.size();
}

std::span<const BlockIndex3D> SparseMacGrid3D::active_block_indices() const {
  return active_block_indices_;
}

absl::StatusOr<std::size_t> SparseMacGrid3D::FaceSlot(FaceIndex3D face_index) const {
  if (!IsValidAxis(face_index.axis)) {
    return absl::InvalidArgumentError("face axis is invalid");
  }
  const FaceLatticeIndex3D lattice_index = face_index.lattice_index;
  const BlockIndex3D owner{
      .x = static_cast<std::int32_t>(FloorDivideByBlockWidth(lattice_index.x)),
      .y = static_cast<std::int32_t>(FloorDivideByBlockWidth(lattice_index.y)),
      .z = static_cast<std::int32_t>(FloorDivideByBlockWidth(lattice_index.z)),
  };
  const auto owner_key = BlockSortKey(owner);
  const auto owner_block_iterator = std::ranges::lower_bound(active_block_indices_, owner_key, {}, BlockSortKey);
  if (owner_block_iterator == active_block_indices_.end() || BlockSortKey(*owner_block_iterator) != owner_key) {
    return absl::FailedPreconditionError("face owner block is inactive");
  }

  const std::int64_t local_x = lattice_index.x - kCellsPerBlockAxis * owner.x;
  const std::int64_t local_y = lattice_index.y - kCellsPerBlockAxis * owner.y;
  const std::int64_t local_z = lattice_index.z - kCellsPerBlockAxis * owner.z;
  const auto local_face_ordinal =
      static_cast<std::size_t>((local_z * kCellsPerBlockAxis + local_y) * kCellsPerBlockAxis + local_x);
  const auto block_ordinal = static_cast<std::size_t>(owner_block_iterator - active_block_indices_.begin());
  const auto axis_ordinal = static_cast<std::size_t>(face_index.axis);
  // The flat buffer groups faces by owner block, then normal, then cell.
  return block_ordinal * kFacesPerBlock + axis_ordinal * kFacesPerNormalInBlock + local_face_ordinal;
}

absl::StatusOr<Vec3> SparseMacGrid3D::FaceWorldAnchor(FaceIndex3D face_index) const {
  if (!IsValidAxis(face_index.axis)) {
    return absl::InvalidArgumentError("face axis is invalid");
  }
  const auto face_lattice_shifts = FaceLatticeShifts(face_index.axis);
  const FaceLatticeIndex3D lattice_index = face_index.lattice_index;
  return ReconstructWorldPosition(
      geometry_,
      {
          .x = lattice_index.x + face_lattice_shifts[0],
          .y = lattice_index.y + face_lattice_shifts[1],
          .z = lattice_index.z + face_lattice_shifts[2],
      }
  );
}

absl::StatusOr<Vec3> SparseMacGrid3D::CellWorldCenter(CellIndex3D cell_index) const {
  return ReconstructWorldPosition(
      geometry_,
      {
          .x = cell_index.x + 0.5,
          .y = cell_index.y + 0.5,
          .z = cell_index.z + 0.5,
      }
  );
}

absl::StatusOr<ParticleBinIndex3D>
ComputeParticleSortBin3D(UniformMacGeometry3D geometry, Vec3 sorted_particle_world_position) {
  const absl::Status geometry_status = ValidateGeometry(geometry);
  if (!geometry_status.ok()) {
    return geometry_status;
  }
  const auto sorted_normalized_position = NormalizePosition(geometry, sorted_particle_world_position);
  if (!sorted_normalized_position.ok()) {
    return sorted_normalized_position.status();
  }
  return CheckedParticleSortBin(*sorted_normalized_position);
}

absl::StatusOr<MacFaceStencil3D> BuildOwnedMacFaceStencil3D(
    const SparseMacGrid3D& mac_grid,
    FaceNormalAxis axis,
    Vec3 sorted_particle_world_position,
    Vec3 trial_particle_world_position
) {
  if (!IsValidAxis(axis)) {
    return absl::InvalidArgumentError("face axis is invalid");
  }
  const UniformMacGeometry3D geometry = mac_grid.geometry();
  const auto sorted_normalized_position = NormalizePosition(geometry, sorted_particle_world_position);
  if (!sorted_normalized_position.ok()) {
    return sorted_normalized_position.status();
  }
  const auto trial_normalized_position = NormalizePosition(geometry, trial_particle_world_position);
  if (!trial_normalized_position.ok()) {
    return trial_normalized_position.status();
  }
  const auto bin = CheckedParticleSortBin(*sorted_normalized_position);
  if (!bin.ok()) {
    return bin.status();
  }
  const double cell_width = geometry.cell_width;
  const bool motion_within_half_cell =
      WithinHalfCell(sorted_particle_world_position.x, trial_particle_world_position.x, cell_width)
      && WithinHalfCell(sorted_particle_world_position.y, trial_particle_world_position.y, cell_width)
      && WithinHalfCell(sorted_particle_world_position.z, trial_particle_world_position.z, cell_width);
  if (!motion_within_half_cell) {
    return absl::FailedPreconditionError("particle motion exceeds half a cell");
  }

  const auto face_lattice_shifts = FaceLatticeShifts(axis);
  const std::array bin_indices{bin->x, bin->y, bin->z};
  std::array<AxisStencil, 3> axis_stencils;
  for (std::size_t axis_index = 0; axis_index < axis_stencils.size(); ++axis_index) {
    const auto stencil = BuildCheckedAxisStencil(
        (*trial_normalized_position)[axis_index],
        face_lattice_shifts[axis_index],
        bin_indices[axis_index],
        cell_width
    );
    if (!stencil.ok()) {
      return stencil.status();
    }
    axis_stencils[axis_index] = *stencil;
  }

  MacFaceStencil3D stencil;
  std::size_t sample_index = 0;
  for (const AxisStencilSample& z_sample : axis_stencils[2]) {
    for (const AxisStencilSample& y_sample : axis_stencils[1]) {
      for (const AxisStencilSample& x_sample : axis_stencils[0]) {
        const FaceIndex3D face_index{
            .axis = axis,
            .lattice_index = {x_sample.lattice_index, y_sample.lattice_index, z_sample.lattice_index},
        };
        const auto face_slot = mac_grid.FaceSlot(face_index);
        if (!face_slot.ok()) {
          return face_slot.status();
        }
        const Vec3 displacement{
            .x = x_sample.face_offset_from_particle,
            .y = y_sample.face_offset_from_particle,
            .z = z_sample.face_offset_from_particle,
        };
        const Vec3 gradient{
            .x = PhysicalWeightGradient(
                {x_sample.normalized_weight_derivative},
                y_sample.weight,
                z_sample.weight,
                cell_width
            ),
            .y = PhysicalWeightGradient(
                x_sample.weight,
                {y_sample.normalized_weight_derivative},
                z_sample.weight,
                cell_width
            ),
            .z = PhysicalWeightGradient(
                x_sample.weight,
                y_sample.weight,
                {z_sample.normalized_weight_derivative},
                cell_width
            ),
        };
        if (!IsFinite(displacement) || !IsFinite(gradient)) {
          return absl::OutOfRangeError("quadratic stencil displacement or gradient is outside the finite range");
        }
        stencil[sample_index] = {
            .face_slot = *face_slot,
            .weight = x_sample.weight.rounded_factor * y_sample.weight.rounded_factor * z_sample.weight.rounded_factor,
            .face_offset_from_particle = displacement,
            .weight_gradient = gradient,
        };
        ++sample_index;
      }
    }
  }
  return stencil;
}

}  // namespace fluidity
