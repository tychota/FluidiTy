#include "fluidity/host/gas_checkpoint_3d.h"

#include <algorithm>
#include <cmath>

#include "absl/status/status.h"
#include "gas_checkpoint_generated.h"

namespace fluidity {
namespace {

namespace wire = cache::v1;
constexpr std::size_t kMaxCheckpointBytes = 64 * 1024 * 1024;
constexpr std::size_t kMaxCells = 4096;

wire::Vector3 ToWire(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}

Vec3 FromWire(const wire::Vector3& vector) {
  return {vector.x(), vector.y(), vector.z()};
}

bool Finite(Vec3 vector) {
  return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

bool Nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

bool Positive(double value) {
  return std::isfinite(value) && value > 0.0;
}

absl::Status CheckState(const GasCheckpoint3D& checkpoint) {
  const auto& state = checkpoint.state;
  if (state.cells.size() > kMaxCells || state.markers.size() > kMaxCells
      || state.face_velocities.size() > kMaxCheckpointBytes / sizeof(double)) {
    return absl::ResourceExhaustedError("gas checkpoint exceeds format capacity");
  }
  if (state.cells.empty() || state.face_velocities.empty() || state.fuel_burn_rate.size() != state.cells.size()
      || !Nonnegative(checkpoint.time) || checkpoint.frame > checkpoint.accepted_steps
      || (checkpoint.frame == 0 && (checkpoint.time != 0.0 || checkpoint.accepted_steps != 0))
      || (checkpoint.frame > 0 && checkpoint.time <= 0.0)) {
    return absl::InvalidArgumentError("invalid gas checkpoint clock or extents");
  }
  for (const auto& cell : state.cells) {
    if (!Positive(cell.density.carrier) || !Nonnegative(cell.density.fuel) || !Nonnegative(cell.density.soot)
        || !Positive(cell.density.enthalpy) || !Positive(cell.absolute_pressure)
        || !std::isfinite(cell.gauge_pressure)) {
      return absl::InvalidArgumentError("invalid gas checkpoint cell state");
    }
  }
  for (const auto& marker : state.markers) {
    if (!Positive(marker.weight) || !Finite(marker.world_position) || !Finite(marker.velocity)
        || !Finite(marker.velocity_slopes.x_row) || !Finite(marker.velocity_slopes.y_row)
        || !Finite(marker.velocity_slopes.z_row)) {
      return absl::InvalidArgumentError("invalid gas checkpoint APIC marker");
    }
  }
  if (!std::ranges::all_of(
          state.face_velocities,
          [](double value) {
            return std::isfinite(value);
          }
      )
      || !std::ranges::all_of(state.fuel_burn_rate, Nonnegative)) {
    return absl::InvalidArgumentError("invalid gas checkpoint face velocity or flame");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<std::uint8_t>>
EncodeGasCheckpoint3D(const GasCheckpoint3D& checkpoint, const LiquidDigest3D& scene_digest) {
  if (const auto status = CheckState(checkpoint); !status.ok()) {
    return status;
  }
  const auto& state = checkpoint.state;
  std::vector<wire::GasCellState> cells;
  cells.reserve(state.cells.size());
  for (std::size_t index = 0; index < state.cells.size(); ++index) {
    const auto& cell = state.cells[index];
    cells.emplace_back(
        cell.density.carrier,
        cell.density.fuel,
        cell.density.soot,
        cell.density.enthalpy,
        cell.absolute_pressure,
        cell.gauge_pressure,
        state.fuel_burn_rate[index]
    );
  }
  std::vector<wire::GasMarkerState> markers;
  markers.reserve(state.markers.size());
  for (const auto& marker : state.markers) {
    const wire::Matrix3 slopes(
        ToWire(marker.velocity_slopes.x_row),
        ToWire(marker.velocity_slopes.y_row),
        ToWire(marker.velocity_slopes.z_row)
    );
    markers.emplace_back(ToWire(marker.world_position), marker.weight, ToWire(marker.velocity), slopes);
  }
  const wire::Digest digest(flatbuffers::span<const std::uint8_t, 32>(scene_digest.data(), scene_digest.size()));
  const wire::Clock clock(checkpoint.time, checkpoint.frame, 0, checkpoint.accepted_steps, checkpoint.accepted_steps);
  flatbuffers::FlatBufferBuilder builder;
  builder.ForceDefaults(true);
  const auto root =
      wire::CreateGasCheckpointDirect(builder, 1, 0, &digest, &clock, &cells, &markers, &state.face_velocities);
  wire::FinishGasCheckpointBuffer(builder, root);
  if (builder.GetSize() > kMaxCheckpointBytes) {
    return absl::ResourceExhaustedError("encoded gas checkpoint exceeds byte limit");
  }
  auto* encoded = wire::GetMutableGasCheckpoint(builder.GetBufferPointer());
  if (!encoded->mutate_byte_length(builder.GetSize())) {
    return absl::InternalError("gas checkpoint length field was not emitted");
  }
  return std::vector<std::uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

absl::StatusOr<GasCheckpoint3D> DecodeGasCheckpoint3D(
    std::span<const std::uint8_t> bytes,
    const GasUniformDomain3D& domain,
    const LiquidDigest3D& scene_digest
) {
  if (bytes.size() < 8 || bytes.size() > kMaxCheckpointBytes) {
    return absl::InvalidArgumentError("invalid gas checkpoint byte length");
  }
  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!wire::VerifyGasCheckpointBuffer(verifier)) {
    return absl::InvalidArgumentError("invalid FLGC buffer");
  }
  const auto* encoded = wire::GetGasCheckpoint(bytes.data());
  const auto& clock = *encoded->clock();
  if (encoded->schema_version() != 1 || encoded->byte_length() != bytes.size() || clock.substep() != 0
      || clock.epoch() != clock.accepted_step() || encoded->markers()->size() > kMaxCells) {
    return absl::InvalidArgumentError("invalid gas checkpoint version, clock or capacity");
  }
  if (!std::ranges::equal(*encoded->scene_digest()->bytes(), scene_digest)
      || encoded->cells()->size() != domain.cells.size()
      || encoded->face_velocity_m_per_s()->size() != domain.grid.face_count()) {
    return absl::FailedPreconditionError("gas checkpoint does not match resolved scene or layout");
  }
  GasCheckpoint3D checkpoint{
      .frame = clock.frame(),
      .time = clock.time(),
      .accepted_steps = clock.accepted_step(),
      .state =
          {
              .cells = domain.cells,
              .markers = {},
              .face_velocities = {encoded->face_velocity_m_per_s()->begin(), encoded->face_velocity_m_per_s()->end()},
              .fuel_burn_rate = {},
          },
  };
  checkpoint.state.fuel_burn_rate.reserve(domain.cells.size());
  for (flatbuffers::uoffset_t index = 0; index < encoded->cells()->size(); ++index) {
    const auto& stored = *encoded->cells()->Get(index);
    auto& cell = checkpoint.state.cells[index];
    cell.density = {
        .carrier = stored.carrier_kg_per_m3(),
        .fuel = stored.fuel_kg_per_m3(),
        .soot = stored.soot_kg_per_m3(),
        .enthalpy = stored.enthalpy_j_per_m3(),
    };
    cell.absolute_pressure = stored.absolute_pressure_pa();
    cell.gauge_pressure = stored.gauge_pressure_pa();
    checkpoint.state.fuel_burn_rate.push_back(stored.fuel_burn_kg_per_m3_s());
  }
  checkpoint.state.markers.reserve(encoded->markers()->size());
  for (const auto* marker : *encoded->markers()) {
    const auto& slopes = marker->slopes_per_s();
    checkpoint.state.markers.push_back({
        .world_position = FromWire(marker->position_m()),
        .weight = marker->weight(),
        .velocity = FromWire(marker->velocity_m_per_s()),
        .velocity_slopes = {FromWire(slopes.x_row()), FromWire(slopes.y_row()), FromWire(slopes.z_row())},
    });
  }
  if (const auto status = CheckState(checkpoint); !status.ok()) {
    return status;
  }
  return checkpoint;
}

}  // namespace fluidity
