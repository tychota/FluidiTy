#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "graded_tank_fixture.h"
#ifdef FLUIDITY_ADAPTIVE_FIXTURE_CUDA
#include "fluidity/host/cuda_liquid_substep_3d.h"
#endif

namespace fluidity {
namespace {

constexpr LiquidStateCapacities3D kCapacities{12, 0, 12, 64, 0};
constexpr AdaptiveMacCellKey3D kParent{.lower_corner = {0, 0, 0}, .width_exponent = 1};

absl::Status
AdvanceFamily(LiquidSubstepScheduler3D& scheduler, double frame_end, bool refine, std::string_view scenario) {
  const auto started = scheduler.BeginFrame(frame_end);
  if (!started.ok()) {
    return started;
  }
  if (!refine && scenario == "eos") {
    const auto before = EncodeLiquidState3D(scheduler.committed());
    if (!before.ok()) {
      return before.status();
    }
    const auto rejected = scheduler.AttemptWithAdaptiveFamily(0.01, kParent, false);
    if (rejected.status().code() != absl::StatusCode::kOutOfRange
        || rejected.status().message() != "adaptive exterior flux is not exactly representable on the coarse face") {
      return absl::FailedPreconditionError("EOS coarsening did not report its unrepresentable exterior flux");
    }
    const auto after = EncodeLiquidState3D(scheduler.committed());
    if (!after.ok()) {
      return after.status();
    }
    if (!std::ranges::equal(before->values(), after->values())) {
      return absl::FailedPreconditionError("refused EOS coarsening changed accepted topology or material");
    }
    const auto advanced = scheduler.Attempt(0.01);
    if (!advanced.ok()) {
      return advanced.status();
    }
    if (advanced->disposition != LiquidAttemptDisposition3D::kAccepted
        || scheduler.committed().adaptive_cells.size() != 12U) {
      return absl::FailedPreconditionError("EOS could not advance on the retained refined topology");
    }
    return scheduler.FinishFrame();
  }
  double duration = 0.01;
  if (!scenario.empty()) {
    const auto before = EncodeLiquidState3D(scheduler.committed());
    if (!before.ok()) {
      return before.status();
    }
    bool reached_accept = false;
    scheduler.SetStageFaultHookForTesting({
        .context = &reached_accept,
        .before_stage =
            [](void* context, LiquidPhysicalStage3D stage, double) {
              if (stage == LiquidPhysicalStage3D::kAccept) {
                *static_cast<bool*>(context) = true;
                return absl::AbortedError("reject prepared family");
              }
              return absl::OkStatus();
            },
    });
    const auto rejected = scheduler.AttemptWithAdaptiveFamily(duration, kParent, refine);
    scheduler.SetStageFaultHookForTesting({});
    if (!rejected.ok()) {
      return rejected.status();
    }
    const auto after = EncodeLiquidState3D(scheduler.committed());
    if (!after.ok()) {
      return after.status();
    }
    if (!reached_accept || scheduler.diagnostics().failed_stage != LiquidPhysicalStage3D::kAccept
        || rejected->disposition != LiquidAttemptDisposition3D::kRetryDt
        || !std::ranges::equal(before->values(), after->values())) {
      return absl::FailedPreconditionError("rejected family trial changed the accepted state or disposition");
    }
    duration /= 2.0;
  }
  const auto attempted = scheduler.AttemptWithAdaptiveFamily(duration, kParent, refine);
  if (!attempted.ok()) {
    return attempted.status();
  }
  if (attempted->disposition != LiquidAttemptDisposition3D::kAccepted
      || scheduler.committed().adaptive_cells.size() != (refine ? 12U : 5U)) {
    return absl::FailedPreconditionError("family trial did not publish the expected topology");
  }
  if (scenario == "viscous" || scenario == "miscible") {
    const auto& measurement = scheduler.diagnostics().accepted_measurement;
    if (!measurement || !measurement->viscous_max_momentum_residual_kg_m_per_s) {
      return absl::FailedPreconditionError("accepted family trial did not run the viscous solve");
    }
  }
  if (scenario == "miscible") {
    const auto& accepted = scheduler.committed();
    if (accepted.particles.size() != 12) {
      return absl::FailedPreconditionError("miscible family transaction changed the particle roster");
    }
    std::array<double, 2> constituent_volumes_m3{};
    for (const auto& cell : accepted.adaptive_cells.values()) {
      const auto& inventory = cell.inventory;
      if (inventory.phase_id != 4 || inventory.constituent_count != 2 || inventory.constituents[0].substance_id != 8
          || inventory.constituents[1].substance_id != 14
          || inventory.constituents[0].volume_m3 + inventory.constituents[1].volume_m3 != inventory.current_volume_m3) {
        return absl::FailedPreconditionError("miscible family transaction lost constituent inventory");
      }
      for (std::size_t substance = 0; substance < constituent_volumes_m3.size(); ++substance) {
        constituent_volumes_m3[substance] += inventory.constituents[substance].volume_m3;
      }
    }
    // The authored tank holds 1.5 m³, split equally between two stable IDs.
    if (constituent_volumes_m3 != std::array{0.75, 0.75}) {
      return absl::FailedPreconditionError("miscible family transaction changed per-substance volumes");
    }
  }
  if (scenario == "eos" && std::ranges::none_of(scheduler.committed().particles.values(), [](const auto& particle) {
        return particle.volume_ratio < 1.0;
      })) {
    return absl::FailedPreconditionError("accepted family trial did not compress under EOS");
  }
  if (!scenario.empty()) {
    const auto settled = scheduler.Attempt(duration);
    if (!settled.ok()) {
      return settled.status();
    }
    if (settled->disposition != LiquidAttemptDisposition3D::kAccepted) {
      return absl::FailedPreconditionError("fixed-topology remainder after family trial was rejected");
    }
  }
  return scheduler.FinishFrame();
}

absl::Status AdvanceBand(LiquidSubstepScheduler3D& scheduler, double frame_end, bool retire) {
  const auto started = scheduler.BeginFrame(frame_end);
  if (!started.ok()) {
    return started;
  }
  constexpr AdaptiveMacCellKey3D kBandLeaf{.lower_corner = {2, 0, 0}, .width_exponent = 0};
  const LiquidHandoffCarrier3D carrier{.current_density = 2, .volume_ratio = 1};
  constexpr auto kFull = LiquidHandoffOccupancy3D::kFullLiquid;
  const auto attempted = retire ? scheduler.AttemptWithAdaptiveBandRetirement(0.01, kBandLeaf, kFull, carrier)
                                : scheduler.AttemptWithAdaptiveCoreReseed(0.01, kBandLeaf, kFull, carrier, 1);
  if (!attempted.ok()) {
    return attempted.status();
  }
  const auto expected_owner =
      retire ? AdaptiveLiquidMaterialOwner3D::kEulerian : AdaptiveLiquidMaterialOwner3D::kParticles;
  if (attempted->disposition != LiquidAttemptDisposition3D::kAccepted
      || scheduler.committed().particles.size() != (retire ? 4U : 5U)
      || scheduler.committed().adaptive_cells[1].material_owner != expected_owner) {
    return absl::FailedPreconditionError("band trial did not publish the expected ownership");
  }
  return scheduler.FinishFrame();
}

absl::Status
AdvanceWidth(LiquidSubstepScheduler3D& scheduler, bool earn_age, absl::FunctionRef<absl::Status()> check_device_epoch) {
  constexpr double kHysteresisDuration = 0.01;
  const double frame_end = earn_age ? 0.02 : 0.03;
  const auto started = scheduler.BeginFrame(frame_end);
  if (!started.ok()) {
    return started;
  }
  const std::array halo{AdaptiveCoarsenHaloSample3D{.demand = {.maximum_width = 2}}};
  LiquidAdaptiveWidthRequest3D request{
      .accepted_epoch = scheduler.committed().clock.epoch,
      .leaf = {.lower_corner = kParent.lower_corner, .width_exponent = 0},
      .demand = {.maximum_width = 2},
      .parent_and_halo = halo,
      .complete_halo = true,
      .hysteresis_duration_s = kHysteresisDuration,
  };
  double duration = frame_end - scheduler.committed().clock.time;
  if (earn_age) {
    scheduler.SetStageFaultHookForTesting({
        .context = nullptr,
        .before_stage =
            [](void*, LiquidPhysicalStage3D stage, double) {
              return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject eligible age")
                                                             : absl::OkStatus();
            },
    });
    const auto rejected = scheduler.AttemptWithAdaptiveWidth(duration, request);
    scheduler.SetStageFaultHookForTesting({});
    const auto credited = std::ranges::any_of(scheduler.committed().adaptive_cells.values(), [](const auto& cell) {
      return cell.coarsening_eligible_duration_s != 0;
    });
    if (!rejected.ok() || rejected->disposition != LiquidAttemptDisposition3D::kRetryDt || credited
        || !rejected->retry_duration || scheduler.committed().clock.epoch != request.accepted_epoch) {
      return absl::FailedPreconditionError("rejected width trial advanced accepted age");
    }
    const auto verified_rollback = check_device_epoch();
    if (!verified_rollback.ok()) {
      return verified_rollback;
    }
    const auto retried = scheduler.AttemptWithAdaptiveWidth(*rejected->retry_duration, request);
    if (!retried.ok()) {
      return retried.status();
    }
    if (retried->disposition != LiquidAttemptDisposition3D::kAccepted) {
      return absl::FailedPreconditionError("width retry did not accept the smaller interval");
    }
    request.accepted_epoch = scheduler.committed().clock.epoch;
    duration = frame_end - scheduler.committed().clock.time;
  }
  const auto attempted = scheduler.AttemptWithAdaptiveWidth(duration, request);
  if (!attempted.ok()) {
    return attempted.status();
  }
  if (attempted->disposition != LiquidAttemptDisposition3D::kAccepted
      || scheduler.committed().adaptive_cells.size() != (earn_age ? 12U : 5U)) {
    return absl::FailedPreconditionError("width trial did not publish the expected topology");
  }
  const auto eligible_count =
      std::ranges::count_if(scheduler.committed().adaptive_cells.values(), [](const auto& cell) {
        return cell.coarsening_eligible_duration_s == kHysteresisDuration;
      });
  if (eligible_count != (earn_age ? 8 : 0)) {
    return absl::FailedPreconditionError("width trial did not publish the expected eligible age");
  }
  return scheduler.FinishFrame();
}

absl::Status
AdvanceRefinementClosure(LiquidSubstepScheduler3D& scheduler, absl::FunctionRef<absl::Status()> check_device_epoch) {
  const auto started = scheduler.BeginFrame(0.01);
  if (!started.ok()) {
    return started;
  }
  constexpr std::array requested{AdaptiveMacCellKey3D{.lower_corner = {4, 0, 0}, .width_exponent = 1}};
  const auto before = EncodeLiquidState3D(scheduler.committed());
  if (!before.ok()) {
    return before.status();
  }
  scheduler.SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject complete refinement closure")
                                                           : absl::OkStatus();
          },
  });
  const auto rejected = scheduler.AttemptWithAdaptiveRefinementClosure(0.01, requested);
  scheduler.SetStageFaultHookForTesting({});
  if (!rejected.ok()) {
    return rejected.status();
  }
  const auto after = EncodeLiquidState3D(scheduler.committed());
  if (!after.ok()) {
    return after.status();
  }
  if (rejected->disposition != LiquidAttemptDisposition3D::kRetryDt || !rejected->retry_duration
      || scheduler.diagnostics().failed_stage != LiquidPhysicalStage3D::kAccept
      || !std::ranges::equal(before->values(), after->values())) {
    return absl::FailedPreconditionError("late closure rejection changed accepted state");
  }
  const auto checked = check_device_epoch();
  if (!checked.ok()) {
    return checked;
  }
  const auto accepted = scheduler.AttemptWithAdaptiveRefinementClosure(*rejected->retry_duration, requested);
  if (!accepted.ok()) {
    return accepted.status();
  }
  if (accepted->disposition != LiquidAttemptDisposition3D::kAccepted
      || scheduler.committed().adaptive_cells.size() != 19) {
    return absl::FailedPreconditionError("refinement closure did not publish both families");
  }
  const auto remainder = scheduler.Attempt(0.01 - scheduler.committed().clock.time);
  if (!remainder.ok()) {
    return remainder.status();
  }
  if (remainder->disposition != LiquidAttemptDisposition3D::kAccepted) {
    return absl::FailedPreconditionError("refinement closure remainder was rejected");
  }
  return scheduler.FinishFrame();
}

constexpr TranslatingHalfspace3D kPhysicalReplayWall{
    .unit_normal = {0, 1, 0},
    .start_plane_coordinate = 0.125,
    .wall_velocity = {0.015625, 0.03125, 0},
};

absl::Status AdvancePhysical(
    LiquidSubstepScheduler3D& scheduler,
    double frame_end,
    bool first_frame,
    absl::FunctionRef<absl::Status()> check_device_epoch
) {
  const auto started = scheduler.BeginFrame(frame_end);
  if (!started.ok()) {
    return started;
  }
  const auto initial_duration = frame_end - scheduler.committed().clock.time;
  if (scheduler.Attempt(initial_duration).status().code() != absl::StatusCode::kFailedPrecondition) {
    return absl::FailedPreconditionError("Physical replay allowed an implicit attempt");
  }
  const auto& state = scheduler.committed();
  std::vector<AdaptiveMacCellKey3D> keys;
  keys.reserve(state.adaptive_cells.size());
  for (const auto& cell : state.adaptive_cells.values()) {
    keys.push_back(cell.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto air = std::ranges::find(
      state.adaptive_cells.values(),
      LiquidCellOccupancy3D::kAir,
      &LiquidAdaptiveCellState3D::occupancy
  );
  std::optional<std::size_t> air_leaf;
  if (air != state.adaptive_cells.values().end()) {
    air_leaf = static_cast<std::size_t>(air - state.adaptive_cells.values().begin());
  }
  std::vector<std::optional<LiquidAdaptivePhysicalExteriorPressure3D>> exterior;
  std::vector<std::size_t> cut_slots;
  if (!first_frame) {
    exterior.resize(state.adaptive_faces.size());
    std::size_t outlet_count = 0;
    for (const auto& face : ownership->exterior_faces) {
      const auto index = face.face_index.lattice_index;
      if (face.face_index.axis == FaceNormalAxis::kX && face.cell_is_tail && index.y == index.z) {
        const double pressure_pa = index.y == 0 ? 0.5 : -0.5;
        exterior[face.face_slot] = LiquidAdaptivePhysicalExteriorPressure3D{
            .pressure_pa = pressure_pa,
            .distance_m = 0.25,
        };
        ++outlet_count;
      }
    }
    if (outlet_count != 2) {
      return absl::FailedPreconditionError("Physical replay lacks its two canonical outlets");
    }
    for (std::size_t ordinal = 0; ordinal < topology->interior_faces().size(); ++ordinal) {
      const auto& face = topology->interior_faces()[ordinal];
      if (face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.y == 0) {
        cut_slots.push_back(ownership->interior_face_slots[ordinal]);
      }
    }
  }
  const auto wall_coordinate = [](double time) {
    return kPhysicalReplayWall.start_plane_coordinate + time * kPhysicalReplayWall.wall_velocity.y;
  };
  std::vector<double> sources(topology->cells().size());
  const auto attempt = [&](double duration, double air_source = 0) {
    const auto clock = scheduler.committed().clock;
    const double wall_y = wall_coordinate(clock.time);
    // The normal wall sweeps the bottom of each incident leaf. Tangential
    // wall motion also gives equal and opposite flux on the cut x subfaces.
    for (std::size_t cell = 0; cell < sources.size(); ++cell) {
      const auto& leaf = topology->cells()[cell];
      if (leaf.key.lower_corner.y == 0) {
        sources[cell] = -kPhysicalReplayWall.wall_velocity.y / (leaf.cell_width - wall_y);
      }
    }
    if (air_leaf) {
      sources[*air_leaf] = air_source;
    }
    return scheduler.AttemptWithAdaptivePhysicalGeometry(
        duration,
        {.accepted_clock = clock, .target_divergences_s_inv = sources, .exterior_pressures = exterior}
    );
  };

  if (first_frame) {
    const auto before = EncodeLiquidState3D(scheduler.committed());
    if (!before.ok()) {
      return before.status();
    }
    if (air_leaf && attempt(initial_duration, 1).status().code() != absl::StatusCode::kFailedPrecondition) {
      return absl::FailedPreconditionError("Physical sparse replay admitted a source in known air");
    }
    scheduler.SetStageFaultHookForTesting({
        .before_stage =
            [](void*, LiquidPhysicalStage3D stage, double) {
              return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject physical trial at acceptance")
                                                             : absl::OkStatus();
            },
    });
    const auto rejected = attempt(initial_duration);
    scheduler.SetStageFaultHookForTesting({});
    if (!rejected.ok()) {
      return rejected.status();
    }
    const auto after = EncodeLiquidState3D(scheduler.committed());
    if (!after.ok()) {
      return after.status();
    }
    if (rejected->disposition != LiquidAttemptDisposition3D::kRetryDt || !rejected->retry_duration
        || scheduler.diagnostics().failed_stage != LiquidPhysicalStage3D::kAccept
        || !std::ranges::equal(before->values(), after->values())) {
      return absl::FailedPreconditionError("Rejected physical trial changed the committed checkpoint");
    }
    const auto checked = check_device_epoch();
    if (!checked.ok()) {
      return checked;
    }
    const auto retried = attempt(*rejected->retry_duration);
    if (!retried.ok()) {
      return retried.status();
    }
    if (retried->disposition != LiquidAttemptDisposition3D::kAccepted) {
      return absl::Status(scheduler.diagnostics().failure_code, "Physical replay retry was rejected");
    }
  }
  const double step_start = scheduler.committed().clock.time;
  const auto accepted = attempt(frame_end - step_start);
  if (!accepted.ok()) {
    return accepted.status();
  }
  if (accepted->disposition != LiquidAttemptDisposition3D::kAccepted) {
    return absl::Status(scheduler.diagnostics().failure_code, "Physical replay continuation was rejected");
  }
  const auto& measurement = scheduler.diagnostics().accepted_measurement;
  const double expected_volume_m3 = 1.5 * (1 - wall_coordinate(step_start));
  if (!measurement || std::abs(measurement->step_start_pressure_cell_volume_m3 - expected_volume_m3) > 1e-13) {
    return absl::FailedPreconditionError("Physical replay did not use the measured pressure volume");
  }
  if (!first_frame && std::ranges::none_of(cut_slots, [&](std::size_t slot) {
        return std::abs(scheduler.committed().adaptive_faces[slot].carrier_velocity) > 1e-6;
      })) {
    return absl::FailedPreconditionError("Physical replay did not drive flow through a cut coarse/fine subface");
  }
  if (air_leaf) {
    const auto& published = scheduler.committed();
    const auto& leaf = published.adaptive_cells[*air_leaf];
    if (leaf.occupancy != LiquidCellOccupancy3D::kAir || leaf.previous_pressure != 0 || leaf.inventory.mass_kg != 0
        || leaf.inventory.current_volume_m3 != 0) {
      return absl::DataLossError("Physical sparse replay changed the certified air leaf");
    }
    const auto missing = [&](std::size_t slot) {
      const auto& face = published.adaptive_faces[slot];
      return face.validity == LiquidFaceValidity3D::kMissing && face.carrier_velocity == 0;
    };
    for (const auto& face : ownership->exterior_faces) {
      if (face.cell_ordinal == *air_leaf && !missing(face.face_slot)) {
        return absl::DataLossError("Physical sparse replay revived an omitted air carrier");
      }
    }
    for (std::size_t ordinal = 0; ordinal < topology->interior_faces().size(); ++ordinal) {
      const auto& face = topology->interior_faces()[ordinal];
      if ((face.tail_cell_ordinal == *air_leaf || face.head_cell_ordinal == *air_leaf)
          && !missing(ownership->interior_face_slots[ordinal])) {
        return absl::DataLossError("Physical sparse replay revived a dry wet/air carrier");
      }
    }
  }
  return scheduler.FinishFrame();
}

absl::Status AdvanceCertifiedQuadraticFamily(
    LiquidSubstepScheduler3D& scheduler,
    const test::QuadraticFamilyReplayFixture3D& fixture,
    bool merge,
    absl::FunctionRef<absl::Status()> check_device_epoch
) {
  const auto& start = scheduler.committed();
  const std::uint64_t epoch = start.clock.epoch;
  const std::uint64_t first_new_id = start.next_particle_id;
  const std::size_t start_particles = start.particles.size();
  const double duration = 0.02 - start.clock.time;
  const auto before = EncodeLiquidState3D(start);
  if (!before.ok()) {
    return before.status();
  }
  const auto begun = scheduler.BeginFrame(0.02);
  if (!begun.ok()) {
    return begun;
  }
  const auto attempt = [&](double attempt_duration) {
    if (merge) {
      return scheduler.AttemptWithAdaptiveQuadraticMerge(
          attempt_duration,
          {
              .accepted_epoch = epoch,
              .exact_coarsening_halo = fixture.closure,
              .merge_parent = test::QuadraticFamilyReplayFixture3D::kParent,
              .source_particle_ids = std::span(fixture.source_ids).first(fixture.source_count),
          }
      );
    }
    return scheduler.AttemptWithAdaptiveQuadraticSplit(
        attempt_duration,
        {
            .accepted_epoch = epoch,
            .exact_refinement_closure = fixture.closure,
            .split_parent = test::QuadraticFamilyReplayFixture3D::kParent,
            .source_particle_id = fixture.source_ids.front(),
        }
    );
  };
  scheduler.SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject certified quadratic trial")
                                                           : absl::OkStatus();
          },
  });
  const auto rejected = attempt(duration);
  scheduler.SetStageFaultHookForTesting({});
  if (!rejected.ok()) {
    return rejected.status();
  }
  if (rejected->disposition != LiquidAttemptDisposition3D::kRetryDt || !rejected->retry_duration
      || scheduler.diagnostics().failed_stage != LiquidPhysicalStage3D::kAccept) {
    return absl::FailedPreconditionError("certified quadratic trial did not reach late rejection");
  }
  const auto after = EncodeLiquidState3D(scheduler.committed());
  if (!after.ok()) {
    return after.status();
  }
  if (!std::ranges::equal(before->values(), after->values())) {
    return absl::DataLossError("rejected certified quadratic trial changed the committed state");
  }
  const auto verified = check_device_epoch();
  if (!verified.ok()) {
    return verified;
  }
  // Reapply the same certificate at the retry duration to the unchanged epoch.
  const auto accepted = attempt(*rejected->retry_duration);
  if (!accepted.ok()) {
    return accepted.status();
  }
  if (accepted->disposition != LiquidAttemptDisposition3D::kAccepted) {
    return absl::Status(scheduler.diagnostics().failure_code, "certified quadratic trial was rejected");
  }
  const auto& final = scheduler.committed();
  const std::size_t replacements = merge ? 1 : 8;
  if (final.clock.epoch != epoch + 1 || final.next_particle_id != first_new_id + replacements
      || final.particles.size() != start_particles - fixture.source_count + replacements
      || final.adaptive_cells.size() != (merge ? 125 : 314)) {
    return absl::DataLossError("accepted certified quadratic roster or topology differs");
  }
  for (std::uint64_t id = first_new_id; id < final.next_particle_id; ++id) {
    const auto roster = final.particles.values();
    const auto replacement = std::ranges::find(roster, id, &LiquidParticleState3D::id);
    if (replacement == roster.end() || !replacement->quadratic_adaptive_transfer) {
      return absl::DataLossError("accepted certified replacement lost its ID or quadratic basis");
    }
  }
  for (const std::uint64_t source_id : std::span(fixture.source_ids).first(fixture.source_count)) {
    if (std::ranges::find(final.particles.values(), source_id, &LiquidParticleState3D::id)
        != final.particles.values().end()) {
      return absl::DataLossError("accepted certified quadratic source ID remains live");
    }
  }
  const auto published = check_device_epoch();
  if (!published.ok()) {
    return published;
  }
  // The ordinary remainder derives its basis from the accepted replacement roster.
  const auto remainder = scheduler.Attempt(0.02 - final.clock.time);
  if (!remainder.ok()) {
    return remainder.status();
  }
  if (remainder->disposition != LiquidAttemptDisposition3D::kAccepted) {
    return absl::Status(scheduler.diagnostics().failure_code, "certified quadratic replay remainder was rejected");
  }
  return scheduler.FinishFrame();
}

absl::Status
Run(std::string_view backend_name, std::string_view mode, const char* checkpoint_path, std::string_view scenario) {
  const bool band = scenario == "band";
  const bool physical_air = scenario == "physical-air";
  const bool physical = physical_air || scenario == "physical";
  const bool width = scenario == "width";
  const bool batch = scenario == "closure";
  const bool merge = scenario == "quadratic-merge";
  const bool quadratic = merge || scenario == "quadratic-split";
  const bool exchange = scenario == "exchange";
  const bool persistent = scenario == "quadratic-persistent";
  const auto family_scenario =
      band || width || batch || physical || quadratic || exchange || persistent ? std::string_view{} : scenario;
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  std::optional<test::QuadraticFamilyReplayFixture3D> quadratic_fixture;
  std::uint64_t persistent_particle_id = 0;
  const auto authored = [&]() -> absl::Status {
    if (persistent) {
      const auto selected = test::PopulateGradedQuadraticTankFixture3D(state, scene);
      if (!selected.ok()) {
        return selected.status();
      }
      persistent_particle_id = *selected;
      return absl::OkStatus();
    }
    if (quadratic) {
      auto authored_quadratic = test::PopulateQuadraticFamilyReplayFixture3D(state, scene, merge);
      if (!authored_quadratic.ok()) {
        return authored_quadratic.status();
      }
      quadratic_fixture = std::move(*authored_quadratic);
      return absl::OkStatus();
    }
    if (exchange) {
      return test::PopulateGradedOwnerExchangeFixture3D(state, scene);
    }
    if (physical) {
      return test::PopulateGradedPhysicalHalfspaceFixture3D(state, scene, kPhysicalReplayWall, physical_air);
    }
    if (batch) {
      return test::PopulateGradedRefinementClosureFixture3D(state, scene, true);
    }
    return band ? test::PopulateGradedBandFixture3D(state, scene) : test::PopulateGradedFamilyFixture3D(state, scene);
  }();
  if (!authored.ok()) {
    return authored;
  }
  const auto capacities = [&] {
    if (persistent) {
      return LiquidStateCapacities3D{
          state.particles.size(),
          0,
          state.adaptive_cells.size(),
          state.adaptive_faces.size(),
          0
      };
    }
    if (quadratic) {
      return test::QuadraticFamilyReplayFixture3D::kCapacities;
    }
    if (exchange) {
      return LiquidStateCapacities3D{3, 0, 8, 36, 0};
    }
    return batch ? LiquidStateCapacities3D{19, 0, 19, 160, 0} : kCapacities;
  }();
  if (scenario == "eos") {
    test::SetGradedTankEos3D(state, scene);
  }
  if (scenario == "viscous") {
    scene.dynamic_viscosity = 2.0;
  }
  if (scenario == "miscible") {
    static constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 2.0, 1.0}, {14, 4, 2.0, 3.0}}};
    scene.constituent_materials = kMaterials;
    for (auto& particle : state.particles.values()) {
      particle.phase_id = 4;
      particle.constituent_count = 2;
      particle.constituents[0] = {8, 0.5};
      particle.constituents[1] = {14, 0.5};
    }
    for (auto& cell : state.adaptive_cells.values()) {
      auto& inventory = cell.inventory;
      inventory.phase_id = 4;
      inventory.constituent_count = 2;
      inventory.constituents[0] = {8, 0.5 * inventory.current_volume_m3};
      inventory.constituents[1] = {14, 0.5 * inventory.current_volume_m3};
    }
  }
  LiquidResumeMetadata3D resume;
  const std::uint8_t fixture_tag = physical_air
      ? 27
      : (persistent
             ? 26
             : (quadratic ? (merge ? 25 : 24) : (physical ? 23 : (batch ? 19 : (band ? 5 : (width ? 13 : 12))))));
  const std::uint8_t fixture_identity[] = {1, 3, fixture_tag};
  resume.resolved_fingerprint = LiquidContentDigest3D(fixture_identity);
  if (scenario == "eos" || scenario == "viscous") {
    const std::uint8_t extended_identity[]{1, 4, 12, static_cast<std::uint8_t>(scenario == "eos" ? 1 : 2)};
    resume.resolved_fingerprint = LiquidContentDigest3D(extended_identity);
  }
  if (scenario == "miscible") {
    const std::uint8_t extended_identity[]{1, 4, 12, 3};
    resume.resolved_fingerprint = LiquidContentDigest3D(extended_identity);
  }
  if (exchange) {
    const std::uint8_t exchange_identity[]{1, 4, 8, 4};
    resume.resolved_fingerprint = LiquidContentDigest3D(exchange_identity);
  }
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  resume.capacities = capacities;
  if (mode == "resume") {
    std::ifstream file(checkpoint_path, std::ios::binary);
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
    auto checkpoint = DecodeLiquidCheckpoint3D(bytes);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    const std::size_t expected_cells = (quadratic || persistent || physical)
        ? state.adaptive_cells.size()
        : (exchange ? 8 : (batch ? 19 : (band ? 5 : 12)));
    if (checkpoint->resume != resume || checkpoint->state.adaptive_cells.size() != expected_cells
        || (band && checkpoint->state.adaptive_cells[1].material_owner != AdaptiveLiquidMaterialOwner3D::kEulerian)) {
      return absl::DataLossError("fixture checkpoint identity, topology, or ownership differs");
    }
    if (width
        && std::ranges::count_if(
               checkpoint->state.adaptive_cells.values(),
               [](const auto& cell) {
                 return cell.coarsening_eligible_duration_s == 0.01;
               }
           ) != 8) {
      return absl::DataLossError("fixture checkpoint lost the accepted width eligibility");
    }
    if (quadratic_fixture) {
      std::ranges::reverse(quadratic_fixture->closure);
      std::ranges::reverse(std::span(quadratic_fixture->source_ids).first(quadratic_fixture->source_count));
    }
    if (persistent) {
      const auto roster = checkpoint->state.particles.values();
      const auto selected = std::ranges::find(roster, persistent_particle_id, &LiquidParticleState3D::id);
      if (selected == roster.end() || !selected->quadratic_adaptive_transfer) {
        return absl::DataLossError("persistent replay checkpoint lost the accepted basis");
      }
    }
    state = std::move(checkpoint->state);
  }

  auto operations = CpuLiquidSubstepBackend3D();
#ifdef FLUIDITY_ADAPTIVE_FIXTURE_CUDA
  std::optional<CudaLiquidSubstepBackend3D> device;
  if (backend_name == "cuda") {
    auto created = CudaLiquidSubstepBackend3D::Create(capacities, 0, false, false, true);
    if (!created.ok()) {
      return created.status();
    }
    device = std::move(*created);
    operations = device->operations();
    if (physical && (operations.project_adaptive_in_arena == nullptr || operations.project_adaptive != nullptr)) {
      return absl::FailedPreconditionError("Physical replay requires CUDA adaptive arena projection");
    }
  }
#else
  if (backend_name != "cpu") {
    return absl::InvalidArgumentError("CUDA fixture was not built");
  }
#endif
  LiquidStepOptions3D options;
  options.capacities = capacities;
  if (exchange) {
    options.maximum_step_duration = 0.125;
  }
  if (scenario == "miscible" && mode != "resume") {
    // Equal and opposite edits preserve the parent inventory but would move substance
    // between children if prolongation ignored the particles' spatial composition.
    state.particles[0].constituents[0].volume_fraction = 0.75;
    state.particles[0].constituents[1].volume_fraction = 0.25;
    state.particles[1].constituents[0].volume_fraction = 0.25;
    state.particles[1].constituents[1].volume_fraction = 0.75;
    auto uneven = LiquidSubstepScheduler3D::Create(state, scene, options);
    for (std::size_t particle = 0; particle < 2; ++particle) {
      state.particles[particle].constituents[0].volume_fraction = 0.5;
      state.particles[particle].constituents[1].volume_fraction = 0.5;
    }
    if (!uneven.ok()) {
      return uneven.status();
    }
    const auto before = EncodeLiquidState3D(uneven->committed());
    if (!before.ok()) {
      return before.status();
    }
    const auto began = uneven->BeginFrame(0.01);
    if (!began.ok()) {
      return began;
    }
    const auto refused = uneven->AttemptWithAdaptiveFamily(0.01, kParent, true);
    if (refused.status().code() != absl::StatusCode::kFailedPrecondition
        || refused.status().message() != "graded family inventory requires certified particle resampling") {
      return absl::FailedPreconditionError("uneven miscible rebinning did not refuse unrepresentable child inventory");
    }
    const auto after = EncodeLiquidState3D(uneven->committed());
    if (!after.ok()) {
      return after.status();
    }
    if (!std::ranges::equal(before->values(), after->values())) {
      return absl::FailedPreconditionError("refused miscible rebinning changed accepted state");
    }
  }
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, operations);
  if (!scheduler.ok()) {
    return scheduler.status();
  }
  const auto check_device_epoch = [&]() -> absl::Status {
#ifdef FLUIDITY_ADAPTIVE_FIXTURE_CUDA
    if (device) {
      const auto epoch = device->AcquireCommitted();
      if (!epoch.ok()) {
        return epoch.status();
      }
      auto readback = CloneLiquidState3D(scheduler->committed(), capacities);
      if (!readback.ok()) {
        return readback.status();
      }
      const auto copied = epoch->Readback(*readback);
      if (!copied.ok()) {
        return copied;
      }
      if (*readback != scheduler->committed()) {
        return absl::DataLossError("Device replay state differs from the committed host epoch");
      }
    }
#endif
    return absl::OkStatus();
  };

  if (!exchange && mode != "resume") {
    const auto refined = [&] {
      if (persistent) {
        constexpr double kDuration = 0.001;
        const auto begun = scheduler->BeginFrame(kDuration);
        if (!begun.ok()) {
          return begun;
        }
        const auto selected =
            scheduler->AttemptWithAdaptiveQuadraticTransfer(kDuration, state.clock.epoch, persistent_particle_id);
        if (!selected.ok()) {
          return selected.status();
        }
        if (selected->disposition != LiquidAttemptDisposition3D::kAccepted
            || !scheduler->committed().particles[persistent_particle_id - 1].quadratic_adaptive_transfer) {
          return absl::FailedPreconditionError("persistent replay did not publish its quadratic basis");
        }
        return scheduler->FinishFrame();
      }
      if (quadratic) {
        return scheduler->AdvanceFrame(0.01);
      }
      if (physical) {
        return AdvancePhysical(*scheduler, 0.01, true, check_device_epoch);
      }
      if (batch) {
        return AdvanceRefinementClosure(*scheduler, check_device_epoch);
      }
      return band ? AdvanceBand(*scheduler, 0.01, true) : AdvanceFamily(*scheduler, 0.01, true, family_scenario);
    }();
    if (!refined.ok()) {
      return refined;
    }
    if (width) {
      const auto aged = AdvanceWidth(*scheduler, true, check_device_epoch);
      if (!aged.ok()) {
        return aged;
      }
    }
  }
  if (width || batch || physical || quadratic || persistent) {
    const auto verified = check_device_epoch();
    if (!verified.ok()) {
      return verified;
    }
  }
  if (mode == "interrupt") {
    const auto checkpoint = EncodeLiquidCheckpoint3D(scheduler->committed(), resume);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    std::ofstream file(checkpoint_path, std::ios::binary | std::ios::trunc);
    const auto bytes = checkpoint->values();
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();
    if (!file) {
      return absl::DataLossError("fixture checkpoint write failed");
    }
    std::cout << "checkpoint=" << LiquidDigestHex3D(LiquidContentDigest3D(bytes)).data() << std::endl;
    std::cin.get();
    return absl::CancelledError("expected test process termination");
  }

  const auto continued = [&] {
    if (persistent) {
      const auto advanced = scheduler->AdvanceFrame(0.002);
      if (!advanced.ok()) {
        return advanced;
      }
      if (!scheduler->committed().particles[persistent_particle_id - 1].quadratic_adaptive_transfer) {
        return absl::FailedPreconditionError("ordinary replay did not continue its accepted quadratic basis");
      }
      return absl::OkStatus();
    }
    if (quadratic) {
      return AdvanceCertifiedQuadraticFamily(*scheduler, *quadratic_fixture, merge, check_device_epoch);
    }
    if (exchange) {
      constexpr double kDuration = 0.125;
      const auto begun = scheduler->BeginFrame(kDuration);
      if (!begun.ok()) {
        return begun;
      }
      const auto attempted = scheduler->AttemptWithAdaptiveOwnerExchange(kDuration);
      if (!attempted.ok()) {
        return attempted.status();
      }
      const auto& particles = scheduler->committed().particles;
      if (attempted->disposition != LiquidAttemptDisposition3D::kAccepted || particles[0].mass != 0.25
          || particles[1].mass != 0.125) {
        return absl::FailedPreconditionError("replay did not publish the expected projected owner exchange");
      }
      return scheduler->FinishFrame();
    }
    if (physical) {
      return AdvancePhysical(*scheduler, 0.02, false, check_device_epoch);
    }
    if (batch) {
      return scheduler->AdvanceFrame(0.02);
    }
    if (width) {
      return AdvanceWidth(*scheduler, false, check_device_epoch);
    }
    return band ? AdvanceBand(*scheduler, 0.02, false) : AdvanceFamily(*scheduler, 0.02, false, family_scenario);
  }();
  if (!continued.ok()) {
    return continued;
  }
  if (batch || physical || quadratic || exchange || persistent) {
    const auto verified = check_device_epoch();
    if (!verified.ok()) {
      return verified;
    }
  }
#ifdef FLUIDITY_ADAPTIVE_FIXTURE_CUDA
  if (device && device->usage().compiled_architecture != 1200) {
    return absl::FailedPreconditionError("fixture requires SM120 device code");
  }
#endif
  if (exchange) {
    const auto encoded = EncodeLiquidCheckpoint3D(scheduler->committed(), resume);
    if (!encoded.ok()) {
      return encoded.status();
    }
    auto decoded = DecodeLiquidCheckpoint3D(encoded->values());
    if (!decoded.ok()) {
      return decoded.status();
    }
    auto restarted = LiquidSubstepScheduler3D::Create(decoded->state, scene, options);
    if (!restarted.ok()) {
      return restarted.status();
    }
    if (restarted->committed() != scheduler->committed()) {
      return absl::DataLossError("projected owner exchange changed during checkpoint restart");
    }
  }
  const auto final_state = EncodeLiquidState3D(scheduler->committed());
  if (!final_state.ok()) {
    return final_state.status();
  }
  std::cout << "final=" << LiquidDigestHex3D(LiquidContentDigest3D(final_state->values())).data() << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace fluidity

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    return 2;
  }
  const std::string_view scenario = argc == 5 ? argv[4] : "";
  if (!scenario.empty() && scenario != "band" && scenario != "eos" && scenario != "viscous" && scenario != "miscible"
      && scenario != "width" && scenario != "closure" && scenario != "physical" && scenario != "physical-air"
      && scenario != "quadratic-split" && scenario != "quadratic-merge" && scenario != "quadratic-persistent"
      && scenario != "exchange") {
    return 2;
  }
  const std::string_view backend = argv[1];
  const std::string_view mode = argv[2];
  if ((backend != "cpu" && backend != "cuda") || (mode != "continuous" && mode != "interrupt" && mode != "resume")) {
    return 2;
  }
  const auto status = fluidity::Run(backend, mode, argv[3], scenario);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
}
