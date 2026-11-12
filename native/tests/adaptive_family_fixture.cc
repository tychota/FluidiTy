#include <algorithm>
#include <array>
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
  if (scenario == "viscous") {
    const auto& measurement = scheduler.diagnostics().accepted_measurement;
    if (!measurement || !measurement->viscous_max_momentum_residual_kg_m_per_s) {
      return absl::FailedPreconditionError("accepted family trial did not run the viscous solve");
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
Run(std::string_view backend_name, std::string_view mode, const char* checkpoint_path, std::string_view scenario) {
  const bool band = scenario == "band";
  const bool width = scenario == "width";
  const auto family_scenario = band || width ? std::string_view{} : scenario;
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored =
      band ? test::PopulateGradedBandFixture3D(state, scene) : test::PopulateGradedFamilyFixture3D(state, scene);
  if (!authored.ok()) {
    return authored;
  }
  if (scenario == "eos") {
    test::SetGradedTankEos3D(state, scene);
  }
  if (scenario == "viscous") {
    scene.dynamic_viscosity = 2.0;
  }
  LiquidResumeMetadata3D resume;
  const std::uint8_t fixture_identity[] = {1, 3, static_cast<std::uint8_t>(band ? 5 : width ? 13 : 12)};
  resume.resolved_fingerprint = LiquidContentDigest3D(fixture_identity);
  if (scenario == "eos" || scenario == "viscous") {
    const std::uint8_t extended_identity[]{1, 4, 12, static_cast<std::uint8_t>(scenario == "eos" ? 1 : 2)};
    resume.resolved_fingerprint = LiquidContentDigest3D(extended_identity);
  }
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  resume.capacities = kCapacities;
  if (mode == "resume") {
    std::ifstream file(checkpoint_path, std::ios::binary);
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
    auto checkpoint = DecodeLiquidCheckpoint3D(bytes);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    if (checkpoint->resume != resume || checkpoint->state.adaptive_cells.size() != (band ? 5U : 12U)
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
    state = std::move(checkpoint->state);
  }

  auto operations = CpuLiquidSubstepBackend3D();
#ifdef FLUIDITY_ADAPTIVE_FIXTURE_CUDA
  std::optional<CudaLiquidSubstepBackend3D> device;
  if (backend_name == "cuda") {
    auto created = CudaLiquidSubstepBackend3D::Create(kCapacities, 0, false, false, true);
    if (!created.ok()) {
      return created.status();
    }
    device = std::move(*created);
    operations = device->operations();
  }
#else
  if (backend_name != "cpu") {
    return absl::InvalidArgumentError("CUDA fixture was not built");
  }
#endif
  LiquidStepOptions3D options;
  options.capacities = kCapacities;
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
      auto readback = CloneLiquidState3D(scheduler->committed(), kCapacities);
      if (!readback.ok()) {
        return readback.status();
      }
      const auto copied = epoch->Readback(*readback);
      if (!copied.ok()) {
        return copied;
      }
      if (*readback != scheduler->committed()) {
        return absl::DataLossError("device width state differs from the committed host epoch");
      }
    }
#endif
    return absl::OkStatus();
  };

  if (mode != "resume") {
    const auto refined =
        band ? AdvanceBand(*scheduler, 0.01, true) : AdvanceFamily(*scheduler, 0.01, true, family_scenario);
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
  if (width) {
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

  const auto coarsened = width
      ? AdvanceWidth(*scheduler, false, check_device_epoch)
      : (band ? AdvanceBand(*scheduler, 0.02, false) : AdvanceFamily(*scheduler, 0.02, false, family_scenario));
  if (!coarsened.ok()) {
    return coarsened;
  }
#ifdef FLUIDITY_ADAPTIVE_FIXTURE_CUDA
  if (device && device->usage().compiled_architecture != 1200) {
    return absl::FailedPreconditionError("fixture requires SM120 device code");
  }
#endif
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
  if (!scenario.empty() && scenario != "band" && scenario != "eos" && scenario != "viscous" && scenario != "width") {
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
