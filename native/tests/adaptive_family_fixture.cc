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
Run(std::string_view backend_name, std::string_view mode, const char* checkpoint_path, std::string_view scenario) {
  const bool band = scenario == "band";
  const auto family_scenario = band ? std::string_view{} : scenario;
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
  const std::uint8_t fixture_identity[] = {1, 3, static_cast<std::uint8_t>(band ? 5 : 12)};
  resume.resolved_fingerprint = LiquidContentDigest3D(fixture_identity);
  if (scenario == "eos" || scenario == "viscous") {
    const std::uint8_t extended_identity[]{1, 4, 12, static_cast<std::uint8_t>(scenario == "eos" ? 1 : 2)};
    resume.resolved_fingerprint = LiquidContentDigest3D(extended_identity);
  }
  if (scenario == "miscible") {
    const std::uint8_t extended_identity[]{1, 4, 12, 3};
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

  if (mode != "resume") {
    const auto refined =
        band ? AdvanceBand(*scheduler, 0.01, true) : AdvanceFamily(*scheduler, 0.01, true, family_scenario);
    if (!refined.ok()) {
      return refined;
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

  const auto coarsened =
      band ? AdvanceBand(*scheduler, 0.02, false) : AdvanceFamily(*scheduler, 0.02, false, family_scenario);
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
  if (!scenario.empty() && scenario != "band" && scenario != "eos" && scenario != "viscous" && scenario != "miscible") {
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
