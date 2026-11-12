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

absl::Status AdvanceFamily(LiquidSubstepScheduler3D& scheduler, double frame_end, bool refine) {
  const auto started = scheduler.BeginFrame(frame_end);
  if (!started.ok()) {
    return started;
  }
  const auto attempted = scheduler.AttemptWithAdaptiveFamily(0.01, kParent, refine);
  if (!attempted.ok()) {
    return attempted.status();
  }
  if (attempted->disposition != LiquidAttemptDisposition3D::kAccepted
      || scheduler.committed().adaptive_cells.size() != (refine ? 12U : 5U)) {
    return absl::FailedPreconditionError("family trial did not publish the expected topology");
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

absl::Status Run(std::string_view backend_name, std::string_view mode, const char* checkpoint_path, bool band) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored =
      band ? test::PopulateGradedBandFixture3D(state, scene) : test::PopulateGradedFamilyFixture3D(state, scene);
  if (!authored.ok()) {
    return authored;
  }
  LiquidResumeMetadata3D resume;
  const std::uint8_t fixture_identity[] = {1, 3, static_cast<std::uint8_t>(band ? 5 : 12)};
  resume.resolved_fingerprint = LiquidContentDigest3D(fixture_identity);
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
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, operations);
  if (!scheduler.ok()) {
    return scheduler.status();
  }

  if (mode != "resume") {
    const auto refined = band ? AdvanceBand(*scheduler, 0.01, true) : AdvanceFamily(*scheduler, 0.01, true);
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

  const auto coarsened = band ? AdvanceBand(*scheduler, 0.02, false) : AdvanceFamily(*scheduler, 0.02, false);
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
  if (argc != 4 && (argc != 5 || std::string_view(argv[4]) != "band")) {
    return 2;
  }
  const std::string_view backend = argv[1];
  const std::string_view mode = argv[2];
  if ((backend != "cpu" && backend != "cuda") || (mode != "continuous" && mode != "interrupt" && mode != "resume")) {
    return 2;
  }
  const auto status = fluidity::Run(backend, mode, argv[3], argc == 5);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
}
