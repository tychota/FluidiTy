#include <cuda_runtime_api.h>

#include <iostream>

#include "fluidity/host/cuda_wetted_piston_3d.h"
#include "wetted_piston_3d_cases.h"

namespace fluidity {
namespace {
void CheckWettedPistonDevice() {
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  ASSERT_EQ(properties.major, 12);
  ASSERT_EQ(properties.minor, 0);
#if defined(FLUIDITY_QUALIFIED_CUDA_DEVICE)
  ASSERT_STREQ(properties.name, FLUIDITY_QUALIFIED_CUDA_DEVICE);
#endif
  std::cout << "CTEST_FULL_OUTPUT\nwetted_piston_device=" << properties.name << '\n';
}
template <typename Control>
void CheckCudaWettedPiston(Control control) {
  ASSERT_NO_FATAL_FAILURE(CheckWettedPistonDevice());
  auto arena = CudaLiquidArena3D::CreateWithWettedPiston(wetted_piston_test::kCapacity);
  ASSERT_TRUE(arena.ok()) << arena.status();
  ASSERT_EQ(arena->metrics().compiled_architecture, 1200);
  control([&](const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace) {
    return CudaComputeWettedPiston3D(*arena, inputs, workspace);
  });
}
TEST(CudaWettedPiston3D, ClipsHorizontalPlaneAndKeepsBothFluxSigns) {
  CheckCudaWettedPiston([](auto compute) {
    wetted_piston_test::PlaneClippingAndSignedFlux(compute, 1200);
  });
}
TEST(CudaWettedPiston3D, EnclosesThinOverlappingCapAndAttachesOnce) {
  CheckCudaWettedPiston([](auto compute) {
    wetted_piston_test::OverlappingThinCapEnclosesAnalyticArea(compute, 1200);
  });
}
TEST(CudaWettedPiston3D, CertifiesTangencyAndRejectsUncertainPositiveArea) {
  CheckCudaWettedPiston([](auto compute) {
    wetted_piston_test::TangencyAndUncertifiableThinWall(compute, 1200);
  });
}
TEST(CudaWettedPiston3D, RejectsDisconnectedAndUnownedWetRows) {
  CheckCudaWettedPiston([](auto compute) {
    wetted_piston_test::RefusesUnsupportedOwnershipAndKeepsPublication(compute);
  });
}
TEST(CudaWettedPiston3D, UsesSignedHalfOpenPistonCellAndChecksQueryCapacity) {
  CheckCudaWettedPiston([](auto compute) {
    wetted_piston_test::PreservesSignedLatticeAndChecksCapacity(compute, 1200);
  });
  auto workspace = WettedPistonWorkspace3D::Create(wetted_piston_test::kCapacity);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  auto arena = CudaLiquidArena3D::CreateWithWettedPiston(
      wetted_piston_test::kCapacity,
      0,
      std::numeric_limits<std::size_t>::max(),
      true
  );
  ASSERT_TRUE(arena.ok()) << arena.status();
  constexpr auto phase = LiquidWorkPhase3D::kForcesGeometryPressure;
  arena->SetWorkPhase(phase);
  const auto before = arena->metrics();
  auto regular_arena = CudaLiquidArena3D::Create(wetted_piston_test::kCapacity);
  ASSERT_TRUE(regular_arena.ok()) << regular_arena.status();
  constexpr auto intervals = CudaLiquidRange3D::kWettedPistonIntervals;
  const auto unreserved = regular_arena->range(intervals);
  EXPECT_TRUE(unreserved.pinned.empty());
  EXPECT_EQ(unreserved.device, nullptr);
  EXPECT_FALSE(arena->range(intervals).pinned.empty());
  EXPECT_LT(regular_arena->metrics().live_device_scratch_bytes, before.live_device_scratch_bytes);
  EXPECT_LT(regular_arena->metrics().live_pinned_bytes, before.live_pinned_bytes);
  const auto unprovisioned = wetted_piston_test::Fixture{}.Run(
      [&](const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& scratch) {
        return CudaComputeWettedPiston3D(*regular_arena, inputs, scratch);
      },
      LiquidHorizontalPlaneSurface3D{1},
      {},
      *workspace
  );
  EXPECT_EQ(unprovisioned.status().code(), absl::StatusCode::kFailedPrecondition);
  const auto rejected = wetted_piston_test::Fixture{}.Run(
      [&](const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& scratch) {
        return CudaComputeWettedPiston3D(*arena, inputs, scratch);
      },
      LiquidHorizontalPlaneSurface3D{34},
      {},
      *workspace
  );
  // The 34 queries complete two actual dispatches before the third wet row
  // fails the one-row attachment rule. Neither batch reserves resources.
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  const auto after = arena->metrics();
  const auto phase_index = static_cast<std::size_t>(phase);
  EXPECT_EQ(
      after.work_phase_events[phase_index].interval_count,
      before.work_phase_events[phase_index].interval_count + 2
  );
  EXPECT_EQ(after.device_reservations, before.device_reservations);
  EXPECT_EQ(after.pinned_reservations, before.pinned_reservations);
  EXPECT_EQ(after.event_creations, before.event_creations);
  EXPECT_EQ(after.stream_creations, before.stream_creations);

  auto tight_capacity = wetted_piston_test::kCapacity;
  tight_capacity.faces = 8;
  auto tight_arena = CudaLiquidArena3D::CreateWithWettedPiston(tight_capacity);
  ASSERT_TRUE(tight_arena.ok()) << tight_arena.status();
  const auto capacity = wetted_piston_test::Fixture{}.Run(
      [&](const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& scratch) {
        return CudaComputeWettedPiston3D(*tight_arena, inputs, scratch);
      },
      LiquidHorizontalPlaneSurface3D{9},
      {},
      *workspace
  );
  EXPECT_EQ(capacity.status().code(), absl::StatusCode::kResourceExhausted);
}
}  // namespace
}  // namespace fluidity
