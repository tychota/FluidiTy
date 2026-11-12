#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>

#include "fluidity/host/liquid_cache_3d.h"
#include "gtest/gtest.h"
#include "mac_pressure_explicit_cases.h"

namespace fluidity::pressure_test {

// Exercise the same pressure operator on both sides of an authenticated
// checkpoint. This is a pressure fixture using the host acceptance transaction.
template <typename Project>
void QualifyAdaptivePressureCheckpoint(Project project) {
  const auto fixture = MakeAdaptiveCutPressureFixture();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  LiquidState3D initial;
  initial.geometry.cell_width = fixture->topology.finest_cell_width();
  auto leaves = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(fixture->cells.size());
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(fixture->subfaces.size());
  ASSERT_TRUE(leaves.ok()) << leaves.status();
  ASSERT_TRUE(faces.ok()) << faces.status();
  initial.adaptive_cells = std::move(*leaves);
  initial.adaptive_faces = std::move(*faces);
  ASSERT_TRUE(initial.adaptive_cells.Resize(fixture->cells.size()).ok());
  ASSERT_TRUE(initial.adaptive_faces.Resize(fixture->subfaces.size()).ok());
  for (std::size_t cell = 0; cell < fixture->cells.size(); ++cell) {
    const double volume = fixture->cells[cell].fluid_volume;
    initial.adaptive_cells[cell] = {
        .key = fixture->topology.cells()[cell].key,
        .inventory = {.mass_kg = 2 * volume, .reference_volume_m3 = volume, .current_volume_m3 = volume},
    };
  }
  LiquidResumeMetadata3D resume;
  constexpr std::array<std::uint8_t, 1> fingerprint_input{42};
  resume.resolved_fingerprint = LiquidContentDigest3D(fingerprint_input);
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  resume.capacities.cells = fixture->cells.size();
  resume.capacities.faces = fixture->subfaces.size();

  const auto attempt = [&](LiquidFrameTransaction3D& run, double dt, LiquidAttemptDisposition3D disposition) {
    const auto trial = run.ResetTrialFromCommitted();
    ASSERT_TRUE(trial.ok()) << trial.status();
    auto pressure_cells = fixture->cells;
    for (std::size_t cell = 0; cell < pressure_cells.size(); ++cell) {
      pressure_cells[cell].previous_pressure = (*trial)->adaptive_cells[cell].previous_pressure;
    }
    const auto graph =
        pressure_internal::PrepareAdaptiveMacPressureSystem3D(fixture->topology, dt, pressure_cells, fixture->subfaces);
    ASSERT_TRUE(graph.ok()) << graph.status();
    const auto projection = project(*graph);
    ASSERT_TRUE(projection.ok()) << projection.status();
    ASSERT_EQ(projection->cell_pressures.size(), (*trial)->adaptive_cells.size());
    ASSERT_EQ(projection->corrected_face_velocities.size(), (*trial)->adaptive_faces.size());
    for (std::size_t cell = 0; cell < pressure_cells.size(); ++cell) {
      (*trial)->adaptive_cells[cell].previous_pressure = projection->cell_pressures[cell];
    }
    for (std::size_t face = 0; face < (*trial)->adaptive_faces.size(); ++face) {
      (*trial)->adaptive_faces[face].carrier_velocity = projection->corrected_face_velocities[face];
      (*trial)->adaptive_faces[face].validity = LiquidFaceValidity3D::kValid;
    }
    if (disposition == LiquidAttemptDisposition3D::kRetryDt) {
      (*trial)->adaptive_cells[0].inventory.mass_kg += 1;
    }
    const auto result = run.FinishAttempt(disposition, dt);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->disposition, disposition);
  };

  auto continuous = LiquidFrameTransaction3D::Create(initial, resume.capacities);
  ASSERT_TRUE(continuous.ok()) << continuous.status();
  ASSERT_TRUE(continuous->BeginFrame(1).ok());
  ASSERT_NO_FATAL_FAILURE(attempt(*continuous, 1, LiquidAttemptDisposition3D::kRetryDt));
  EXPECT_EQ(continuous->committed(), initial);
  ASSERT_NO_FATAL_FAILURE(attempt(*continuous, 0.5, LiquidAttemptDisposition3D::kAccepted));
  ASSERT_NO_FATAL_FAILURE(attempt(*continuous, 0.5, LiquidAttemptDisposition3D::kAccepted));
  ASSERT_TRUE(continuous->FinishFrame().ok());
  const auto boundary = EncodeLiquidCheckpoint3D(continuous->committed(), resume);
  ASSERT_TRUE(boundary.ok()) << boundary.status();
  const auto restored = DecodeLiquidCheckpoint3D(boundary->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto restarted = LiquidFrameTransaction3D::Create(restored->state, restored->resume.capacities);
  ASSERT_TRUE(restarted.ok()) << restarted.status();
  for (auto* run : {&*continuous, &*restarted}) {
    ASSERT_TRUE(run->BeginFrame(2).ok());
    ASSERT_NO_FATAL_FAILURE(attempt(*run, 1, LiquidAttemptDisposition3D::kAccepted));
    ASSERT_TRUE(run->FinishFrame().ok());
  }
  const auto uninterrupted = EncodeLiquidCheckpoint3D(continuous->committed(), resume);
  const auto interrupted = EncodeLiquidCheckpoint3D(restarted->committed(), resume);
  ASSERT_TRUE(uninterrupted.ok()) << uninterrupted.status();
  ASSERT_TRUE(interrupted.ok()) << interrupted.status();
  EXPECT_TRUE(std::ranges::equal(uninterrupted->values(), interrupted->values()));
  std::cout << "adaptive_pressure_checkpoint_digest="
            << LiquidDigestHex3D(LiquidContentDigest3D(interrupted->values())).data() << '\n';
}

}  // namespace fluidity::pressure_test
