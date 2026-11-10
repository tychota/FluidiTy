#pragma once

#include "fluidity/host/wetted_piston_3d.h"

namespace fluidity::wetted_piston_internal {

// Backend functions evaluate sections once, then bounded batches of patches.
// CPU uses the supplied interval pair. A device backend reserves its own
// pairs and keeps its sections live between the two operations.
struct Backend {
  using MeasureSections = absl::Status(
      void* context,
      const Field& field,
      std::span<const Vec3> world_centers,
      std::span<Disk> disk_sections,
      int& compiled_architecture
  );
  using MeasurePatches = absl::Status(
      void* context,
      const Field& field,
      std::span<const Disk> disk_sections,
      std::span<const PatchIndex> patch_indices,
      std::span<PatchMeasure> patch_measures,
      std::span<Interval> interval_banks,
      int& compiled_architecture
  );
  void* context = nullptr;
  MeasureSections* sections = nullptr;
  MeasurePatches* patches = nullptr;
};

absl::StatusOr<WettedPistonView3D>
Compute(const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace, Backend backend);

}  // namespace fluidity::wetted_piston_internal
