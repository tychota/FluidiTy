#pragma once

#include "fluidity/host/wetted_piston_3d.h"

namespace fluidity::wetted_piston_internal {

// Synchronous stage callbacks run in section-then-patch order. Their output
// spans are unpublished scratch. A non-OK status discards the attempt's scratch
// results. The callback must not retain borrowed spans after returning.
struct Backend {
  // Encloses one piston-plane disk section for each world center in source
  // order. The two spans have equal size.
  using MeasureSections = absl::Status(
      void* context,
      const Field& field,
      std::span<const Vec3> world_centers,
      std::span<Disk> disk_sections,
      int& compiled_architecture
  );
  // Produces one measure for each patch coordinate in request order. Disk
  // sections preserve source order. On CPU, interval_banks contains two
  // consecutive banks of disk_sections.size() records each. A CUDA backend
  // uses its own reserved interval banks and may reuse staged device sections.
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

// Runs checked wetted-piston publication with a supplied backend. Both
// callbacks must be callable for backend.context. The workspace owns the
// successful borrow. A callback error preserves the preceding publication.
absl::StatusOr<WettedPistonView3D>
Compute(const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace, Backend backend);

}  // namespace fluidity::wetted_piston_internal
