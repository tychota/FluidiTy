#include <algorithm>

#include "../src/wetted_piston_3d_internal.h"
#include "fluidity/host/cuda_wetted_piston_3d.h"
#include "liquid_arena_dispatch_3d.h"
#include "wetted_piston_3d_bridge.h"

namespace fluidity {
namespace {
namespace wet = wetted_piston_internal;
namespace gpu = cuda_internal;
using Range = CudaLiquidRange3D;

absl::Status Status(absl::StatusCode code) {
  return absl::Status(code, {});
}

absl::Status
Finish(CudaLiquidArena3D& arena, const gpu::LiquidTypedRange3D<int>& architecture, int& compiled_architecture) {
  auto status = gpu::CopyLiquidRange3D(arena, architecture, false);
  if (!status.ok()) {
    return status;
  }
  status = gpu::FinishLiquidDispatch3D(arena);
  if (!status.ok()) {
    return status;
  }
  if (architecture.pinned.front() != 1200) {
    return Status(absl::StatusCode::kFailedPrecondition);
  }
  compiled_architecture = architecture.pinned.front();
  return absl::OkStatus();
}

absl::Status Sections(
    void* context,
    const wet::Field& field,
    std::span<const Vec3> world_centers,
    std::span<wet::Disk> measured_disks,
    int& compiled_architecture
) {
  if (world_centers.empty()) {
    return absl::OkStatus();
  }
  auto& arena = *static_cast<CudaLiquidArena3D*>(context);
  const auto centers = gpu::CheckedLiquidRange3D<Vec3>(arena, Range::kWettedPistonCenters, world_centers.size());
  const auto disks = gpu::CheckedLiquidRange3D<wet::Disk>(arena, Range::kWettedPistonDisks, measured_disks.size());
  const auto architecture = gpu::CheckedLiquidRange3D<int>(arena, Range::kArchitecture, 1);
  if (!centers.ok()) {
    return centers.status();
  }
  if (!disks.ok()) {
    return disks.status();
  }
  if (!architecture.ok()) {
    return architecture.status();
  }
  std::ranges::copy(world_centers, centers->pinned.begin());
  auto status = gpu::CopyLiquidRange3D(arena, *centers, true);
  if (!status.ok()) {
    return status;
  }
  status = arena.RecordRuntimeResult(gpu::LaunchWettedPistonSections3D(
      field,
      centers->device,
      disks->device,
      world_centers.size(),
      architecture->device,
      arena.stream()
  ));
  if (!status.ok()) {
    return status;
  }
  status = gpu::CopyLiquidRange3D(arena, *disks, false);
  if (!status.ok()) {
    return status;
  }
  status = Finish(arena, *architecture, compiled_architecture);
  if (!status.ok()) {
    return status;
  }
  std::ranges::copy(disks->pinned, measured_disks.begin());
  return absl::OkStatus();
}

absl::Status Patches(
    void* context,
    const wet::Field& field,
    std::span<const wet::Disk> sections,
    std::span<const wet::PatchIndex> patch_indices,
    std::span<wet::PatchMeasure> patch_measures,
    std::span<wet::Interval>,
    int& compiled_architecture
) {
  if (patch_indices.empty()) {
    return absl::OkStatus();
  }
  auto& arena = *static_cast<CudaLiquidArena3D*>(context);
  if (patch_indices.size() > arena.capacities().faces) {
    return Status(absl::StatusCode::kResourceExhausted);
  }
  const auto can_wet_wall = [](const wet::Disk& disk) {
    return disk.radius_squared.upper > 0;
  };
  const auto wet_disk_count = static_cast<std::size_t>(std::ranges::count_if(sections, can_wet_wall));
  const auto disks = gpu::CheckedLiquidRange3D<wet::Disk>(arena, Range::kWettedPistonDisks, wet_disk_count);
  const auto architecture = gpu::CheckedLiquidRange3D<int>(arena, Range::kArchitecture, 1);
  if (!disks.ok()) {
    return disks.status();
  }
  if (!architecture.ok()) {
    return architecture.status();
  }
  const auto interval_stride = std::max<std::size_t>(1, wet_disk_count);
  // The shared integrator skips certified zero-radius disks in every band.
  // Preserve the original disk order and reuse the resident list when all can wet.
  if (wet_disk_count > 0 && wet_disk_count < sections.size()) {
    std::ranges::copy_if(sections, disks->pinned.begin(), can_wet_wall);
  }
  for (std::size_t begin = 0; begin < patch_indices.size();) {
    const auto count = std::min(wet::kDevicePatchBatch, patch_indices.size() - begin);
    const auto indices = gpu::CheckedLiquidRange3D<wet::PatchIndex>(arena, Range::kWettedPistonIndices, count);
    const auto measures = gpu::CheckedLiquidRange3D<wet::PatchMeasure>(arena, Range::kWettedPistonMeasures, count);
    const auto intervals =
        gpu::CheckedLiquidRange3D<wet::Interval>(arena, Range::kWettedPistonIntervals, 2 * count * interval_stride);
    if (!indices.ok()) {
      return indices.status();
    }
    if (!measures.ok()) {
      return measures.status();
    }
    if (!intervals.ok()) {
      return intervals.status();
    }

    std::ranges::copy(patch_indices.subspan(begin, count), indices->pinned.begin());
    // Upload only after all first-batch ranges have been checked. The batch
    // fence protects the pinned disk span until the next call can overwrite it.
    if (begin == 0 && wet_disk_count < sections.size()) {
      const auto uploaded = gpu::CopyLiquidRange3D(arena, *disks, true);
      if (!uploaded.ok()) {
        return uploaded;
      }
    }
    auto status = gpu::CopyLiquidRange3D(arena, *indices, true);
    if (!status.ok()) {
      return status;
    }
    status = arena.RecordRuntimeResult(gpu::LaunchWettedPistonPatches3D(
        field,
        disks->device,
        wet_disk_count,
        indices->device,
        measures->device,
        count,
        intervals->device,
        interval_stride,
        architecture->device,
        arena.stream()
    ));
    if (!status.ok()) {
      return status;
    }
    status = gpu::CopyLiquidRange3D(arena, *measures, false);
    if (!status.ok()) {
      return status;
    }
    status = Finish(arena, *architecture, compiled_architecture);
    if (!status.ok()) {
      return status;
    }
    std::ranges::copy(measures->pinned, patch_measures.subspan(begin, count).begin());
    begin += count;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<WettedPistonView3D> CudaComputeWettedPiston3D(
    CudaLiquidArena3D& arena,
    const WettedPistonInputs3D& inputs,
    WettedPistonWorkspace3D& workspace
) {
  const auto ready = arena.RecordRuntimeResult(cudaSuccess);
  if (!ready.ok()) {
    return ready;
  }
  if (arena.range(Range::kWettedPistonCenters).pinned.empty()) {
    return Status(absl::StatusCode::kFailedPrecondition);
  }
  const auto& frozen = inputs.frozen;
  const auto capacity = arena.capacities();
  if (inputs.particles.size() > capacity.particles || frozen.cells.size() > capacity.cells
      || frozen.faces.size() > capacity.faces) {
    return Status(absl::StatusCode::kResourceExhausted);
  }
  if (gpu::AliasesLiquidScratch3D(arena, inputs.particles) || gpu::AliasesLiquidScratch3D(arena, frozen.faces)
      || gpu::AliasesLiquidScratch3D(arena, frozen.cells)
      || gpu::AliasesLiquidScratch3D(arena, frozen.mac_grid.active_block_indices())
      || gpu::AliasesLiquidScratch3D(arena, std::span{&inputs, 1})
      || gpu::AliasesLiquidScratch3D(arena, std::span{&inputs.field, 1})
      || gpu::AliasesLiquidScratch3D(arena, std::span{&frozen, 1})) {
    return Status(absl::StatusCode::kInvalidArgument);
  }
  return wet::Compute(inputs, workspace, {.context = &arena, .sections = Sections, .patches = Patches});
}

}  // namespace fluidity
