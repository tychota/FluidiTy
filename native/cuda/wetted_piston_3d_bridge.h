#pragma once

#include "fluidity/base/wetted_piston_math_3d.h"

namespace fluidity::cuda_internal {

/// Enqueues one piston-plane disk per center, in source order, and writes the
/// compiled architecture. Both device ranges cover source_count > 0 records.
/// Returns the CUDA launch error. Keep the disjoint ranges, architecture slot,
/// and stream live until the caller's completion fence.
int LaunchWettedPistonSections3D(
    wetted_piston_internal::Field field,
    const Vec3* device_centers,
    wetted_piston_internal::Disk* device_disks,
    std::size_t source_count,
    int* device_architecture,
    void* stream
);

/// Enqueues one bounded area and flux per patch, in patch order, and writes the
/// compiled architecture. The disk range covers source_count records. The
/// index and measure ranges cover patch_count > 0 records, up to the batch limit.
/// Each patch uses two interval banks with stride >= max(1, source_count), in
/// interval records. Returns the CUDA launch error. Keep the disjoint ranges,
/// architecture slot, and stream live until the caller's completion fence.
int LaunchWettedPistonPatches3D(
    wetted_piston_internal::Field field,
    const wetted_piston_internal::Disk* device_disks,
    std::size_t source_count,
    const wetted_piston_internal::PatchIndex* device_indices,
    wetted_piston_internal::PatchMeasure* device_measures,
    std::size_t patch_count,
    wetted_piston_internal::Interval* device_interval_banks,
    std::size_t interval_bank_stride,
    int* device_architecture,
    void* stream
);

}  // namespace fluidity::cuda_internal
