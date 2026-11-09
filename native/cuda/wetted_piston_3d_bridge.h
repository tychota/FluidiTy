#pragma once

#include "fluidity/base/wetted_piston_math_3d.h"

namespace fluidity::cuda_internal {

// Device ranges and stream remain live through the caller's fence. Each
// launch publishes its actual compiled architecture in the reserved slot.
int LaunchWettedPistonSections3D(
    wetted_piston_internal::Field field,
    const Vec3* device_centers,
    wetted_piston_internal::Disk* device_disks,
    std::size_t source_count,
    int* device_architecture,
    void* stream
);

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
