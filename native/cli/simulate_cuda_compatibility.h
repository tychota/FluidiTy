#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity::simulate {

// Build the versioned digest record for the selected CUDA device and exact
// resolved execution. Queries identity without launching any numerical work.
// Fails closed if CUDA or the exact driver release is unavailable. The result
// contains only a domain-separated digest and format marker.
absl::StatusOr<std::string> CudaCompatibilityRecord(int device_ordinal, const LiquidDigest3D& resolved_fingerprint);

// Read a record in an already verified cache directory without creating files.
// Missing records return FailedPrecondition and malformed records return DataLoss.
absl::StatusOr<std::string> ReadCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory);

// Flush and publish the record without replacement in a verified cache directory.
// The caller holds the job lease. A different existing record fails with
// FailedPrecondition; exact reuse requires the same durability barrier.
absl::Status
PublishCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory, std::string_view record);

}  // namespace fluidity::simulate
