#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <string_view>

#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity::simulate {

inline constexpr std::string_view kCudaCompatibilityRecordPrefix = "FLCUDA1\n";

// CUDA identity fields from a single admission sample. The borrowed release
// string is the exact NVIDIA driver release. API versions use CUDA integer
// encoding, and the compiled runtime version is CUDART_VERSION.
struct CudaExecutionIdentityInputs {
  std::array<char, 16> uuid;  // Raw CUDA device UUID bytes in API order.
  int compute_major;
  int compute_minor;
  std::string_view driver_release;
  int driver_api_version;
  int runtime_api_version;
  int compiled_runtime_version;
};

struct CudaExecutionIdentity {
  std::string compatibility_record;  // Exact FLCUDA1 bytes for this resolved scene.
  std::string device_digest;  // Lowercase hex BLAKE3-256 over device and runtime identity.
};

// Encode the admitted identity as a scene-bound record and a scene-independent
// digest. Uses the driver release view only during this call and never queries CUDA.
CudaExecutionIdentity
BuildCudaExecutionIdentity(const CudaExecutionIdentityInputs& inputs, const LiquidDigest3D& resolved_fingerprint);

// Query the selected CUDA device and exact driver/runtime without launching
// numerical work. Fails closed if CUDA or the exact driver release is unavailable.
absl::StatusOr<CudaExecutionIdentity>
QueryCudaExecutionIdentity(int device_ordinal, const LiquidDigest3D& resolved_fingerprint);

// Read a record in an already verified cache directory without creating files.
// Missing records return FailedPrecondition and malformed records return DataLoss.
absl::StatusOr<std::string> ReadCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory);

// Flush and publish the record without replacement in a verified cache directory.
// The caller holds the job lease. A different existing record fails with
// FailedPrecondition; exact reuse requires the same durability barrier.
absl::Status
PublishCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory, std::string_view record);

}  // namespace fluidity::simulate
