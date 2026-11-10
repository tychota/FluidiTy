#include "simulate_cuda_compatibility.h"

#include <nvml.h>

#include <algorithm>
#include <array>
#include <bit>

#include "../cuda/liquid_device_3d_internal.h"
#include "../src/liquid_cache_files_3d_internal.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace fluidity::simulate {
namespace {
namespace files = liquid_cache_files_internal;

constexpr char kCompatibilityFile[] = ".simulate-cuda-compatibility";
constexpr std::size_t kHexDigestLength = 64;
constexpr std::size_t kRecordLength = kCudaCompatibilityRecordPrefix.size() + kHexDigestLength + 1;
constexpr LiquidCacheLimits3D kRecordLimits{.maximum_bytes = kRecordLength};

// The NVIDIA management library is loaded only during CUDA execution admission.
class NvmlLibrary {
 public:
  NvmlLibrary() = default;
  NvmlLibrary(const NvmlLibrary&) = delete;
  NvmlLibrary& operator=(const NvmlLibrary&) = delete;
  ~NvmlLibrary() {
    if (handle_ != nullptr) {
#if defined(_WIN32)
      FreeLibrary(handle_);
#else
      dlclose(handle_);
#endif
    }
  }

  template <typename FunctionPointer>
  FunctionPointer Find(const char* name) const {
    if (handle_ == nullptr) {
      return nullptr;
    }
#if defined(_WIN32)
    return std::bit_cast<FunctionPointer>(GetProcAddress(handle_, name));
#else
    return std::bit_cast<FunctionPointer>(dlsym(handle_, name));
#endif
  }

 private:
#if defined(_WIN32)
  HMODULE handle_ = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
  void* handle_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
};

absl::StatusOr<std::string> NvidiaDriverRelease() {
  const NvmlLibrary library;
  const auto initialize = library.Find<decltype(&nvmlInit_v2)>("nvmlInit_v2");
  const auto get_driver_release = library.Find<decltype(&nvmlSystemGetDriverVersion)>("nvmlSystemGetDriverVersion");
  const auto shutdown = library.Find<decltype(&nvmlShutdown)>("nvmlShutdown");
  if (initialize == nullptr || get_driver_release == nullptr || shutdown == nullptr) {
    return absl::UnavailableError("");
  }
  if (initialize() != NVML_SUCCESS) {
    return absl::UnavailableError("");
  }
  std::array<char, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE> release{};
  const auto query_status = get_driver_release(release.data(), static_cast<unsigned int>(release.size()));
  const auto shutdown_status = shutdown();
  if (query_status != NVML_SUCCESS || shutdown_status != NVML_SUCCESS) {
    return absl::UnavailableError("");
  }
  const auto end = std::ranges::find(release, '\0');
  if (end == release.begin() || end == release.end()) {
    return absl::UnavailableError("");
  }
  return std::string(release.begin(), end);
}

bool IsLowercaseHex(char character) {
  return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
}
}  // namespace

absl::StatusOr<CudaExecutionIdentity>
QueryCudaExecutionIdentity(int device_ordinal, const LiquidDigest3D& resolved_fingerprint) {
  auto properties = cuda_internal::QueryLiquidCudaDevice3D(device_ordinal);
  if (!properties.ok()) {
    return properties.status();
  }
  if (std::ranges::all_of(properties->uuid.bytes, [](char value) {
        return value == 0;
      })) {
    return absl::UnavailableError("");
  }
  int driver_api_version = 0;
  auto status = cuda_internal::LiquidCudaDeviceSetupStatus3D(cudaDriverGetVersion(&driver_api_version));
  if (!status.ok()) {
    return status;
  }
  int runtime_api_version = 0;
  status = cuda_internal::LiquidCudaDeviceSetupStatus3D(cudaRuntimeGetVersion(&runtime_api_version));
  if (!status.ok()) {
    return status;
  }
  if (driver_api_version <= 0 || runtime_api_version <= 0) {
    return absl::UnavailableError("");
  }
  auto driver_release = NvidiaDriverRelease();
  if (!driver_release.ok()) {
    return driver_release.status();
  }

  // The Windows hybrid runtime comes from the display driver. Bind its
  // exact release and both reported API versions to the admission sample.
  const CudaExecutionIdentityInputs inputs{
      .uuid = std::to_array(properties->uuid.bytes),
      .compute_major = properties->major,
      .compute_minor = properties->minor,
      .driver_release = *driver_release,
      .driver_api_version = driver_api_version,
      .runtime_api_version = runtime_api_version,
      .compiled_runtime_version = CUDART_VERSION,
  };
  return BuildCudaExecutionIdentity(inputs, resolved_fingerprint);
}

absl::StatusOr<std::string> ReadCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory) {
  auto bytes = files::ReadBytes(verified_cache_directory / kCompatibilityFile, kRecordLimits, kRecordLength);
  if (absl::IsNotFound(bytes.status())) {
    return absl::FailedPreconditionError("");
  }
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto values = bytes->values();
  const std::string_view record(reinterpret_cast<const char*>(values.data()), values.size());
  if (!record.starts_with(kCudaCompatibilityRecordPrefix) || record.back() != '\n'
      || !std::ranges::all_of(record.substr(kCudaCompatibilityRecordPrefix.size(), kHexDigestLength), IsLowercaseHex)) {
    return absl::DataLossError("");
  }
  return std::string(record);
}

absl::Status
PublishCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory, std::string_view record) {
  return files::PublishBytes(
      verified_cache_directory / kCompatibilityFile,
      files::Bytes(record),
      kRecordLimits,
      {},
      absl::StatusCode::kFailedPrecondition
  );
}

}  // namespace fluidity::simulate
