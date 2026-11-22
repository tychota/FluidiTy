#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

#include "fluidity/base/buffer.h"
#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/processor_description.h"
#include "fluidity/base/processor_description_text.h"
#include "fluidity/base/run_on.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/processor_description_gpu.h"
#include "fluidity/base/run_on_call.h"
#endif

#if defined(__APPLE__)
#include <mach/host_info.h>
#include <mach/kern_return.h>
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/message.h>
#include <mach/vm_page_size.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/utsname.h>

#include <charconv>
#include <fstream>
#include <iterator>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace fluidity {
namespace {

// The compiler's name and version from its own macros. Clang is tested before GCC because it defines both.
std::string ReadCompilerText() {
#if defined(__clang__)
  return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "."
      + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "."
      + std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_FULL_VER);
#else
  return "";
#endif
}

// `text` without leading and trailing spaces and tabs. Some processors pad the brand string with spaces.
std::string TrimSpaces(const std::string& text) {
  const std::size_t text_start = text.find_first_not_of(" \t");
  if (text_start == std::string::npos) {
    return "";
  }
  const std::size_t text_end = text.find_last_not_of(" \t");
  return text.substr(text_start, text_end - text_start + 1);
}

#if defined(__APPLE__) || defined(__linux__)

// The operating system's own name and version, such as "Darwin 25.5.0" or "Linux 6.8.0", or "" when uname fails.
std::string ReadOperatingSystemText() {
  struct utsname kernel_names{};
  if (uname(&kernel_names) != 0) {
    return "";
  }
  return std::string(kernel_names.sysname) + " " + kernel_names.release;
}

#endif

#if defined(__APPLE__)

// The brand string of the processor, such as "Apple M3 Pro", or "" when sysctl cannot answer.
std::string ReadCpuModelText() {
  std::size_t text_size = 0;
  if (sysctlbyname("machdep.cpu.brand_string", nullptr, &text_size, nullptr, 0) != 0 || text_size == 0) {
    return "";
  }
  std::string text(text_size, '\0');
  if (sysctlbyname("machdep.cpu.brand_string", text.data(), &text_size, nullptr, 0) != 0) {
    return "";
  }
  // The size sysctl wrote counts the terminating null character.
  text.resize(text_size - 1);
  return text;
}

// Apple computers: the physical memory from sysctl, and the free and inactive pages from the virtual memory counters.
// Inactive pages hold files read earlier and memory that programs have not touched for a while. The operating
// system can usually hand them to a program that asks, so they count as free here; the reading is an estimate,
// and some of that memory may first have to be written out.
MemoryAvailable ReadCpuMemoryAvailable() {
  std::uint64_t total_bytes = 0;
  std::size_t total_size = sizeof(total_bytes);
  if (sysctlbyname("hw.memsize", &total_bytes, &total_size, nullptr, 0) != 0) {
    return {};
  }
  vm_statistics64_data_t page_counts{};
  mach_msg_type_number_t count_size = HOST_VM_INFO64_COUNT;
  // auto: the port and status types come from private headers of the platform, which no file should include.
  const auto host_port = mach_host_self();
  const auto status =
      host_statistics64(host_port, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&page_counts), &count_size);
  // mach_host_self gives a new right to the port each time; giving it back keeps the process from leaking rights.
  mach_port_deallocate(mach_task_self(), host_port);
  if (status != KERN_SUCCESS) {
    return {};
  }
  const std::uint64_t free_pages = std::uint64_t{page_counts.free_count} + page_counts.inactive_count;
  const std::uint64_t free_bytes = free_pages * vm_page_size;
  return {
      .total_bytes = total_bytes,
      .free_bytes = std::min(free_bytes, total_bytes),
  };
}

#elif defined(__linux__)

// The number of kibibytes on the line of /proc/meminfo that starts with `key`, such as "MemAvailable:", or 0 when
// the line is missing or unreadable.
std::uint64_t ReadMeminfoKibibytes(std::string_view key) {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    if (!line.starts_with(key)) {
      continue;
    }
    const std::size_t digits_start = line.find_first_of("0123456789", key.size());
    if (digits_start == std::string::npos) {
      return 0;
    }
    std::uint64_t kibibytes = 0;
    const char* const digits_end = line.data() + line.size();
    const std::from_chars_result parsed = std::from_chars(line.data() + digits_start, digits_end, kibibytes);
    if (parsed.ec != std::errc{}) {
      return 0;
    }
    return kibibytes;
  }
  return 0;
}

// The first "model name" of /proc/cpuinfo, such as "AMD Ryzen 9 7950X". No such line, or an empty one, so the machine
// architecture name from uname, such as "aarch64", stands in. "" when both fail.
std::string ReadCpuModelText() {
  std::ifstream cpuinfo_file("/proc/cpuinfo");
  const std::string cpuinfo_text((std::istreambuf_iterator<char>(cpuinfo_file)), std::istreambuf_iterator<char>());
  std::string model = FindCpuModelInText(cpuinfo_text);
  if (!model.empty()) {
    return model;
  }
  struct utsname kernel_names{};
  if (uname(&kernel_names) != 0) {
    return "";
  }
  return kernel_names.machine;
}

// Linux: the physical memory from sysinfo, and MemAvailable, the operating system's own estimate of what a program
// can take without swapping. Both describe the whole machine: inside a container with a memory limit, the limit is
// ignored and the reading can be larger than what this program may take. Linux before version 3.14 lacks MemAvailable;
// free plus buffer memory is then the estimate.
MemoryAvailable ReadCpuMemoryAvailable() {
  struct sysinfo memory_counts{};
  if (sysinfo(&memory_counts) != 0) {
    return {};
  }
  const std::uint64_t unit_bytes = memory_counts.mem_unit;
  const std::uint64_t total_bytes = std::uint64_t{memory_counts.totalram} * unit_bytes;
  std::uint64_t free_bytes = ReadMeminfoKibibytes("MemAvailable:") * 1024;
  if (free_bytes == 0) {
    free_bytes = (std::uint64_t{memory_counts.freeram} + memory_counts.bufferram) * unit_bytes;
  }
  return {
      .total_bytes = total_bytes,
      .free_bytes = std::min(free_bytes, total_bytes),
  };
}

#elif defined(_WIN32)

// The text value `value_name` under HKEY_LOCAL_MACHINE\\`key_path`, or "" when the registry cannot answer.
std::string ReadRegistryText(const char* key_path, const char* value_name) {
  char text[256] = {};
  DWORD text_size = sizeof(text);
  if (RegGetValueA(HKEY_LOCAL_MACHINE, key_path, value_name, RRF_RT_REG_SZ, nullptr, text, &text_size)
      != ERROR_SUCCESS) {
    return "";
  }
  return text;
}

// The processor's name, such as "AMD Ryzen 9 7950X", from the registry.
std::string ReadCpuModelText() {
  return TrimSpaces(ReadRegistryText("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString"));
}

// The Windows product name and build number, such as "Windows 10 Pro 19045". ProductName still reads "Windows 10" on
// Windows 11, so the build number tells them apart: Windows 11 starts at build 22000, such as "Windows 10 Pro 26100".
std::string ReadOperatingSystemText() {
  const char* const version_key = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
  const std::string product_name = ReadRegistryText(version_key, "ProductName");
  const std::string build_number = ReadRegistryText(version_key, "CurrentBuildNumber");
  if (product_name.empty()) {
    return build_number;
  }
  return product_name + " " + build_number;
}

// Windows: the physical memory and the available physical memory of GlobalMemoryStatusEx.
MemoryAvailable ReadCpuMemoryAvailable() {
  MEMORYSTATUSEX memory_status{};
  memory_status.dwLength = sizeof(memory_status);
  if (GlobalMemoryStatusEx(&memory_status) == 0) {
    return {};
  }
  return {
      .total_bytes = memory_status.ullTotalPhys,
      .free_bytes = std::min<std::uint64_t>(memory_status.ullAvailPhys, memory_status.ullTotalPhys),
  };
}

#else

// Another platform: unknown, which is not an error.
MemoryAvailable ReadCpuMemoryAvailable() {
  return {};
}

std::string ReadCpuModelText() {
  return "";
}

std::string ReadOperatingSystemText() {
  return "";
}

#endif

}  // namespace

std::string FindCpuModelInText(std::string_view cpuinfo_text) {
  constexpr std::string_view kModelNameKey = "model name";
  while (!cpuinfo_text.empty()) {
    const std::size_t line_end = std::min(cpuinfo_text.find('\n'), cpuinfo_text.size());
    const std::string_view line = cpuinfo_text.substr(0, line_end);
    cpuinfo_text.remove_prefix(std::min(line_end + 1, cpuinfo_text.size()));
    if (!line.starts_with(kModelNameKey)) {
      continue;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos) {
      continue;
    }
    return TrimSpaces(std::string(line.substr(colon + 1)));
  }
  return "";
}

std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on) {
  if (run_on == RunOn::kCpu) {
    return ReadCpuMemoryAvailable();
  }
  if (!CheckCanRunOn(RunOn::kGpu).has_value()) {
    // No GPU code, no GPU, or a GPU whose math differs: there is no GPU memory to report.
    return std::unexpected(BufferError::kRunOnCallFailed);
  }
#if defined(FLUIDITY_HAS_GPU)
  MemoryAvailable memory;
  if (const RunOnCallStatus status = ReadGpuMemoryAvailable(&memory.total_bytes, &memory.free_bytes);
      status != RunOnCallStatus::kDone) {
    // The GPU passed the check but the runtime cannot answer now: the driver failed after the check.
    return std::unexpected(ToBufferError(status));
  }
  return memory;
#else
  // Not reached: CheckCanRunOn fails for the GPU in a build without GPU code.
  return std::unexpected(BufferError::kRunOnCallFailed);
#endif
}

ProcessorDescription ReadProcessorDescription(RunOn run_on) {
  if (run_on == RunOn::kCpu) {
    return {
        .model = ReadCpuModelText(),
        .driver = ReadOperatingSystemText(),
        .runtime = ReadCompilerText(),
    };
  }
  if (!CheckCanRunOn(RunOn::kGpu).has_value()) {
    // No GPU code, no GPU, or a GPU whose math differs: there is nothing to describe.
    return {};
  }
#if defined(FLUIDITY_HAS_GPU)
  constexpr std::size_t kTextSize = 256;
  char model[kTextSize] = {};
  char driver[kTextSize] = {};
  char runtime[kTextSize] = {};
  ReadGpuProcessorDescription(model, driver, runtime, kTextSize);
  return {.model = model, .driver = driver, .runtime = runtime};
#else
  // Not reached: CheckCanRunOn fails for the GPU in a build without GPU code.
  return {};
#endif
}

}  // namespace fluidity
