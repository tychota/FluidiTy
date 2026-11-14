#include <algorithm>
#include <cstdint>
#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/processor_description.h"
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

#include <cstddef>
#elif defined(__linux__)
#include <sys/sysinfo.h>

#include <charconv>
#include <fstream>
#include <string>
#include <string_view>
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

#if defined(__APPLE__)

// Apple computers: the physical memory from sysctl, and the free and inactive pages from the virtual memory counters.
// Inactive pages hold files read earlier and memory that programs have not touched for a while. The operating
// software can usually hand them to a program that asks, so they count as free here; the reading is an estimate,
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

// Linux: the physical memory from sysinfo, and MemAvailable, the operating software's own estimate of what a program
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

#endif

}  // namespace

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

}  // namespace fluidity
