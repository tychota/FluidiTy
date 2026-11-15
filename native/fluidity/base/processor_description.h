// What a processor reports about itself: how much memory it has and how much is free, and its name and versions.
//
// Purpose: let a caller compare the memory a run needs with the memory the chosen processor has, before it builds
//   the run's buffers, and let a run report say which processor and which software produced a result.
//
// Idea: the CPU answer comes from the operating system of the machine (on Apple computers, Linux or Windows):
//   the physical memory and the part of it that programs can take now. The GPU answer comes from the CUDA runtime
//   for the current GPU. Free memory changes all the time, so the answer is a reading at one moment, not a
//   promise. The description is three strings exactly as the machine reports them, for people to read, not for
//   code to parse.
//
// Owns:    MemoryAvailable, ReadMemoryAvailable, ProcessorDescription, ReadProcessorDescription
// Uses:    RunOn (base/run_on.h), BufferError (base/buffer.h), CPU-only guard (base/cpu_only.h)
// Implementation uses: CheckCanRunOn (base/check_can_run_on.h), FindCpuModelInText
//          (base/processor_description_text.h) and, in a build with GPU code, the private GPU half
//          (base/processor_description_gpu.h)
// Used by: the step's memory plan, the command line, the run report
// Read:    docs/method-paper/fluidity-method.typ <sec-memory-estimate>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>
#include <string>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {

// The memory of one processor at one moment.
struct MemoryAvailable {
  std::uint64_t total_bytes = 0;  // Bytes. All the memory of the processor. 0 means unknown.
  std::uint64_t free_bytes = 0;  // Bytes. The part a program could take now, at most total_bytes. 0 when unknown.
};

// Reads the total and the free memory of the processor `run_on`.
//
// Method: for RunOn::kCpu, asks the operating system for the physical memory and the memory programs can take
//   now (Apple computers: free and inactive pages; Linux: MemAvailable of /proc/meminfo, which ignores a container's
//   memory limit; Windows: available physical memory). When that query fails, total_bytes and free_bytes are 0, meaning
//   unknown; it is not an error. For RunOn::kGpu, calls CheckCanRunOn(RunOn::kGpu) first, then makes sure the GPU
//   context exists (cudaFree(nullptr)) and only then asks the CUDA runtime about the current GPU: the context and the
//   GPU code loaded so far take memory, and reading after they exist keeps them outside "free". GPU code loads when
//   it is first launched, so code launched later may still take a little. The step's run plan computes the allowed
//   bytes of a run from this free reading. Free memory is a reading at one moment: other programs take and give back
//   memory, and the CUDA runtime does not promise that all of it can be allocated.
// Example: a machine with 32 GiB of which 20 GiB can be taken gives {34359738368, 21474836480}.
// Fails: kRunOnCallFailed RunOn::kGpu was asked for and CheckCanRunOn failed (also in a build without GPU code),
//                         or creating the context or the CUDA runtime's reading failed
//        Never fails for RunOn::kCpu. On failure no reading is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-memory-estimate>
std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on);

// The name and the software versions of one processor, as text for the run report. A field the machine cannot
// answer is an empty string.
struct ProcessorDescription {
  // The processor's name. CPU: the brand string, such as "Apple M3 Pro" or "AMD Ryzen 9 7950X";
  // on Linux machines that report none (ARM), the machine architecture name, such as "aarch64". GPU: the device
  // name, such as "NVIDIA GeForce RTX 5060 Ti".
  std::string model;
  // The software that talks to the processor. CPU: the operating system name and version, such as
  // "Darwin 25.5.0" or "Linux 6.8.0". GPU: the driver version as "major.minor", such as "12.8".
  std::string driver;
  // The software the program was built with. CPU: the compiler name and version, such as "Clang 19.1.5".
  // GPU: the CUDA runtime version as "major.minor", such as "12.8".
  std::string runtime;
};

// Reads the name and software versions of the processor `run_on`.
//
// Method: for RunOn::kCpu, asks the operating system for the brand string and its own version, and takes the
//   compiler version from the compiler's own macros. For RunOn::kGpu, calls CheckCanRunOn(RunOn::kGpu) first,
//   then asks the CUDA runtime about the current GPU. Each field is read on its own, so one failed query leaves
//   only that field empty.
// Example: on an Apple computer, {"Apple M3 Pro", "Darwin 25.5.0", "Clang 19.1.5"}.
// Fails: never. Any query that fails gives an empty string for that field. RunOn::kGpu in a build without GPU
//        code, or on a machine without a usable GPU, gives three empty strings.
// Read: docs/method-paper/fluidity-method.typ <sec-memory-estimate>
ProcessorDescription ReadProcessorDescription(RunOn run_on);

}  // namespace fluidity
