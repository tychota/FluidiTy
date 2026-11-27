# Shared interface of the module designs

Status: final cross-module reference, rewritten on 2026-11-14 after the consistency sweep of the fifteen
module designs, and brought up to date the same day with Foundation as built and with `plan-decisions.md`
(including the reconcile pass, A24 and A25: g.2, g.7, g.8; A26 and A27: g.9), and on 2026-11-15 with
`plan-decisions.md` "Owner answers A40–A64 applied (2026-11-15)" (g.10; rows marked "(A<n>, 2026-11-15)") and
"Owner answers A65–A68 applied (2026-11-15)" (g.11, which also adds the rows the A40–A64 pass missed). Not
committed to the `rebuild` branch.

It replaces the first version, written from the reconciliation of the first six designs. It records what every
design shares: Foundation's `base/` as built (a), the cross-module names in their final form (b), what each
module provides and to whom (c), the material-set rule (d), the step's stage list (e), the build order (f) and
the shared declarations decided while writing the plans (g). The order of authority is `owner-answers.md`, then,
for `base/`, the built headers on the `rebuild` branch; otherwise `plan-decisions.md`, then `changes-to-apply.md`
(its final table "Names changed while the first designs were updated", later rows newest), then the owning
module's section 3. Where this file and those disagree, this file has a mistake.

Rules every design follows:

1. The module that owns a type or function decides its name and shape; consumers use that spelling.
2. What several modules need and none owns is in `base/` (section a).
3. An error with context is `std::expected<T, <Module>Failure>` with `struct <Module>Failure { <Module>Error
   error; ...context... }` (A2). Every error enum starts with `kNone`; "the chosen processor cannot be used, or
   a call to it failed" is `kRunOnCallFailed`; an input bound to another grid version is `kWrongGridVersion`.
   Every module provides `std::string_view ToMessage(<Module>Error)` (A13) and `std::string_view
   ToName(<Module>Error)` (the enumerator's name without `k`, in lower snake case: `solid_too_thin`), the stable
   code the CLI and the add-on use. `std::expected` is used directly, with no alias. Every module error enum ends
   with `kOutOfMemory`, appended as its last value, returned only by that module's building functions (and what
   `BufferError::kRunMemoryTooSmall` maps to, `magnitude` = bytes asked) (A58, reconcile item 31, 2026-11-15).
4. Every module that allocates exposes `Build<Module>MemoryEstimate`. Every function that builds `kRunOn` buffers
   takes `RunMemory& run_memory` as its last parameter (g.10.2) (A58, 2026-11-15).
5. Any total over ranges of cell blocks is one partial per cell block, in the grid's fixed block order, then
   `math::TreeSum` over the block partials; a per-group total (blob, pocket, free solid) is the group's block
   partials summed by `math::TreeSum` in ascending block index (g.3). Each run mode is then repeatable byte for
   byte, and a batched run agrees with a one-piece run within the solver's tolerance (A15).
6. `cancel_requested` is never a field of a `...Settings` type. A long-running solve takes `const
   std::atomic<bool>* cancel_requested` (may be null) and returns its module's `kCancelled` when it is set.
7. A design refers to another design by its name ("the solids design"), never by a module or file number.

## (a) Foundation `base/`

Built on the `rebuild` branch (Foundation, 2026-11-14, re-copied after the owner's review, 2026-11-15) in
`native/fluidity/base/`, included as `"fluidity/base/<name>.h"`. The declarations below are copied from the built
headers; where they and a design differ, the headers are right. Everything is `namespace fluidity` unless shown.
Function bodies of inline and `constexpr` functions are left out.

Rules the built headers follow:

- A **shared** header (CPU and GPU) is written in the C++20 subset, marks its functions `FLUIDITY_CPU_AND_GPU`
  and never includes `<expected>`, `<string>` or anything that allocates. A **CPU-only** header includes
  `base/cpu_only.h` first, which stops the CUDA compiler with an error naming the fix.
- **The two halves of an operation** (AGENTS.md "Structure"). The public function lives in `<name>_cpu.cc`: it
  checks its inputs, chooses the half by `RunOn`, runs the CPU loop, and calls the GPU half under
  `#if defined(FLUIDITY_HAS_GPU)`. The **private GPU half** is `<name>_gpu.cu`, declared in `<name>_gpu.h` (for
  example `buffer_gpu.h`, `ordered_sums_gpu.h`, `stable_order_gpu.h`, `check_can_run_on_gpu.h`,
  `processor_description_gpu.h`), in the C++20 subset with no CPU-only include, included only from its own
  folder; its names may say `Gpu`. Every GPU half returns `RunOnCallStatus` (`base/run_on_call.h`), which the CPU
  half maps into its error (`ToBufferError` for `BufferError`), and starts GPU functions, waits and holds
  temporary GPU memory of its own only through `base/gpu_launch.cuh`; it may call `cudaMemcpy` or `cudaMemset`
  directly, and every such status goes through `ToRunOnCallStatus` (`kCallFailed` on any failure). Only an
  allocation returns `kOutOfMemory`. A GPU function is a thin loop around shared
  per-item or per-thread functions (`FLUIDITY_CPU_AND_GPU`, in a shared header) that CPU tests run over every item,
  or every thread block and thread in launch order and in reverse. Callers use the public header; public
  identifiers never say `Gpu`. The GPU sense is always "thread block", never "block" or "group" alone.
- **Working buffers.** An operation that needs temporary arrays takes them from the caller in a move-only
  `<Name>Buffers` built once by `Build<Name>Buffers(run_on, max_count)`, with its exact bytes from
  `Compute<Name>BuffersSize(run_on)`. Each function has ONE form, which takes its buffers and allocates nothing
  (`kSizeMismatch` for a count above `max_count()`); `kOutOfMemory` comes only from `Build<Name>Buffers` and
  `BuildBuffer`. Working buffers hold their memory in one `Buffer`, and their contents mean nothing between calls.
  `SumBuffers` and `OrderBuffers` are the two in `base/`.
- **Preconditions.** A function that takes two buffers (`CopyBuffer`, `ComputeTreeDot`) or a buffer and its working
  buffers (the sums, the order, the prefix sum, the grouping) requires them on one processor. Its comment says so
  on a `Requires:` line, and `assert` (`<cassert>`) checks it in debug builds only; there is no error value for it,
  so a module has nothing to map. The checks that remain run sizes, then values.
- **Exactness.** Discrete outputs (integers, orders, counts, flags) are equal on CPU and GPU. Floating-point
  results agree within a tolerance, except the functions whose header promises the same bits: the marked
  functions of `scalar_math.h`, `vec3.h`, `mat3.h`, and every function of `tree_sum.h` and `ordered_sums.h`.
- "Same bits on CPU and GPU" always means: unless the result is NaN, whose bits IEEE 754 leaves open.
- Verbs used by `base/` beyond `Compute`, `Build`, `Check`, `Find`, `Apply`, `Solve`, `Read`, `Write`: `Copy`
  (`CopyBuffer`), `Free` (`FreeBufferBytes`, private; `GpuArray::Free`) and `Take` (none in `base/` yet).
- `Mat3`, like `Vec3`, is taken by value by every function, so GPU code may pass `kIdentity`.
- `base/` returns the bare enum (`RunOnError`, `BufferError`), the one exception to the module error rule; a
  module turns it into its own failure (`kOutOfMemory` or `kRunOnCallFailed` of its enum; a module with no
  `kOutOfMemory` reports `kRunOnCallFailed` with the bytes asked in its magnitude field). A failed check writes
  nothing; after a failed GPU call (`kRunOnCallFailed`) the outputs are unspecified and the run must stop.
- CMake: every target is created through `cmake/fluidity_targets.cmake` (configuring fails, naming the target,
  otherwise): `fluidity_add_test(<name> SOURCES <files...> LABEL <cpu|gpu|no_gpu> [LINKS <targets...>])` and
  `fluidity_add_test_library(<name> SOURCES <files...> [GPU_SOURCES <files...>] [LINKS <targets...>])` for
  test-only static libraries (a module names its own, such as `fluidity_grid_test_helpers`).

### `base/cpu_and_gpu.h`, `base/cpu_only.h`

```cpp
#define FLUIDITY_CPU_AND_GPU   // __host__ __device__ under the CUDA compiler (__CUDACC__), nothing otherwise (A13).
// cpu_only.h declares nothing: under __CUDACC__ it is
// #error "A CPU-only header was included in GPU code. Split the header that included it."
```

### `base/run_on.h` (shared), `base/check_can_run_on.h` (CPU only)

```cpp
// run_on.h
// The processor that runs an operation. The caller makes this choice; no function in the code base makes it.
enum class RunOn : std::uint8_t {
  kCpu,
  kGpu,
};
// The reasons a choice of processor cannot run here.
enum class RunOnError : std::uint8_t {
  kNone,  // No error. CheckCanRunOn never returns it as an error.
  kGpuNotBuilt,  // This build has no GPU code.
  kNoGpuFound,  // The build has GPU code but the machine has no usable GPU.
  kGpuCallFailed,  // A GPU call returned an error.
  kGpuMathDiffers,  // The shared math gave different numbers on the GPU.
};

// check_can_run_on.h (A12). The CPU always succeeds; the GPU check runs once per process.
std::expected<void, RunOnError> CheckCanRunOn(RunOn run_on);
std::string_view ToMessage(RunOnError error);
std::string_view ToName(RunOnError error);   // "gpu_not_built"; "unknown" for an out-of-range value.
```

### `base/run_on_call.h` (shared), `base/gpu_launch.cuh` (GPU halves only)

```cpp
// run_on_call.h. How a call to a GPU half ended. The CPU half turns it into its error: ToBufferError for BufferError.
enum class RunOnCallStatus : std::uint8_t {
  kDone,  // The outputs hold the answer.
  kOutOfMemory,  // An allocation of the call failed. Nothing was written. Only allocations return it.
  kSizeMismatch,  // A check made by the GPU half refused the inputs, such as a total above 2^32 - 1. Nothing written.
  kCallFailed,  // Another GPU call failed. The outputs are unspecified.
};
inline constexpr unsigned int kThreadBlockSize = 256;   // Threads in one thread block of a simple GPU function.
inline constexpr unsigned int kMaxThreadBlockCount = 65535;   // Each thread loops over several items beyond it.
struct LaunchSize {
  unsigned int thread_block_count = 0;  // Thread blocks; at least 1 for a launch.
  unsigned int thread_block_size = 0;  // Threads in each thread block.
};
// One thread per item in thread blocks of kThreadBlockSize, at most kMaxThreadBlockCount thread blocks; 0 items give
// 0 thread blocks.
FLUIDITY_CPU_AND_GPU constexpr LaunchSize ComputeLaunchSize(std::size_t item_count);

// gpu_launch.cuh (needs the CUDA compiler; included only by `<name>_gpu.cu` files).
inline RunOnCallStatus ToRunOnCallStatus(cudaError_t error);   // kDone for cudaSuccess, otherwise kCallFailed.
// Launches `function` on the default stream and checks that it started; an error while it runs is reported by the
// next call that waits.
template <typename... Parameters>
RunOnCallStatus ApplyGpuFunction(
    void (*function)(Parameters...),
    LaunchSize launch_size,
    std::type_identity_t<Parameters>... arguments
);
inline RunOnCallStatus CheckGpuWork();   // Waits for every GPU call made so far (cudaDeviceSynchronize).
struct GridStride {
  std::size_t first_item = 0;  // The global index of the thread.
  std::size_t item_step = 0;  // The number of threads in the launch.
};
__device__ inline GridStride ComputeGridStride();
template <typename T>
class GpuArray {   // GPU memory for one call, freed on every path. Build(count), Free(), values().
};
```

### `base/vec3.h`, `base/mat3.h`, `base/axis.h` (shared)

```cpp
// vec3.h. No default values: Vec3 stays trivial. Write Vec3{} for a zero vector.
struct Vec3 {
  double x;  // Along +X, in the unit of the quantity.
  double y;  // Along +Y, in the unit of the quantity.
  double z;  // Along +Z, which points up, in the unit of the quantity.
};
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator+(Vec3 left, Vec3 right) noexcept;
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator-(Vec3 left, Vec3 right) noexcept;
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator-(Vec3 vector) noexcept;
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator*(Vec3 vector, double factor) noexcept;
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator*(double factor, Vec3 vector) noexcept;
namespace math {                                                       // A11: short nouns.
FLUIDITY_CPU_AND_GPU constexpr double Dot(Vec3 first, Vec3 second) noexcept;      // Added in x, y, z order.
FLUIDITY_CPU_AND_GPU constexpr Vec3 Cross(Vec3 first, Vec3 second) noexcept;
FLUIDITY_CPU_AND_GPU inline double Length(Vec3 vector) noexcept;                  // Same bits on CPU and GPU.
FLUIDITY_CPU_AND_GPU inline Vec3 Normalized(Vec3 vector) noexcept;                // Caller: a non-zero axis in range.
FLUIDITY_CPU_AND_GPU inline bool IsFinite(Vec3 vector) noexcept;
}  // namespace math

// mat3.h. Nine doubles in matrix-row order; no default values (Mat3{} is zeros). Taken by value everywhere.
struct Mat3 {
  double xx;  // Matrix row x, column x. No unit of its own: the unit is the output's unit per input unit.
  double xy;  // Matrix row x, column y.
  double xz;  // Matrix row x, column z.
  double yx;  // Matrix row y, column x.
  double yy;  // Matrix row y, column y.
  double yz;  // Matrix row y, column z.
  double zx;  // Matrix row z, column x.
  double zy;  // Matrix row z, column y.
  double zz;  // Matrix row z, column z.
};
inline constexpr Mat3 kIdentity = {.xx = 1, .xy = 0, .xz = 0, .yx = 0, .yy = 1, .yz = 0, .zx = 0, .zy = 0, .zz = 1};
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator*(Mat3 matrix, Vec3 vector) noexcept;
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator*(Mat3 left, Mat3 right) noexcept;   // Right first, then left.
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator+(Mat3 left, Mat3 right) noexcept;
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator-(Mat3 left, Mat3 right) noexcept;
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator*(Mat3 matrix, double factor) noexcept;
namespace math {
FLUIDITY_CPU_AND_GPU constexpr Mat3 Transpose(Mat3 matrix) noexcept;
FLUIDITY_CPU_AND_GPU constexpr double Determinant(Mat3 matrix) noexcept;          // Expanded along matrix row x.
FLUIDITY_CPU_AND_GPU inline bool IsFinite(Mat3 matrix) noexcept;
// Right-handed turn by Length(rotation_vector) rad about it; kIdentity exactly for a zero vector. Uses Sin: may
// differ in the last bits between CPU and GPU.
FLUIDITY_CPU_AND_GPU inline Mat3 Rotation(Vec3 rotation_vector) noexcept;
}  // namespace math

// axis.h. Right-handed, Z up, as in Blender. Gravity points along -kUp.
enum class Axis : std::uint8_t {
  kX,  // Left to right: -X is the left side, +X the right side.
  kY,  // Front to back: -Y is the front side, +Y the back side.
  kZ,  // Bottom to top: -Z is the bottom side, +Z the top side. Up.
};
inline constexpr Vec3 kUp = {.x = 0, .y = 0, .z = 1};
namespace math {
FLUIDITY_CPU_AND_GPU constexpr Vec3 UnitVector(Axis axis) noexcept;
}  // namespace math
```

The CUDA compiler of the pinned toolkit accepts GPU code that reads `kUp` and `kIdentity` at run time.

### `base/per_particle_velocity.h` (shared)

```cpp
// One value for each of the particle's two velocities (A1). T is Vec3 for a velocity and double for a speed.
// No default values: the type stays trivial. Write PerParticleVelocity<T>{} for zeros.
template <typename T>
struct PerParticleVelocity {
  T for_next_step;  // Kept on the particle for the next step.
  T for_this_step;  // Moves the particle in this step.
};
```

Decided names that contain `travel` (`own_motion_in_travel`) keep it.

### `base/scalar_math.h` (shared)

Two groups. Same bits on CPU and GPU: `Floor`, `Abs`, `Sqrt`, `Min`, `Max`, `IsFinite` (and any formula built only
from these and `+ - * /`, as `Length` and `Normalized`). May differ in the last bits: `Cbrt`, `Exp`, `Expm1`, `Log`,
`Pow`, `Sin`, `Cos` (and `Rotation`); a formula that uses them is compared between CPU and GPU with a tolerance.

```cpp
namespace fluidity::math {
FLUIDITY_CPU_AND_GPU inline double Floor(double value) noexcept;
FLUIDITY_CPU_AND_GPU constexpr double Abs(double value) noexcept;          // Both zeros give +0.
FLUIDITY_CPU_AND_GPU inline double Sqrt(double value) noexcept;
FLUIDITY_CPU_AND_GPU inline double Cbrt(double value) noexcept;            // Accepts a negative value.
FLUIDITY_CPU_AND_GPU inline double Exp(double value) noexcept;
FLUIDITY_CPU_AND_GPU inline double Expm1(double value) noexcept;           // exp(value) - 1 without losing the small part.
FLUIDITY_CPU_AND_GPU inline double Log(double value) noexcept;             // Natural logarithm.
FLUIDITY_CPU_AND_GPU inline double Pow(double base, double exponent) noexcept;
FLUIDITY_CPU_AND_GPU inline double Sin(double angle) noexcept;             // rad
FLUIDITY_CPU_AND_GPU inline double Cos(double angle) noexcept;             // rad
FLUIDITY_CPU_AND_GPU constexpr double Min(double first, double second) noexcept;   // A NaN loses (fmin).
FLUIDITY_CPU_AND_GPU constexpr double Max(double first, double second) noexcept;   // A NaN loses (fmax).
FLUIDITY_CPU_AND_GPU inline bool IsFinite(double value) noexcept;          // False for infinity and NaN.
}  // namespace fluidity::math
```

### `base/tree_sum.h` (shared)

```cpp
namespace fluidity::math {
// Sum of value_at(0), ..., value_at(count - 1) in the fixed order of the padded binary tree (pad with zeros to the
// next power of two; each pair sum is its left side plus its right side). Calls value_at once per index, in index
// order; allocates nothing, does not recurse (64 doubles on the stack), so GPU code may call it. Same bits on
// every run and on CPU and GPU. count 0 gives +0.0.
template <typename ValueAt>
  requires std::invocable<ValueAt&, std::size_t>
FLUIDITY_CPU_AND_GPU double TreeSumOf(std::size_t count, ValueAt value_at) noexcept;
// TreeSumOf over values[0..count). count 0 gives +0.0 and `values` may then be null.
FLUIDITY_CPU_AND_GPU inline double TreeSum(const double* values, std::size_t count) noexcept;
// The larger of the magnitudes of two values, or NaN when either is NaN (a running maximum never hides a NaN).
FLUIDITY_CPU_AND_GPU constexpr double LargerMagnitude(double first, double second) noexcept;
}  // namespace fluidity::math
```

### `base/buffer.h` (CPU only; private `buffer_bytes.h`, `buffer_cpu.cc`, `buffer_gpu.h`, `buffer_gpu.cu`)

An array of fixed size on the processor the simulation runs on: the buffer under grid's `CellValues`,
`FaceValues`, `CornerValues`, `CellBlockValues` and under every per-particle array. Only `WriteBuffer` and
`ReadBuffer` move values between CPU memory and a buffer; `CopyBuffer` copies between two buffers on one
processor. A slice is `count` items from index `first`; every copy has a slice form; sizes, slices and overlap
are checked before memory is touched. The span parameters are `std::type_identity_t<...>`, so the value type is
deduced from the buffer alone and a `std::vector` or `std::array` is passed directly (a braced list is not).

```cpp
enum class BufferError : std::uint8_t {
  kNone,  // No error. No function returns it as an error.
  // The chosen processor could not hold the array. Only allocations return it; nothing was written.
  kOutOfMemory,
  // CheckCanRunOn failed for the chosen processor, or a call to it failed. The reason is not kept here: the outer
  // boundary calls CheckCanRunOn(run_on) before any work, which returns the RunOnError and its message.
  kRunOnCallFailed,
  // Sizes do not fit: a span and a buffer, or two buffers, have different sizes; a slice is out of range; two ranges
  // overlap; or a count or a total is above what the function can index (each function's Fails says which). The
  // last case can come from a scene that is too large, not only from a bug in the caller.
  kSizeMismatch,
};
// kDone gives kNone, kOutOfMemory kOutOfMemory, kSizeMismatch kSizeMismatch, kCallFailed kRunOnCallFailed. The one
// mapping every GPU half's status goes through.
BufferError ToBufferError(RunOnCallStatus status);
std::string_view ToMessage(BufferError error);
std::string_view ToName(BufferError error);   // "size_mismatch"; "unknown" for an out-of-range value.

// The kind of CPU memory a CPU buffer asks for. GPU buffers ignore it.
enum class CpuMemory : std::uint8_t {
  kOrdinary,  // Ordinary memory.
  kLockedForCopies,  // Locked for copies (papers: page-locked, pinned); ordinary memory when no GPU is usable.
};
inline constexpr std::size_t kBufferAlignmentBytes = 64;   // Every CPU buffer's data() is a multiple of this.
template <typename Value>
concept BufferValue = std::is_trivially_copyable_v<Value> && alignof(Value) <= kBufferAlignmentBytes;

template <BufferValue Value>
std::expected<Buffer<Value>, BufferError>
BuildBuffer(RunOn run_on, std::size_t size, CpuMemory cpu_memory = CpuMemory::kOrdinary);   // Every byte zero.

template <BufferValue Value>
class Buffer {   // Move-only; a moved-from buffer has size 0, null data, and keeps its run_on(). Fixed size.
 public:
  Buffer() = default;   // An empty CPU buffer: size 0, null data.
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  ~Buffer();
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] RunOn run_on() const;
  // A CPU pointer for RunOn::kCpu, a GPU pointer for RunOn::kGpu. Use it on run_on() only. May be null at size 0.
  [[nodiscard]] Value* data();
  [[nodiscard]] const Value* data() const;
};

// kSizeMismatch when first > size or count > size - first (no sum, so no wrap-around).
constexpr std::expected<void, BufferError> CheckSliceIsInside(std::size_t first, std::size_t count, std::size_t size);

// Slice and whole forms. kSizeMismatch for a slice outside the buffer, a size that differs, or a span that
// overlaps the slice; kRunOnCallFailed for a failed GPU copy, a memory error included. A failed check writes
// nothing; after a failed GPU call the target slice is unspecified.
template <BufferValue Value>
std::expected<void, BufferError>
WriteBuffer(std::type_identity_t<std::span<const Value>> values, Buffer<Value>& buffer, std::size_t first);
template <BufferValue Value>
std::expected<void, BufferError>
WriteBuffer(std::type_identity_t<std::span<const Value>> values, Buffer<Value>& buffer);
template <BufferValue Value>
std::expected<void, BufferError>
ReadBuffer(const Buffer<Value>& buffer, std::size_t first, std::type_identity_t<std::span<Value>> values);
template <BufferValue Value>
std::expected<void, BufferError>
ReadBuffer(const Buffer<Value>& buffer, std::type_identity_t<std::span<Value>> values);
// On the processor both buffers live on. Requires: from.run_on() == to.run_on() (assert in debug builds).
// kSizeMismatch for a slice outside its buffer or two slices that overlap in one buffer (a buffer copied onto
// itself included); count 0 copies nothing.
template <BufferValue Value>
std::expected<void, BufferError>
CopyBuffer(const Buffer<Value>& from, std::size_t first, std::size_t count, Buffer<Value>& to, std::size_t to_first);
template <BufferValue Value>
std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, Buffer<Value>& to);
```

### `base/processor_description.h` (CPU only; private `processor_description_cpu.cc`, `processor_description_gpu.h`, `processor_description_gpu.cu`)

```cpp
// The memory of one processor at one moment.
struct MemoryAvailable {
  std::uint64_t total_bytes = 0;  // Bytes. All the memory of the processor. 0 means unknown.
  std::uint64_t free_bytes = 0;  // Bytes. The part a program could take now, at most total_bytes. 0 when unknown.
};
// CPU: never fails (a failed query gives 0, 0). GPU: CheckCanRunOn first, then the CUDA runtime; kRunOnCallFailed
// for any failed query.
std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on);

// The name and the software versions of one processor, as text for the run report. A field the machine cannot
// answer is an empty string.
struct ProcessorDescription {
  std::string model;    // CPU brand string (or the architecture name), GPU device name.
  std::string driver;   // CPU: operating system name and version. GPU: driver version "major.minor".
  std::string runtime;  // CPU: compiler name and version. GPU: CUDA runtime version "major.minor".
};
ProcessorDescription ReadProcessorDescription(RunOn run_on);   // Never fails.
```

### `base/ordered_sums.h` (CPU only; private `ordered_sums_cpu.cc`, `ordered_sums_gpu.h`, `ordered_sums_gpu.cu`)

```cpp
class SumBuffers;
// Working buffers of the three sums on one processor, for up to max_count values (GPU: the products of a dot and
// the two layer arrays; CPU: nothing). kOutOfMemory (byte count or memory), kRunOnCallFailed.
std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count);
// Exact bytes BuildSumBuffers allocates: bytes_per_item * max_count + fixed_bytes.
struct SumBuffersSize {
  std::uint32_t bytes_per_item = 0;  // Bytes for each value of max_count.
  std::uint64_t fixed_bytes = 0;  // Bytes that do not depend on max_count.
};
SumBuffersSize ComputeSumBuffersSize(RunOn run_on);   // kGpu {14, 128}; kCpu {0, 0}.
// The one form of each sum: allocates nothing. Requires: every buffer and `buffers` on one processor (assert in
// debug builds). kSizeMismatch (count above a buffer or buffers.max_count()), kRunOnCallFailed. The first `count`
// values in the order of math::TreeSum; same bits on every run and on both processors.
std::expected<double, BufferError> ComputeTreeSum(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);
// Sum of first[i] * second[i]; each product rounded, never fused.
std::expected<double, BufferError>
ComputeTreeDot(const Buffer<double>& first, const Buffer<double>& second, std::size_t count, SumBuffers& buffers);
// math::LargerMagnitude over the values: exact in any order; NaN anywhere gives NaN; count 0 gives 0.0.
std::expected<double, BufferError>
ComputeLargestMagnitude(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);
class SumBuffers {   // Move-only; moved-from buffers have max_count() 0 and keep their run_on().
 public:
  [[nodiscard]] std::size_t max_count() const;   // The most values one call may use these buffers for.
  [[nodiscard]] RunOn run_on() const;
  [[nodiscard]] std::size_t byte_count() const;  // ComputeSumBuffersSize(run_on()) applied to max_count().
};
```

### `base/stable_order.h` (CPU only; private `stable_order_cpu.cc`, `stable_order_digit.h`, `stable_order_gpu.h`, `stable_order_gpu.cu`)

```cpp
class OrderBuffers;
// Working buffers of BuildStableOrder, ComputeExclusivePrefixSum and BuildGroupsByDestination on one processor, for
// up to max_count items, so none allocates. kSizeMismatch (max_count above 2^32 - 1), kOutOfMemory,
// kRunOnCallFailed.
std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count);
// Each function below: one form, allocates nothing. Requires: every buffer and `buffers` on one processor (assert
// in debug builds).
// order[i] becomes the index of the item at place i when items are sorted by ascending key. Items with equal keys
// keep the order they have in previous_order (index order when it is null). kSizeMismatch (count above 2^32 - 1, a
// buffer or buffers.max_count(); previous_order == &order), kRunOnCallFailed.
std::expected<void, BufferError> BuildStableOrder(
    const Buffer<std::uint64_t>& keys,
    std::size_t count,
    const Buffer<std::uint32_t>* previous_order,
    Buffer<std::uint32_t>& order,
    OrderBuffers& buffers
);
// sums[i] = values[0] + ... + values[i - 1] for i in [0, count]; sums[count] is the total. Exact unsigned
// arithmetic; a total above 2^32 - 1 is kSizeMismatch (sizes, then values). sums.size() >= count + 1. `sums` may be
// the same buffer as `values` (in place). Same bytes on CPU and GPU.
std::expected<void, BufferError> ComputeExclusivePrefixSum(
    const Buffer<std::uint32_t>& values,
    std::size_t count,
    Buffer<std::uint32_t>& sums,
    OrderBuffers& buffers
);
// Groups items by destination: item_order lists the items of destination 0, then 1, ..., each group in ascending
// item index; first_of_group has destination_count + 1 entries. kSizeMismatch (sizes, then values: item_count above
// 2^32 - 1, a buffer or buffers.max_count(); first_of_group too small; two buffers the same; a destination of
// destination_count or more), kRunOnCallFailed.
std::expected<void, BufferError> BuildGroupsByDestination(
    const Buffer<std::uint32_t>& destination,
    std::size_t item_count,
    std::uint32_t destination_count,
    Buffer<std::uint32_t>& first_of_group,
    Buffer<std::uint32_t>& item_order,
    OrderBuffers& buffers
);
// Exact bytes BuildOrderBuffers allocates: bytes_per_item * max_count + fixed_bytes.
struct OrderBuffersSize {
  std::uint32_t bytes_per_item = 0;  // Bytes for each item of max_count.
  std::uint64_t fixed_bytes = 0;  // Bytes that do not depend on max_count.
};
OrderBuffersSize ComputeOrderBuffersSize(RunOn run_on);   // kCpu {4, 8192}; kGpu {6, 4096}.
class OrderBuffers {   // Move-only; moved-from buffers have max_count() 0 and keep their run_on().
 public:
  OrderBuffers(const OrderBuffers&) = delete;
  OrderBuffers& operator=(const OrderBuffers&) = delete;
  OrderBuffers(OrderBuffers&& other) noexcept;
  OrderBuffers& operator=(OrderBuffers&& other) noexcept;
  ~OrderBuffers() = default;
  [[nodiscard]] std::size_t max_count() const;   // The most items one call may use these buffers for.
  [[nodiscard]] RunOn run_on() const;
  [[nodiscard]] std::size_t byte_count() const;  // ComputeOrderBuffersSize(run_on()) applied to max_count().
};
```

### `base/memory_count.h` (CPU only)

```cpp
enum class CountedPer : std::uint8_t { kCellBlock, kCellPlace, kFacePlace, kCornerPlace, kParticle, kGivenCount };
enum class MemoryWhere : std::uint8_t {
  kRunOn,  // In the memory of the processor the run uses: GPU memory in a GPU run, CPU memory in a CPU run.
  kCpu,  // Always in CPU memory, even in a GPU run.
};
enum class KeptFor : std::uint8_t { kWholeRun, kOneStepTry, kOneCall, kOneSnapshot };   // Does not change the sum.
struct MemoryEstimateEntry {
  std::string_view name;  // What the array holds, such as "grid: face piece kinds". Must point at a string literal.
  CountedPer counted_per = CountedPer::kGivenCount;  // Which count of MemoryCounts gives the number of items.
  std::uint64_t given_count = 0;  // Items. Read only when counted_per is kGivenCount.
  std::uint32_t bytes_each = 0;  // Bytes of one item.
  std::uint32_t copies = 1;  // Arrays of the same shape: 2 when a committed and a try copy exist.
  KeptFor kept_for = KeptFor::kWholeRun;  // The lifetime of the array. Does not change the sum.
  MemoryWhere where = MemoryWhere::kRunOn;  // In the run's memory, or always in CPU memory.
};
struct MemoryCounts {
  std::uint64_t cell_blocks = 0;  // Cell blocks in the grid.
  std::uint64_t cell_places = 0;  // 64 per block.
  std::uint64_t face_places = 0;  // 240 per block.
  std::uint64_t corner_places = 0;  // 125 per block.
  std::uint64_t particles = 0;  // Particles in the run.
};
// Saturating sums (overflow gives the largest std::uint64_t). Every entry, or only the entries kept in `where`.
std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts);
std::uint64_t
ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts, MemoryWhere where);
// The bytes of the largest single entry kept in `where`, 0 when none (A115). The GPU driver keeps one stack for the
// largest need of any function launched, so a plan adds the kRunOn sum and the largest kDriverStack entry, not their sum.
std::uint64_t
ComputeLargestEntryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts, MemoryWhere where);
```

Every module's rows set `where`, and sizes are `sizeof` of the real types with a `static_assert` on each. The
rows of each module: `contact::BuildContactMemoryEstimate`, `grid::BuildGridMemoryEstimate(RunOn run_on,
std::uint32_t storage_count)` (the step passes 2, a batched piece 1), `grid::BuildBorderSearchMemoryEstimate(RunOn)`,
`particles::BuildParticlesMemoryEstimate`, `transfer::BuildTransferMemoryEstimate(ingredient_count, material_count,
material_set_count, sets_without_solve)`, `pressure::BuildPressureMemoryEstimate`, `pressure::BuildSolveBatchesMemoryEstimate`,
`viscosity::BuildViscosityMemoryEstimate`, `materials::BuildMaterialsMemoryEstimate`, `solids::BuildSolidsMemoryEstimate`,
`gas::BuildGasMemoryEstimate`, `cell_resize::BuildCellResizeMemoryEstimate`, `render_output::BuildRenderOutputMemoryEstimate`,
`io::BuildIoMemoryEstimate`. Pressure, viscosity and gas buffers built with `max_loaded_blocks > 0` return their
streamed rows twice: `where = kCpu` at the limits and `where = kRunOn` at `max_loaded_blocks` (A15, (g)). The step
sums them in `step::BuildRunPlan(..., std::span<const MemoryEstimateEntry> output_estimate, std::span<const MemoryEstimateEntry>
cell_resize_estimate = {})`.

### Private to `base/`, listed so no module includes them

`buffer_bytes.h` (`BufferBytes`, `BuildBufferBytes`, `FreeBufferBytes`, `WriteBufferBytes`, `ReadBufferBytes`,
`CopyBufferBytes`), the five `<name>_gpu.h` halves, `stable_order_digit.h` (shared digit rule of the counting
passes), `stable_order_steps.h` (the per-thread steps of the GPU order and prefix sum: `PassSource`, `ChunkItems`,
`CopyChunkItem`, `ComputeChunkDigitCount`, `CopyChunkItemsOfDigit`, `ComputeThreadValues`, `ApplyHalvingRound`,
`ComputeRunningSumAddend`, `kChunkThreadBlockSize`, `ThreadBlockRound`, `CopyThreadPrefixSums`, `ComputeGroupStart`, `ComputeDestinationPassCount`,
`BuildPrefixSumLayers`, ...), `tree_sum_layers.h` (one layer of the GPU tree sum: `ComputePairCount`,
`ComputeLayerValueCount`, `ComputeLayerArrayOffset`, `LayerSides`, `ComputeLayerPairSum`,
`ComputeLayerLargerMagnitude`), `processor_description_text.h` (`FindCpuModelInText`), `check_can_run_on_math.h`
(the fixed inputs of the GPU math check; renamed from `run_on_gpu_check_math.h`). `run_on_call.h` and
`gpu_launch.cuh` above are meant for every module's GPU halves, as patterns to copy, not for callers.

### Test support (`base/test_support.h`, library `fluidity_base_test_support`; tests only)

```cpp
std::uint64_t ToBits(double value);
std::vector<std::uint64_t> ToBits(std::span<const double> values);
struct Tolerance {   // The method paper's formula: absolute_floor + relative_term * |reference|.
  double absolute_floor = 0.0;  // Same unit as the values; the allowed difference near zero.
  double relative_term = 0.0;  // No unit; the allowed difference per unit of |reference|.
};
::testing::AssertionResult CheckWithinTolerance(double reference, double compared, Tolerance tolerance);
::testing::AssertionResult CheckSameBitsOrBothNaN(double expected, double actual);
template <std::ranges::contiguous_range Values>
Buffer<std::ranges::range_value_t<Values>> BuildBufferHolding(RunOn run_on, const Values& values);
template <BufferValue Value>
std::vector<Value> ReadAll(const Buffer<Value>& buffer);
std::vector<double> BuildMixedValues(std::size_t count);
std::vector<std::uint64_t> BuildKeysWithManyTies(std::size_t count);
std::vector<std::uint64_t> BuildKeysDifferingInTopByte(std::size_t count);
inline constexpr std::string_view kResultDigestFileVariable = "FLUIDITY_RESULT_DIGEST_FILE";
std::uint64_t ComputeResultDigest(std::span<const std::byte> bytes);   // FNV-1a, 64 bits.
void WriteResultDigestToFile(const std::filesystem::path& path, std::string_view name, std::span<const std::byte> bytes);
// Appends `<name> <digest>` to the file FLUIDITY_RESULT_DIGEST_FILE names; nothing when it is unset. The gpu tests
// call it for every GPU result they check, and tools/run_gpu_tests.ps1 compares the two runs' files.
template <typename Value>
void WriteResultDigest(std::string_view name, std::span<const Value> values);
std::string FindResultDigestFile();
```

A module's own test helpers go into its own library made with `fluidity_add_test_library` (for example
`fluidity_grid_test_helpers`, `fluidity_step_test_support`), linking `fluidity_base_test_support` when they need it.

## (b) Cross-module names

| Concept | Final name | Owner | Names it replaced |
|---|---|---|---|
| An error with context | `std::expected<T, <Module>Failure>`, `<Module>Failure { <Module>Error error; ... }`, `ToMessage`, `ToName` | each module | `GridProblem`, `PressureErrorContext`, `ParticlesErrorDetails` |
| The chosen processor is not usable | `kRunOnCallFailed` | each module | `kProcessorUnavailable`, `kCannotRunOnGpu`, `kGpuCallFailed` |
| Out of memory (A58, reconcile item 31, 2026-11-15) | `<Module>Error::kOutOfMemory`, **the last value** of every module error enum (earlier values keep their numbers), returned only by the module's building functions; `BufferError::kRunMemoryTooSmall` maps to it with `magnitude` = bytes asked. Viscosity and solids move theirs to the end; pressure, gas and transfer gain it; each design's value count and each plan's enum test follow | each module | `kOutOfMemory` in the middle of `ViscosityError` / `SolidsError`; out of memory reported as `kRunOnCallFailed` in a module with no `kOutOfMemory` |
| One reserved block per run (A58, 2026-11-15; pending Foundation Tasks 17–19, g.10.2) | `RunMemory` (`base/run_memory.h`, move-only), `BuildRunMemory`, `BuildBuffer<Value>(RunOn, std::size_t size, RunMemory&)`, `ComputeRunMemoryBytes`, `ComputeRunMemoryPieceBytes`, `kRunMemoryPieceAlignmentBytes = 256`, `BufferError::kRunMemoryTooSmall` (appended last), the `RunMemory&` forms of `BuildSumBuffers` and `BuildOrderBuffers`; **every module builder takes `RunMemory& run_memory` last** (`Requires:` `run_on == run_memory.run_on()`, debug assertion, A30); one `RunMemory` per processor a run uses (`run_memory` for `kRunOn` entries; in a GPU run `cpu_run_memory`, locked for copies in a batched run, for `kCpu` entries kept `kWholeRun`/`kOneStepTry`; one object in a CPU run); a builder takes the block of its buffers' processor, one building on both takes `RunMemory& run_memory, RunMemory& cpu_run_memory` last; `ComputeRunMemoryBytes(estimate, counts, MemoryWhere)`; `MemoryReport::handed_out_cpu_bytes`, `reserved_cpu_bytes` (correction of 2026-11-15); no `kRunOn` entry is `kOneCall`; estimate test `Build<Module>Buffers, IntoRunMemory_HandsOutExactlyTheEstimate` | base, each module, step | one allocation per buffer; builders without a `RunMemory` |
| Buffers on two processors where one is required | A documented precondition (`Requires:` same processor), checked by an assertion in debug builds; no error value (A30). A caller-facing boundary that receives the processor choice from the user (step run settings, I/O, CLI, the add-on's command) still reports a wrong or unavailable processor as an error | base, each module | `BufferError::kRunOnMismatch` mapped to `kBadInput`; "a view or array whose `run_on` differs from the buffers' is `kBadInput`" |
| An input of another grid version | `kWrongGridVersion` | each module | `kGridChanged` |
| A cancelled solve | `kCancelled`, input `const std::atomic<bool>* cancel_requested` | pressure, viscosity, gas, ... | `cancel_requested` in settings |
| The two velocities of a particle | `PerParticleVelocity<T> { for_next_step; for_this_step; }` | base | `{ state; travel; }`, `velocity.state`, `velocity.travel`, `state_velocity`, `travel_velocity` |
| Particle fields for them | `particles::Field::kVelocityForNextStep`, `kVelocityForThisStep`; snapshot array `velocity_for_next_step` | particles | `kStateVelocity`, `kTravelVelocity`, `velocity` |
| Change of a particle's next-step velocity by a reacting solid | `contact::SolidReaction::velocity_for_next_step_change` | contact | `state_velocity_change` |
| Contact's moved velocities | `contact::MovedParticles::velocity_for_next_step`; `FindFirstHit(..., Vec3 velocity_for_this_step, ...)` | contact | `state_velocity`, `travel_velocity` |
| The grid, owning form | `grid::StaggeredGrid` (move-only; `GridVersion` 1 or more, 0 means "bound to no grid") | grid | `Grid`, `GridLayout`, `MixedSizeGrid` |
| Building a grid version | `grid::BuildGrid(const GridInputs&, GridBuffers&)` on the run's processor (A17: a GPU run builds on the GPU; `GridInputs::active_cell_buffer`, `active_cell_buffer_count`); buffers from `BuildGridBuffers(const GridSettings&, GridBuilderBuffers&, RunMemory& run_memory)` (A58, 2026-11-15), taken back with `TakeBuffers(StaggeredGrid&&)` (A29, A31); checks `ComputeGridCounts(inputs, buffers)`, `CheckSizeStepRule(inputs, buffers)` (their span forms kept; a span form given a GPU grid is a `Requires:` precondition, not `kBadInput`: reconcile items 32, 33, 2026-11-15); the one copy of a GPU grid `ReadGrid(grid, cpu_buffers)` (g.1); test-only `BuildTestGrid` replaces the allocating `BuildGrid(inputs, settings)` (reconcile item 33, 2026-11-15) | grid | `BuildGrid(inputs, settings, storage)`; "`BuildGrid` runs on the CPU, also for GPU runs"; `BuildGrid(inputs, settings)` |
| The grid for shared formulas | `grid::GridTables` (from `grid.tables(run_on)`; `tables(kCpu)` on a GPU grid is version 0 with null arrays) | grid | `GridOnProcessor`, `GridLayoutFields` |
| Positions in the arrays of one version | `grid::CellPlaceIndex`, `FacePlaceIndex`, `CornerPlaceIndex`, `CellBlockIndex`; none is `grid::kNoIndex` | grid | raw `cell_index`, `face_index`, `kNoCell`, `-1` |
| Stable identities | `grid::CellId`, `FacePieceId`, `CellBlockId` | grid | `std::int64_t cell_id` |
| A run of cell blocks | `grid::CellBlockRange { first_block; block_count; }` | grid | `{ first; end; }` |
| One value per place | `grid::CellValues<T>`, `FaceValues<T>`, `CornerValues<T>`, `CellBlockValues<T>`, built with `Build...Values(RunOn, std::uint32_t max_cell_blocks, RunMemory& run_memory)` (A58, 2026-11-15) and bound with `ApplyGridVersion(values, const StaggeredGrid&)` | grid | `ApplyGridVersion(grid, values)`, `CellField`, `FacePieceValues` |
| A piece of the grid for a batch | class `grid::GridPiece` (in `grid/grid_piece.h`) with `grid()`, `range()`, `range_in_whole_grid()`, `block_in_whole_grid()` (CPU span; empty for a GPU piece), `block_in_whole_grid_buffer()`; `BuildGridPiece(const StaggeredGrid& whole, CellBlockRange, std::span<const CellBlockIndex> border_blocks, GridBuffers&)` (requires a CPU whole grid: a `Requires:` precondition with a debug assertion, not `kBadInput`, reconcile item 32, 2026-11-15) and the buffer form `BuildGridPiece(whole, range, const Buffer<CellBlockIndex>& border_blocks, std::uint32_t border_block_count, GridBuffers&)` (either processor; g.1); blocks keep the whole grid's fixed order; `grid::CellBlockRange` is in the shared `grid/grid_ids.h` | grid | struct `GridPiece { grid; range; block_in_whole_grid; }`, `BuildGridPiece(whole, RangeWithBorder, settings)`, `BuildRangeWithBorder` |
| Border blocks of a range | `grid::FindBorderBlocks(grid, range, border_width)` returning `std::expected<std::vector<CellBlockIndex>, GridFailure>`, and a form writing a span and returning the count (both require a CPU grid: a `Requires:` precondition, A30); the buffer form on either processor with `BorderSearchBuffers` from `BuildBorderSearchBuffers(RunOn, max_cell_blocks, RunMemory& run_memory)` (A58, 2026-11-15), one stacked layer per call (g.1); the border includes blocks storing a face piece of a touching cell | grid | a plain `std::vector` return |
| Memory estimate entries of the grid | `grid::BuildGridMemoryEstimate(RunOn, std::uint32_t storage_count, std::uint32_t max_cell_blocks, BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid)` (a GPU grid: 1,996 B per block, all `kRunOn`, plus order buffers; g.1; the third parameter from the reconcile pass, g.7), `grid::BuildBorderSearchMemoryEstimate(RunOn, std::uint32_t max_cell_blocks)` | grid | `kGridMemoryEstimate`, the two-parameter form |
| Grid builder buffers shared by several grids (reconcile) | `grid::GridBuilderBuffers` (move-only; `max_cell_blocks()`, `run_on()`), `BuildGridBuilderBuffers(RunOn, std::uint32_t max_cell_blocks, RunMemory& run_memory)`, `BuildGridBuffers(const GridSettings&, GridBuilderBuffers&, RunMemory& run_memory)` (lent builder buffers; the only form, A31; `RunMemory&` last, A58, 2026-11-15), `BuilderBuffersSharing { kOnePerGrid, kOneForAllGrids }`; `GridSettings::max_cell_blocks` defaults to 0 (g.7) | grid (used by step, cell resize, batches, pressure) | `GridSettings::max_cell_blocks = 1u << 22`; `GridScratch`, `BuildGridScratch`, `ScratchSharing { kOnePerBuffers, kOneForAllBuffers }`, `BuildGridBuffers(const GridSettings&)` |
| The shared cell size rule (A41, A61, A62, A63, 2026-11-15) | `grid/cell_size_rule.h` (shared, `FLUIDITY_CPU_AND_GPU`, no allocation; g.10.1): `grid::CellSizeSettings` and `grid::CellSizeFacts` (moved from cell resize), `CellWidthsByLevel`, `CellSizeRegion`, `ArtistCellSizes`, `ComputeReferenceSpeed`, `ComputeWantedCellWidth` (moved), `ComputeWantedSizeLevel`, `ComputeArtistSizeLevel`; `grid/new_cell_rule.h` (shared; A118): `NewCellNeighbors` and `FindNewCellNeighbors` (stage 2's fill of new cells); `cell_resize::CellResizeSettings::cell_size` is a `grid::CellSizeSettings`; cell resize's `ComputeVelocityGradientSize` (`velocity_gradient_math.h`), `ComputeCellSizeChanges`, `CellSizeChange { kKeep, kSplit, kMayMerge, kMergeReady }` | grid (read by step and cell resize) | `cell_resize/cell_size_rule.h`, `cell_resize::CellSizeSettings`, `cell_resize::ChooseCellSize`, `step::StepCallbacks::choose_cell_size`, "Open (owner): a shared size rule" |
| Refinement volume in the size rule (A78, A79, 2026-11-15; replaces the A63 artist term) | `grid::kCellSizeFactorStepsPerDoubling` (4), `kMaxCellSizeFactorCode` (24), `CellSizeFactorTable`, `ComputeCellSizeFactorTable` (CPU), `RefinementFactors` (min-pyramid of one-byte codes, `f = 2^(c/4)`, 1/64..64), `ComputeCellSizeFactorCode(factors, smallest_size_level, cell)`; `CellSizeFacts::cell_size_factor` (replaces `artist_size_level`); `wanted = clamp(min(w_band, w_solid, w_shear) x f, w_min, w_max)`, `w_min` in the pinned surface band, pockets and gas. **Removed:** `CellSizeRegion`, `ArtistCellSizes`, `ComputeArtistSizeLevel` | grid; read by cell resize, step | `w_artist`, artist regions, top-view map |
| Refinement volume in cell resize and step (A78, 2026-11-15) | `BuildCellResize(..., const grid::RefinementFactors& refinement_layout, ...)`, `cell_resize::WriteRefinementFactors(std::span<const std::int8_t>, CellResize&)`, row `"cell resize: refinement factors"`, `ResizeCellsReport::refinement_copy_bytes`, `BuildCellResizeMemoryEstimate(..., refinement_code_count)`; `step::BuildStartCells(..., const grid::RefinementFactors&, const grid::CellSizeFactorTable&, max_cell_count)`; `step::ReadyScene::refinement_code_count`, `refinement_key_count` | cell resize, step | `const grid::ArtistCellSizes& artist` parameters; rows "cell size regions", "cell size map" |
| Refinement volume builder (A77, A78, A79, 2026-11-15) | `io/refinement_volume.h` (CPU): `RefinementZoneShape` (`kBox`, `kEllipsoid`, `kClosedMesh`), `RefinementZonePoseKey`, `RefinementZone`, `RefinementCameraKey`, `RefinementCamera`, `RefinementKey`, `RefinementKeyFile`, `RefinementVolume`, `BuildRefinementVolume`, `WriteRefinementVolume` (OpenVDB `cell_size_factor` per key), `ReadRefinementVolume`; `RunScene::refinement_volume()`; CLI `build-refinement`; CPU entry `"io: refinement keys"`; run-folder `inputs/refinement_codes`, `inputs/refinement/` | I/O | `RunScene::artist_cell_sizes()`, libspng, `kMaxCellSizeMapSidePixels` |
| Refinement scene fields and add-on (A77, A78, A79, 2026-11-15) | scene `refinement_sample_spacing_log2`, `refinement_zones[]`, `refinement_camera`, `refinement_volume_files[]`; add-on `Object.fluidity_refinement_zone.{enabled, factor}` ("Cell Size Factor"), `refinement_camera_enabled`, `refinement_camera_object`, `refinement_pixels_per_cell`, `refinement_sample_spacing_log2`, `logic/zone_mesh.py`; findings `refinement_zone_unsupported_display`, `refinement_mesh_not_closed`, `refinement_mesh_deforms`, `cell_size_factor_out_of_range`, `cell_size_factor_clamped`, `refinement_camera_missing`, `refinement_camera_on`, `refinement_zone_without_cell_resize`, `refinement_volume_mismatch` | I/O, add-on | `cell_size_regions`, `cell_size_map`, `cell_size_region_collection`, `fluidity_cell_size_region.cell_width`, `cell_size_map_image`, `cell_size_region_rotated`, `grey_png.py` |
| One value per particle | `Buffer<T>`, or a pointer inside `particles::ParticleFields` | base, particles | `particles::ParticleValues<T>` |
| Particles in a try | `particles::ParticlesTry`, `ParticleFields`, `ParticleFieldsToChange` | particles | `ParticleGroups` |
| Particle order | by (`CellPlaceIndex` of the containing active cell, id), `group_begin` per block, `cell_begin` per cell place | particles | (cell block, id) |
| Memory estimate entries of particles | `particles::BuildParticlesMemoryEstimate` | particles | `particles::BuildMemoryEstimate` |
| A complete neighbourhood | `particles::FindNeighbourhood` with `NeighbourhoodInputs { outside_solid; face_side; }`; the field `is_grid_only_region` is not in the first version (A64, 2026-11-15; it returns with "Deep liquid without particles (later; needs review)") | particles | `grid::FindNeighbourhood`, `can_hold_liquid` |
| Split, merge, count changes | `SplitParticle`, `MergeParticles`, `AdjustParticleCount` taking `grid_before_resize` and `grid` with their `NeighbourhoodInputs`; called only inside the resize transaction; every liquid cell holds particles of its own `size_level` (eight of level `l + 1` merge into one of level `l` when their cells merge, one splits into eight when its cell splits; drifted particles keep their level and are counted) (A64, 2026-11-15); **every** resize pass also splits a too-large and merges too-small particles of `kMayBeSplitAndMerged` materials (chunks of eight, or two to seven after `kMaxMergeWaitPasses = 16` passes; per-particle `merge_wait_passes`; `AdjustParticleCountResult::{mismatched_particle_count, waiting_particle_count, lone_particle_count, incomplete_split_count, kinetic_energy_change}`; g.11.3) (A67, 2026-11-15) | particles | one-grid forms; "deep liquid needs no particles"; drifted particles keep their level (A64, replaced by A67) |
| Batch copies of particles | `particles::WriteParticleGroups`, `ReadParticleGroups` | particles | |
| Restart from a snapshot | `particles::BuildParticlesFromSnapshot(const ParticlesSettings&, const ParticlesSnapshot&, const grid::StaggeredGrid&)`; `SnapshotFields { bool all = true; FieldList fields{}; }`, `kOutputSnapshotFields` | particles | two-argument form |
| A marker no longer needed | `particles::RemoveReason::kMarkerNotNeeded` | particles | |
| Plate side identity | `contact::PlateSide { plate_id; SolidSide side; }`; on particles `plate_side` | contact, particles | `SideId`, `SolidSideId`, `solid_side` |
| Domain sides | `contact::DomainSide { kLeft, kRight, kFront, kBack, kBottom, kTop }` | contact | `grid::DomainSide` |
| Nearest solid surface | `contact::SolidVelocitySample`, `ComputeSolidVelocitySample` | contact | |
| Round solid shapes (A74, 2026-11-15) | `contact::Sphere { center; radius; }`, `contact::Cylinder { center; axis; radius; half_height; }`, `ToSolid(Sphere, ...)`, `ToSolid(Cylinder, ...)`, `SolidShape::kSphere`, `kCylinder` (appended); `Solid::half_size` = `{r, r, r}` / `{r, r, half_height}` | contact; consumed by solids, I/O, add-on | (the cube stand-in for a sphere; "known-open: needs a cylinder") |
| Faces of round solids (A74, 2026-11-15) | `SolidFace::kCurved` (sphere surface, cylinder wall), `kLowRim`, `kHighRim` (appended after `kHighZ`); caps are `kLowZ`/`kHighZ` | contact | |
| Spin of a round solid (A74, 2026-11-15) | a turn about a sphere's centre or a cylinder's axis: not a shape turn, not counted by `kMaxTurnPerStep`; surface velocity uses the full angular velocity (contact plan P17) | contact, solids | |
| Round shapes in the scene file and add-on (A74, 2026-11-15) | `solids[].shape` `sphere` \| `cylinder`; `Shape.SPHERE`, `Shape.CYLINDER` (role `SOLID`); finding `round_solid_scaled_unevenly`; I/O derives free sphere and cylinder inertia | I/O, add-on | |
| Round-body force tests (A74, 2026-11-15) | `LiquidForce, RotatingCylinder_TorquePerLength`, `LiquidForce, SettlingSphere_BalancesNetWeight`; `ComputeSolidCoverage, Sphere_BlockedAreaConverges` | viscosity, solids | `RotatingCylinders_TorquePerLength` (known-open), `SettlingCube_BalancesNetWeight` |
| Per face: outside every solid, plate side, solid velocity | `transfer::SolidsAtFaces { outside_solid; side; solid_velocity; time; grid_version; }`, filled by solids | transfer | `can_hold_liquid`, `grid::SolidsAtFaces` |
| A material set | `std::span<const std::uint8_t> material_is_counted` (one 0/1 entry per material, at least one 1; empty is `kBadInput`); `transfer::MaterialSetKey`, `ToMaterialSetKey`, `CheckMaterialSets`, `TransferError::kWrongMaterialSet` | transfer | "empty mask means all materials" |
| Face sums and snapshots | `transfer::GridFromParticles`, `GridSnapshot`, `FaceValueKind` (with `kFromSolve`); `kFromGridOnlyRegion` and its uses are not in the first version (A64, 2026-11-15) | transfer | |
| After-pressure save | `SaveGridAfterPressure(..., const grid::FaceValues<std::uint8_t>* was_solved, ...)`, `GridToParticlesInputs::cell_was_solved` as a pointer; both null for a set no solve writes | transfer | references |
| Per-cell sums from particles | `transfer::CellsFromParticles`, `ComputeCellsFromParticles`, `ComputeMixtureDensity`, `ComputeSensedVolumeRatio` | transfer | `IngredientVolumes` |
| The solve without viscosity | `pressure::SolvePressure(const PressureEquations&, const SolveSettings&, const std::atomic<bool>* cancel_requested, SolvedArrays outputs, PressureBuffers&)`; `SolvedArrays { pressure; velocity_after_pressure; blocking_material_velocity_after_pressure; sealed_gas_pressure; }` (g.2); `SolveResult { report; blobs; iteration_count; ladder_rung_count; blobs_without_direct_solve; }` (g.3); solves equations with blocking-material faces and sealed gas blobs, as viscosity's joint solve does (A24, g.2) | pressure | `SolvePressure(equations, velocity_response, settings, buffers)`, `VelocityResponse`, `connection_scale` |
| Air pockets | `pressure::FindAirPockets(const AirPocketInputs&, AirPocketArrays outputs, PressureBuffers&)`; `AirPocketArrays { pocket_of_cell; pocket_storage; }` | pressure | `FindAirPockets(inputs, buffers)` |
| Liquid distance | `pressure::ComputeDistanceToSurface` with `material_is_counted`: once with `kAllLiquidMaterials`, once per liquid material that touches another; an all-zero mask means "no liquid" (every open cell is air) | pressure | |
| Where the liquid is | `pressure::GeometrySnapshot` with `CellGeometry`, `FacePieceGeometry`, and in viscous scenes `CellEighthGeometry`, `FacePieceCentre` | pressure | |
| Volumes and rates | `filled_volume`, `open_volume`, `solid_volume_rate`, `face_inertia` (`ComputeFaceInertia`) | pressure | `liquid_volume`, `solid_flux`, `face_mass` |
| Pieces of the joint solve | `GivenTerms`, `ComputePressurePush`, `ApplyPressure`, `ComputeVolumeError`, `ComputeOwnWeight(equations, connection_scale, own_weight)`, `ComputeStartPressure`, `ApplyZeroMeanToSealedBlobs`, `ApplyPressureOffset`, `ComputeWasSolved`, `CheckFlowImbalance`, `ComputeFlowImbalance`, `BuildPressureLadder`, `ApplyPressureLadder`; A24: the forms of `ComputePressurePush`, `ComputeVolumeError`, `ComputeOwnWeight`, `ComputeStartPressure` and `CheckFlowImbalance` with a sealed-gas blob buffer, and `ComputeBlockingMaterialVelocities`; every piece accepts material faces, and only a form without a blob buffer given `sealed_gas_blob_count > 0` is `kBadInput` (g.2); A26, A27: for the viscous solve `pressure::CoupledVelocities { blocking_material_velocity; free_solid_motion; }`, the forms `ComputeVolumeError(..., velocity, CoupledVelocities, GivenTerms, volume_error, sealed_gas_volume_error)` and `CheckFlowImbalance(..., velocity, CoupledVelocities, settings, buffers)`, `ComputeCoupledPush(equations, pressure, blocking_material_push, free_solid_push)`, `ComputeOwnWeight(equations, connection_scale, const Buffer<double>* material_face_scale, own_weight, sealed_gas_own_weight)`; the A24 forms without `CoupledVelocities` stay (they eliminate material faces, `SolvePressure`'s arithmetic) and viscosity no longer calls them (g.9) | pressure | `GivenFlows`, `ApplyStartPressure`; `kBadInput` for equations with either term; viscosity calling the A24 forms that eliminate material faces (A26) |
| Boundary between two liquids that do not mix | `pressure::MaterialBoundary` (no viscous term), `FindMaterialBoundaries`; supported, no exclusion (`ToName(PressureError)` covers 15 values) | pressure | `IngredientBoundary`, `viscous_jump`, `kMaterialBoundaryNotSupported` |
| Sealed gas inside the liquid solve (A18) | `pressure::SealedGasBlob`, `EquationInputs::sealed_gas_blobs`, `sealed_gas_blob_of_cell`, `gas_pressure_above_background`; filled by `gas::ApplyGasStepStart` returning `SealedGasForLiquid`; solved by `SolvePressure` and, since A24, by `viscosity::SolveViscosityAndPressure` (one unknown per blob; g.2) | pressure, gas, viscosity | sealed gas read one step late; `gas::SealedGasCoupling { kInsideLiquidSolve, kReadOneStepLate }`, `sealed_gas_coupling`, `ComputeSealedGasStepLimit` (A24) |
| Gas seen by the liquid where no sealed blob claims a cell | `EquationInputs::gas_pressure_above_atmosphere` (Pa), `holds_tracked_gas`; `AirPocketInputs::holds_tracked_gas`; filled by `gas::ComputeGasSeenByLiquid`, one step late | pressure, gas | |
| Blocking materials inside the liquid solve (A18) | `pressure::BlockingMaterialFace`, `EquationInputs::blocking_material_faces`; `solids::BlockingSet`, `BuildBlockingMaterialFaces`, `ToBlockingMaterialFaces`, `ApplyPressureToMaterials(const BlockingMaterialFaces&, velocity_after_pressure, blocking_sets, SolidsBuffers&)` returning `PressureOnMaterials`; solved by `SolvePressure` (faces eliminated, A18) and, since A24, by `viscosity::SolveViscosityAndPressure` (since A26 as velocity unknowns: see "Material faces in the viscous solve") (g.2) | pressure, solids, viscosity | `MaterialPushFields`, the particle form of `ApplyPressureToMaterials`, liquid pressure on snow and springy solids one step late |
| Equations and blobs | `ComputeEquationIndexes`, `kNoEquation`, `ComputeBlobOfCell`, `kNoBlob`, `SealedBlobCheck { kWhenBuilt, kWhenRatesApplied }`, `CellContent::kGas`, `kNoAirPocket`, `ComputeBlockedCornerAreas` | pressure | `border_cell_pressure`, `kLiquidToBorderCell` (removed) |
| The solve with viscosity | `viscosity::SolveViscosityAndPressure(inputs, settings, cancel_requested, buffers, velocity_after_pressure, cell_pressure, SolvedCouplingArrays coupling_arrays = {})` (A24: carries the blocking-material faces and the sealed gas blobs; `SolvedCouplingArrays { blocking_material_velocity_after_pressure; sealed_gas_pressure; }`, `ViscosityBufferSettings::max_sealed_gas_blobs`; g.2); A26, A27: `ViscosityInputs` gains `std::span<const solids::ViscousMaterialFace> material_faces` and `std::span<const solids::FreeSolidForViscousSolve> free_solids`, `SolvedCouplingArrays` gains `material_face_velocity_after_solve` (one per `ViscousMaterialFace`) and `free_solid_motion` (`std::span<solids::FreeSolidMotion>`, one per free solid), `ViscosityBufferSettings` gains `max_material_faces`, `max_free_solids` (g.9); `cells_from_particles` is `std::span<const transfer::CellsFromParticles* const>` with exactly one entry | viscosity | `ApplyViscosity`; the six-argument form |
| Viscosity exclusions | No exclusion by material (A27): snow, springy solids (A24) and free solids (A27) beside a viscous liquid run; `kCancelled`; only the three geometry exclusions the viscosity plan tries to lift (inclined solid, driven solid cutting cells, size change at a surface, solid or viscosity change) remain if their lifting tests fail; the sand module adds `has_sand`, `kSandNotSupported` (g.9) | viscosity | `has_snow_or_sand`, `kSnowOrSandNotSupported`; `has_snow`, `kSnowNotSupported` (A24); `has_free_solid` / `has_free_solid_touching_liquid`, `kFreeSolidNotSupported` (A27) |
| Material faces in the viscous solve (A26) | `solids::ViscousMaterialFace { face_piece_index; material_set; face_mass; velocity_before_pressure; blocking_face_index; }` (`solids/viscous_material_faces.h`): the band faces of each blocking set (face mass, centre within `kFitReach` cell widths of a liquid cell centre), velocity unknowns of the joint solve, the liquid sticks there; `ViscousMaterialFaces` (move-only view into `SolidsBuffers`), `BuildViscousMaterialFaces(const BlockingMaterialFacesInputs&, const BlockingMaterialFaces&, const pressure::GeometrySnapshot&, SolidsBuffers&)`, `ToViscousMaterialFaces`, `ApplyViscousSolveToMaterials(const ViscousMaterialFaces&, std::span<const double> velocity_after_solve, std::span<const BlockingSet>, SolidsBuffers&)` returning `PressureOnMaterials` (g.9) | solids, viscosity, step | material faces eliminated in the joint solve; the liquid sliding along a material (A24) |
| Free solids in the viscous solve (A27) | `solids::FreeSolidForViscousSolve { solid_index; mass; inertia_at_start; origin_at_start; velocity_before_pressure; turn_rate_before_pressure; allowed_motions; }` (`AllowedMotions allowed_motions` instead of the guide flags, A59, 2026-11-15), `ToFreeSolidsForViscousSolve(const MotionMap&)`, `FreeSolidMotion { velocity; turn_rate; }`, `ApplySolvedMotionToSolids(const MotionMap&, std::span<const FreeSolidMotion>, const grid::CellValues<double>& pressure, SolidsTry&, SolidsBuffers&)` (`solids/motion_map.h`); `m = column_count` motion unknowns per free solid touching viscous liquid (six when unguided; A59, 2026-11-15), the liquid sticks at its surface (g.9) | solids, viscosity, step | `kFreeSolidNotSupported`, `kViscousLiquidWithFreeSolid` (A27); `FreeMotionLimits limits` held at their value before pressure after the solve (A59) |
| Guided rigid bodies (A59, 2026-11-15) | `solids::GuideKind { kAxisLimits, kRail, kHinge }`; `FreeMotionLimits` keeps its axis flags and gains `guide_kind`, `rail_direction`, `hinge_point`, `hinge_axis` (world frame); shared `guide_math.h`: `AllowedMotions { columns[6][6]; column_count; prescribed_motion[6]; }`, `ComputeAllowedMotions(const FreeMotionLimits&, Vec3 origin_at_start)`; motion `ξ = J q + ξ_0`; the viscous block `Jᵀ (M + Δt Rᵀ E_sᵀ W E_s R) J`; the pressure solve's free-solid term `(K J) (Jᵀ M J)⁻¹ (K J)ᵀ` (rank m); a hinged body's pose rotates about `hinge_point` at commit; scene `free_solid.guide`; add-on `guide_kind`, `guide_object` (g.10.4) | solids (used by viscosity, pressure, step, I/O, Blender add-on) | axis flags only; rank-6 term `K M⁻¹ Kᵀ`; "a motion a guide forbids keeps its value before pressure after the solve" |
| Force of the liquid on every solid (A60, 2026-11-15) | `solids::LiquidForceOnSolid { solid_id; pressure_force; pressure_torque; viscous_force; viscous_torque; }` (N, N m; mean over the step; world; torque about the body's origin at the step start), every solid ascending `solid_id`, fixed and domain box included; `SolidsTry::liquid_forces()`; `step::StepReport::liquid_force_on_solids`; `ViscosityResult::viscous_push_on_solids` returns one `solids::ViscousPushOnSolid { push; turning_push; }` per solid (N s, N m s), stored by `solids::ApplyViscousImpulses` right after `ApplySolvedMotionToSolids` (A60, 2026-11-15; g.11.5); memory entry `"viscosity: solid sample coefficients"` 52 B per solid sample, `kRunOn`; I/O `liquid_force_on_solids` in `frame_done` and `run_report.json` (g.10.4) | solids, viscosity, step, I/O, Blender add-on | force reported for free solids only |
| Stick and slip on prescribed-motion solids (A53, 2026-11-15) | `solids::SolidSettings::slip_length` (m; 0 full stick, default; `+infinity` free slip; read for fixed and driven solids and the domain box beside a viscous liquid; a value on a free solid gives NOTE `slip_ignored_on_free_solid`); the viscous solve's tangential coefficient `μ / (d + λ)`; scene `slip_length` per solid (`"free"` or a number); add-on `slip_length`, `free_slip` | solids, viscosity, I/O, Blender add-on | full stick on every solid beside a viscous liquid |
| What counts as viscous (A25) | A liquid ingredient is viscous exactly when its `viscosity` is above 0; the scene is viscous when any liquid is (stage 11, else stage 12), chosen once per run; `ViscositySceneSummary::has_viscous_ingredient`; `viscosity::ComputeViscousEffectPerStep(viscosity, step_duration, density, cell_width)` (`deformation_rate.h`, shared); viscosity advice (A57, 2026-11-15; `viscosity_advice.h`, CPU only): `kNegligibleViscosityRatio = 0.1`, `kSmallestVisibleWavelengthInCells = 4`, `kNegligibleViscousChange = 0.01`, `ComputeViscosityToSmoothingRatio`, `ComputeFourCellFeatureChange`, `ComputeReynoldsNumber`; `step::ViscousPath { kPressureSolve, kViscousSolve }`, `step::LiquidViscousEffect` (A57 fields, 2026-11-15: `viscosity_to_smoothing_ratio`, `four_cell_feature_change`, `spreading_in_cells`, `reynolds_number`, `advises_zero_viscosity`), `RunPlan::liquid_viscous_effects`, `viscous_effect_cell_width`, `viscous_effect_step_duration`, `viscous_effect_run_duration`, `reynolds_length` (A57, 2026-11-15), `viscous_path`; I/O `RunBudget::viscous_path`, `liquids` (`io::ViscousEffectOfLiquid`, schema table `ViscousEffectOfLiquid`, with the A57 fields in place of `effect_not_visible`), `viscous_effect_run_seconds`, `reynolds_length_m` (A57, 2026-11-15) and the `liquids:` budget lines with the advice and the slip note, `ToName(step::ViscousPath)`; the add-on's `RunBudgetFacts.liquids`, `viscous_path` (g.8, g.10.3) | viscosity, step, I/O, Blender add-on | the water preset's 1e-3 Pa s; `io::LiquidViscousEffect` and the schema table `LiquidViscousEffect` (same name as the step's struct, other fields); `kSmallestVisibleViscousEffectPerStep = 2.5e-4`, `effect_not_visible`, the A25 "note" line (A57) |
| Material kinds | `materials::MaterialKind { kLiquid, kGas, kSnow, kSpringySolid }` (the sand module appends `kSand = 4`) | materials | five values with `kSand` |
| Stretch limits | `materials::LimitCase { kInsideLimits, kClamped }`, `kLimitCaseCount = 2` (sand appends two); headers `stretch_response.h`, `snow_material.h` | materials | four values, `snow_and_sand_material.h` |
| Material words | `springy_stretch`, `springy_deformation`, `permanent_volume_change`, `internal_force_per_area`, `deformation_rate` (`stretch`, `shear`), `rest_volume`, `volume_ratio`, `friction_angle`, `cohesion` (A4) | materials | textbook words |
| Internal force | `ApplyInternalForce(..., transfer::GridFromParticles&, std::span<InternalForceTotals> totals_of_block)` returning `std::expected<void, MaterialsFailure>`, `ComputeInternalForceTotal` | materials | one returned budget |
| Ingredient table | `MaterialTable::first_ingredient_of_material`, `ingredient_viscosity`, `ingredient_rest_density`, `gas_ingredient_index()`, `ingredient_export_names()` | materials | |
| Solids' scene check | `std::vector<SolidsFailure> CheckSolidsCanRun(const SolidsSettings&, double smallest_cell_width, double largest_cell_width)` (empty: can run); `kMaxPlateCount = 1`, `kMaxFreeSolidCount = 16`; `SolidsSettings::has_liquid_or_gas` | solids | `std::expected<void, ...>` form, `max_free_solid_count` |
| Solids in a step | `ComputeStepDurationLimit`, `BuildTry(solids, step_duration, smallest_cell_width)`, `BuildMotionMap` with `ToFreeSolidPushes`, `ToFreeSolidInertia` (A59, 2026-11-15; g.11.5), `ApplyContactReactions`, `BuildBlockingMaterialFaces` (stage 10) and `ApplyPressureToMaterials` (stage 12; g.2), in a viscous scene also `BuildViscousMaterialFaces` and `ToFreeSolidsForViscousSolve` (stage 10) and, at stage 11, `ApplySolvedMotionToSolids`, then `ApplyViscousImpulses` (A60, 2026-11-15; g.11.5), and `ApplyViscousSolveToMaterials` in place of `ApplyPressureToSolids` and `ApplyPressureToMaterials` (A26, A27; g.9), `CommitTry` returning `std::expected<void, SolidsFailure>`, `SolidsSummary`, `BuildSolids(const SolidsSettings&)`, `DriveWaves` | solids | `ComputeSolidsForStep`, `ComputeStepLimits`, `BuildFreeSolidPushes`, `SolidsSceneSummary`; `ToFreeSolidInverseInertia` (A59); `ApplyPressureToMaterials` at stage 11 (A24, replaced by A26) |
| Solids for other modules | `BuildContactSolids(std::span<const SolidSettings>, const SolidsSnapshot&)`, `ComputeRotationVector(Orientation from, Orientation to)`, `ComputeSolidsAtFaces(grid, range, solids, SolidsAtFaces&)`, `grid_edge_is_open_to_sky` | solids | `contact::BuildContactSolids` |
| Gas in the step | `gas::FindMarkerChanges`, `FindCellsToKeepActive`, `BuildTry`, `ComputeGasSeenByLiquid` (in `ChooseActiveCells`), `ApplyGasStepStart` (stage 10, or the start of stage 13 without stage 10; g.2), `SolveGas(inputs, settings, cancel_requested, SolvedGasArrays{velocity_after_pressure, was_solved, cell_was_solved}, amounts_try, gas_buffers, pressure_buffers)`, `GasResult::step_limit` (no sealed term, A24), `GasAmountsTry` | gas | `AddGasFromSources`, `ComputeCellsNeeded`, `ApplyGasToGrid`, `ApplyGasForces`, `ComputeGasGeometry`, `ComputeAddedVolumeRate`, `MoveGas`, `gas::ComputeStepLimit`, `gas::GasTry` |
| Gas state and output | `gas::GasAmountsSnapshot` with `ReadGasAmountsSnapshot`, `BuildGasAmountsFromSnapshot` (saved state); `GasSnapshot` for volumes, from `ToGasSnapshot(const GasAmountsSnapshot&, const GasLaw&, const grid::GridFrame&)`; render output passes the law through `FrameInputs::gas_law` | gas | `GasSnapshot` in the step snapshot |
| Gas sources and wind | keyed `gas::GasSource { source_id; low_corner; high_corner; keys; }`, `gas::GasWind { keys; }`, `SampleGasSource`, `SampleGasWind`, `CheckGasSources`; the scene file's top-level `wind: [{time, acceleration}]` (renamed from `gas_wind`, A47, 2026-11-15; s from the first frame; m/s2, world, Z-up) is written by the Blender add-on from Blender's Wind force fields whenever one exists (`build_wind_keys(readings)`, no `has_gas`; `WIND_ACCELERATION_PER_STRENGTH = 1.0` m/s2 per unit of Strength, A20) and read as written: neither I/O nor gas converts; `ReadyScene` keeps `gas::GasWind wind`; wind acts on gas in the solver and on spray and foam in render output (`render_output::RenderSettings::wind`, filled by I/O; `kFoamWindDriftDuration = 0.1` s, starting value), never on the liquid body; findings `wind_without_gas` (NOTE, A47 text) and `wind_flow_not_honoured` only when `field.flow != 1.0` (A50) (A47, A50, 2026-11-15) | gas, render output, I/O, Blender add-on | box with rates, one acceleration; the add-on's `wind_acceleration`; scene field `gas_wind`; wind keys written only with gas |
| Stages | `step::Stage` (section e); `TryStepWithStages(Simulation&, const StepInputs&, const StageTable&)` | step | `TryStepWithStages(simulation, request, stages, ranges)` |
| Callbacks | `step::Callback<Function> { Function* function; void* context; }`; `StepCallbacks` members are pairs; no `choose_cell_size` (A41, 2026-11-15: stage 2 fills new cells with `grid::FindNewCellNeighbors`, A118); `apply_grid_only_region`, `move_grid_only_liquid`, `read_largest_grid_only_speed` are not in the first version (A64, 2026-11-15); `limits_raised` (`Callback<bool(void*, Simulation&, const SimulationLimits&)>`, called by `ApplyLimits` before the old blocks are freed) with `FindRunMemory(Simulation&)` / `FindCpuRunMemory(Simulation&)` (A58 reconcile, 2026-11-15; g.11.6) | step | one `void* context`, `cell_resize::CallbackContext` used outside cell resize; `choose_cell_size` (A41) |
| The run plan | `step::BuildRunPlan(..., std::span<const MemoryEstimateEntry> output_estimate, std::span<const MemoryEstimateEntry> cell_resize_estimate = {})`; `SceneCounts`' particle estimate counts every liquid cell of the start grid at the levels the depth term gives, times the seeded count, plus the sources (A64, 2026-11-15), **replaced by** the exact start state: `SceneCounts::{start_cells_per_level, start_particle_count, start_cell_blocks}` from `step::BuildStartCells` and `io::BuildStartParticles`, 8 particles per liquid cell at every level, limits 1.25 x these plus sources (A65, A66, 2026-11-15; g.11.1, g.11.2); last parameter `std::optional<RunMode> forced_mode = std::nullopt` (A40, 2026-11-15; g.11.6) | step | the start counted at the smallest cell (A64 text) |
| Particle size (A65, 2026-11-15) | "every liquid cell holds about eight particles of its own size: a particle's `size_level` is its cell's level, its rest volume is one eighth of its cell (`(w/2)³`), so a cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest particle's volume" (g.11.1); `pressure::SurfaceDistanceInputs::particle_radius_in_cell_widths` (radius = it x the particle's level width) | particles, step, pressure, I/O | one particle size everywhere; the `w_min / 2` lattice over the whole liquid; `SurfaceDistanceInputs::particle_radius` (m) |
| Start cells and seeding (A66, 2026-11-15) | `step::StartRegionShape`, `StartCells`, `BuildStartCells` (`step/start_cells.h`, CPU only); `io::BuildStartParticles` (8 points per start cell at its cell-eighth centres); the grid's first version from `StartCells::cells` (g.11.2) | step, I/O, grid | I/O 5.1's `particle_spacing = smallest_cell_width / 2`; the first grid from particles at `new_cell_size_level` |
| Size change report (A67, 2026-11-15) | `step::StepReport::size_change_kinetic_energy` (J), `StepReport::mismatched_particle_count`; memory row `"particles: merge wait passes"` (g.11.3) | step, particles, I/O | |
| Later work and the slab limit (A68, 2026-11-15) | later groups in the order "surface band 8 to 4 (test only)", "Coarser cells on calm surfaces" with "Batches cut by columns or tiles", "Deep liquid without particles (needs review)"; the budget line naming the plan-area limit `a_max = room x 16 w_min² / (layers x bytes_per_loaded_block)` (g.11.4) | cell resize and batches, I/O | |
| Guided free solids in the pressure solve (A59, 2026-11-15) | `pressure::FreeSolidInertia`, `EquationInputs::free_solids`, `GuidedInverseMass`, `ComputeGuidedInverseMass`, `ComputeGuidedSolidMotion`; `solids::ToFreeSolidInertia(const MotionMap&)` (g.11.5) | pressure, solids | `free_solid_inverse_inertia`, `ToFreeSolidInverseInertia` |
| Memory budget (A58, 2026-11-15) | `StepSettings::memory_share_of_free = kDefaultMemoryShareOfFree` (0.9), `StepSettings::memory_left_free_bytes = kDefaultMemoryLeftFreeBytes` (536,870,912 B, starting value); `step::ComputeAllowedMemoryBytes(MemoryAvailable, double memory_share_of_free, std::uint64_t memory_left_free_bytes)` = `min(share x free, free - left_free)` (0 when negative), the same rule on both processors (`run_plan.h`); suggestion `more_memory_bytes = max(ceil(needed / share), needed + left_free) - free`; the add-on's `memory_share_of_free` (0.9) and `memory_left_free_bytes` (MiB) | step, I/O, Blender add-on | `StepSettings::memory_share_of_total` (0.7), `memory_share_of_free` 0.8, CPU rule min(0.7 total, 0.8 free) |
| Committed limits (A41, 2026-11-15) | `step::CommittedStepLimits::largest_liquid_speed` (m/s; exact max of particle speed over liquid materials at the last commit, filled by `CommitStep`), read by `ResizeCells` for `grid::ComputeReferenceSpeed`; `CommittedStepLimits::grid_only_speed` is not in the first version (A64) | step, cell resize | `grid_only_speed`; `ReadLargestGridOnlySpeed` |
| Step grid fields (A63, A64, 2026-11-15) | `step::GridFields` gains, committed and try copies beside `merge_ready_duration`: `grid::CellValues<double> held_velocity_gradient_size` (1/s), `grid::CellValues<std::uint8_t> merge_ready_step_count` (saturating at 255); memory rows `"step: held velocity gradient size"` 8 B x 2, `"step: merge wait steps"` 1 B x 2 per cell place, `kRunOn`; saved in the state. The grid-only arrays (`is_grid_only_region`, `layers_from_grid_only_region`, `grid_only_mass`, `grid_only_rest_volume`, `grid_only_current_volume`, `grid_only_ingredient_volume`) and the mirror `"step: committed grid-only marks on the CPU"` are not in the first version (A64) | step, cell resize | |
| Step snapshot | `step::SimulationSnapshot` with `std::optional<gas::GasAmountsSnapshot> gas` and the committed `held_velocity_gradient_size`, `merge_ready_step_count` (A63, 2026-11-15); `is_grid_only_region`, `grid_only_current_volume`, `grid_only_ingredient_volume` are not in the first version (A64, 2026-11-15) | step | |
| Capability names | `step::CapabilityLevel { kCanRun, kExperimental, kUnavailable }` (the full list); `CapabilityReason::kMaterialSetsNotValid`; `SuggestedChange::kPutAllLiquidsInOneSet`; `kCapabilityReasonCount = 16` at most (A24 removed `kViscousLiquidWithSnow`, A27 `kViscousLiquidWithFreeSolid`; lower by one for each of `kViscousLiquidWithInclinedSolid`, `kViscousLiquidWithMovingSolidThroughCells`, `kViscousLiquidAtSizeChange` the viscosity plan lifts, 13 if all three are); `kRejectReasonCount = 23`, `kSuggestedChangeCount = 12` (A27: no `kSetViscosityToZero`; the sand module appends it back with `kViscousLiquidWithSand`, stable name `set_viscosity_to_zero`, and the add-on's repair), `kStepLimitCount = 11` | step | `liquids_that_do_not_mix_are_blended`, `kLiquidsThatDoNotMixTouch`, `step::Support`, `kSupported`; `kViscousLiquidWithSnow`, `"viscous_liquid_with_snow"` (A24); `kViscousLiquidWithFreeSolid`, `"viscous_liquid_with_free_solid"`, `kSetViscosityToZero` and `"set_viscosity_to_zero"` in the main sequence (A27) |
| Transfer buffers and builders | `transfer::TransferBuffers`, `BuildTransferBuffers`, `BuildSolidsAtFaces`, `BuildGridFromParticles`, `BuildGridSnapshot(..., FilledFromSolved)`, `BuildCellsFromParticles`, `ToSolidsAtFacesOnProcessor`, `ToGridSnapshotOnProcessor`; each stage has one form, taking `TransferBuffers&` last (g.5; A31); each builder takes `RunMemory& run_memory` last (A58, 2026-11-15) | transfer | `TransferScratch`, `BuildTransferScratch`; the stage forms without working buffers |
| Particle limits and amounts | `particles::ApplyParticleLimits(Particles&, max_particle_counts, max_cell_blocks)`; `ComputeAmounts` -> `std::expected<ParticleAmounts, ParticlesFailure>`, `ComputeAmountsPerBlock` -> `std::expected<void, ParticlesFailure>` (g.5) | particles | NaN signalling |
| Solids buffers | `solids::BuildSolidsBuffers(settings, max_particle_count, max_cell_blocks, RunMemory& run_memory)`, `BuildSolidCoverageValues(RunOn, max_cell_blocks, RunMemory& run_memory)` (A58, 2026-11-15), `MotionMap` (move-only view into `SolidsBuffers`), `WriteBlockPartialsRange` (g.5, g.6) | solids | |
| Step plumbing | `step::RandomWords`, `ComputeRandomWords`, `ComputeSignedUnitValue`; `StageOutcome` (`step_error.h`); `StageTable::run_whole_grid`; `kStages`; `FindLoadedTry`, `FindLoadedGridBuffers`, `BlockCopy`, `WriteStepArrayBlocks`, three-argument `ApplyGridPiece` (g.5, g.6) | step | two-argument `ApplyGridPiece` |
| Buffers in batched runs | placement rule; `step::ArrayPlace`, `StepArray::whole_place`; `FindCpuGridCopy`, `FindContactReactionBuffers`; `gas::MarkerGridInputs` and the CPU form of `FindMarkerChanges` (g.6) | step, solids, gas | CPU block ids of a GPU grid |
| Resized grid handed to the step (reconcile) | `step::ResizedCells { cells; cell_buffer; cell_buffer_count; cpu_resized_grid; }`, `BuildResizedGrid(ResizeTry&, const ResizedCells&)`, `ApplyResizedCpuGridCopy(ResizeTry&, grid::StaggeredGrid&&)` (`resize_access.h`); `cell_resize::GridCopies { kNone, kHeldByCellResize, kHeldByStep }`, `CellResizeSettings::grid_copies`, `BuildCellResizeMemoryEstimate(ingredient_count, grid_run_on = RunOn::kCpu, grid_copies = GridCopies::kNone)` (g.7); kept by A41 (2026-11-15): `FindCellsThatMustSplit`, the list edit, `BuildResizeMap`, grouping and the marker upkeep still read CPU tables in a GPU run; only `choose_cell_size` stops being a reader; on a GPU grid `ComputeCellSizeChanges`' `size_change` bytes are read back in one counted copy (1 B per cell place) | step, cell resize | the CPU grid copy refreshed with `grid::ReadGrid` after a committed resize (Appendix B row B5) |
| Copy report reset (reconcile) | `pressure::ResetSolveCopyReport(PressureBuffers&)`, `viscosity::ResetSolveCopyReport(ViscosityBuffers&)`, `gas::ResetSolveCopyReport(GasBuffers&)`: zero the held `SolveBatches`' report, nothing for one-piece buffers; the step calls them per try (g.3, g.7) | pressure, viscosity, gas, step | |
| Copies counted | `step::CopyCounts`, `StepReport::copies`, `ReadCopyCounts`, `RunPlan::planned_step_copies`; `copy_count` / `bytes_copied` in `MarkerChanges`, `AddParticlesResult`, `ResizeCellsReport`; `render_output::ReadRecorderCopyCounts` (g.6) | step, gas, particles, cell resize, render output | |
| Plan that does not fit | `step::PlanFit { kFits, kDoesNotFit }`, `RunPlan::fit`; `MemoryEstimate::on_cpu`, `ModuleBytes::on_cpu`, `MemoryReport::allocated_cpu_bytes`; `BuildStepMemoryEstimate(..., const RunPlan* batched_plan = nullptr)` (g.6); `MemoryEstimate::on_processor` is `ComputeRunMemoryBytes` (rounded per piece) (A58, 2026-11-15); when particles are the largest processor entry `BuildBudgetLines` adds the line `particles: <n> needed, <m> of them in cells deeper than <8 w_min> m below the surface; in this version every liquid cell holds particles (deep liquid without particles comes later)` (A64, 2026-11-15) | step, I/O | `BuildRunPlan` failing when nothing fits |
| Reserved memory measured (A58, 2026-11-15) | `step::MemoryReport` gains `planned_bytes`, `reserved_bytes`, `handed_out_bytes`, `reserved_block_count`, `memory_taken_bytes` (GPU only), `requested_beyond_plan_bytes`; `allocated_bytes` becomes `handed_out_bytes`; `BuildSimulation` builds `run_memory` before anything else; `ApplyLimits` builds a new block for the whole new plan and frees the old one whole (check `new reserved + old reserved <= allowed_bytes`) | step, I/O | `MemoryReport::allocated_bytes`; per-buffer allocation and freeing at a limit raise |
| Planned and measured cost | `io::RunBudget` and `io/run_budget.h`, `io::RunCost` and `io/run_cost.h`, `BuildSceneBudget`, `RunOverrides::assume_cpu_memory_bytes`, `check --assume-cpu-memory`; schema `run_budget.fbs`; the add-on's `RunPlanFacts.budget` (g.6); warnings without margins (A43, A58, 2026-11-15): `requested beyond the planned`, `copies per step ... (max) are above the planned`, the same per solve iteration, `gpu memory taken ... is less than the ... reserved` (`kMemoryTakenToleranceBytes = 2,097,152`, starting value), the cost line `gpu memory: planned, reserved in <n> block(s), handed out, taken`, the Windows fallback note; `RunCost` gains `planned_bytes`, `reserved_bytes`, `reserved_block_count`, `handed_out_bytes`, `memory_taken_bytes`, `requested_beyond_plan_bytes`; `MemoryBudget` gains `memory_share_of_free`, `memory_left_free_bytes` (g.10.2) | I/O, Blender add-on | `kMemoryWarningShare = 1.10`, `kCopyWarningShare = 1.25` (A43, A58) |
| Material presets and frame seams | `io::MaterialPreset { kCustom, kWater, kSyrup, kSnow, kSpringySolid, kAir }`, `kMaterialPresets` (`kWater`: 1000 kg/m3, 0 Pa s, 0.072 N/m; A25), `FindMaterialPreset`; `FrameWriter` (with `start(void* context)`), `FrameOutput`, `KeptFrameInputs`, `cli::BuildFrameWriterWithoutProducts()`, `RunParts` (g.5) | I/O | |
| Run modes | `step::RunMode { kOnePiece, kBatched, kBatchedWithSolvesInPieces }` (I/O `ToName`: `"one_piece"`, `"batched"`, `"batched_with_solves_in_pieces"`; `ToName(step::Module)`, 14 lower-snake names; reconcile); `RunPlan::max_solve_loaded_blocks`, `allowed_cpu_bytes`; `batches::SolvePlace { kWholeGridOnProcessor, kInPieces }` (g.3); `resume` plans with the recorded `run_mode` forced (A40, 2026-11-15) | step, batches, I/O | `batches::SolvePlace::kWholeGridOnCpu` |
| Solves in pieces | `pressure::SolveBatches` and `solve_batches.h` (`BuildSolveBatches`, `ApplySolveBatches`, `LoadSolveBatch`, `WriteBatchValues`, `ReadBatchValues`, `WriteBatchGroups`, `ReadBatchGroups`, `JoinLabelsAcrossBatches`, `SolveCopyReport`), `max_loaded_blocks` in `BufferSizes`, `ViscosityBufferSettings`, `GasBufferSizes`; `WriteSnapshotRange` (g.3) | pressure (used by viscosity, gas, batches, step) | "the solves are not batched" |
| Grid-only region: **not in the first version (A64, 2026-11-15)** | every liquid cell holds particles; `is_grid_only_region` (`std::uint8_t` per cell place), `RegionSettings`, `kMaxFilledShare`, `GridOnlyAmounts`, `cell_resize::ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep`, `ReadLargestGridOnlySpeed`, `kGridOnlyCellDrained`, `kGridOnlyCellTooFull`, the I/O and add-on fields `particle_region_depth_in_cell_widths`, `grid_only_depth_in_cell_widths` move to the cell resize and batches plan's group "Deep liquid without particles (later; needs review)" (gate tests `GridOnlyRegion, StandingWave_KeepsAmplitude`, `GridOnlyRegion, ParticleBandEdge_StaysStable`); other plans' grid-only pieces move to a group of the same name pointing at it | cell resize (later) | `grid::CellRegion` |
| Liquid totals | `cell_resize::ComputeLiquidTotals(const particles::ParticlesTry&, std::span<const std::uint8_t> material_is_counted, const grid::StaggeredGrid&, grid::CellBlockRange, SumBuffers& sum_buffers)`, the only form (A31), over particles only (A64, 2026-11-15; the `const GridOnlyAmounts&` parameter returns with the later group); `ComputeDepthBelowSurface` stays | cell resize | four-argument form; the five-argument form without `SumBuffers`; the form with `const GridOnlyAmounts&` (A64) |
| Foam and spray recording | `render_output::RecordStep(StepRecorder&, const step::FoamAndSprayInputs&)`; `FoamAndSprayInputs::after_pressure` a span with one entry; `CheckRenderSettings(const RenderSettings&, double smallest_cell_width, const materials::MaterialTable&)`; `RenderSettings::wind` (`gas::GasWind`, filled by I/O from `ReadyScene::wind`): spray `v += Δt (g + a_wind(t + Δt / 2))`, foam drift `kFoamWindDriftDuration x` the horizontal part of `a_wind` (A47, 2026-11-15); render output's grid-only surface inputs are not in the first version (A64, 2026-11-15) | render output, step | `RecordStep(recorder, inputs, ranges)` |
| Open-format writers | `WritePointsFile`, `WriteSurfaceFile`, `WriteVolumeFile`, `WriteSolidMotionFile` in `fluidity_interchange`; `ReadFormatChoice(kind)`; `kOutputContractVersion = 2`; `InterchangeLibraries::missing_resource_folder`; each library pinned to the newest release on Conan Center when pinned (A19) | render output | I/O's `ParticlesFrame`, `MotionFrame`; versions matched to Blender's bundled libraries |
| Witness records | `render_output::WitnessRecord { property; blender_5_3; evidence; }`; `capabilities` prints `{"property", "blender_5_3", "evidence"}`; the add-on's `WitnessRecordFact`, `WitnessCaseResult`, `write_witness_records.py` (one Blender 5.3 results file) (g.4) | render output, I/O, Blender add-on | `blender_5_2`, `WitnessOutcome` |
| Output channel value type | `render_output::ChannelValueType`, field `OutputChannel::value_type`; the add-on's `OutputChannelFact::value_type` | render output, Blender add-on | `ChannelStorage`, `storage` |
| Run folder words | a file a reader can see is `listed`; `WriterLock`; `WriteListedFile(RunFolder&, OutputFile&&, CheckAfterWrite)`; `last_listed_frame`; `lock_holder.is_held` | I/O | "publish", `published_through`, "lease" |
| CLI | `capabilities`, `check [--cache <run_folder>]`, `run <scene_file> --into <run_folder>`, `resume [--choose-mode-again]` (`Command::choose_mode_again`, A40, 2026-11-15), `report`, `read-particle`; events `started`, ..., `frame_done` (gains `liquid_force_on_solids`, A60, 2026-11-15), `done` / `stopped` / `failed`; exit codes 0, 1, 3, 4, 5, 6 | I/O | `read-cache`, `run_started`, `frame_published`, ... |
| Resume in the recorded mode (A40, 2026-11-15) | `resume` plans with the recorded `run_mode` forced; if it does not fit it fails `IoError::kRecordedModeDoesNotFit` (appended at the end of `IoError`; code `recorded_mode_does_not_fit`) naming `--choose-mode-again`; with the flag the mode is planned afresh and the run record gains `run_mode_changes: [{frame, from, to}]`; `kWrongRunFingerprint` with `kRunMode` is not raised for that change; the add-on's "Resume in another mode" | I/O, Blender add-on | a mode change refused by the fingerprint |
| Scene file fields (2026-11-15) | `wind` (renamed from `gas_wind`, A47); ~~`cell_size_regions`, `cell_size_map`~~ (A63, replaced 2026-11-15 by the refinement fields, A78); `slip_length` per solid (A53); `free_solid.guide: {kind: "axis_limits" \| "rail" \| "hinge", moves_along_x, ..., turns, rail_direction, hinge_point, hinge_axis}` (A59); `particle_region_depth_in_cell_widths`, `grid_only_depth_in_cell_widths` are not in the first version (A64) | I/O, Blender add-on | `gas_wind` |
| Blender versions and the CUDA runtime (A48, A49, 2026-11-15) | manifest `blender_version_min = "5.3.0"`, no maximum; one NOTE "FluidiTy was tested with Blender 5.3 only." on a newer Blender; the executable links the shared CUDA runtime and the package ships `cudart64_<major>.dll` / `libcudart.so.<major>` in `bin/<platform>/` (Linux `RUNPATH $ORIGIN`; macOS none) and `licenses/cuda-runtime.txt`; CPU runs need no GPU driver | Blender add-on, packaging | "Blender 5.3 only" as a maximum |

## (c) What each module provides

Each paragraph names the module's section 3 for the declarations.

**Contact** (contact design, 3.1 to 3.5). One particle's motion against solids in one step. 3.1 errors
(`ContactError`, `ContactCloseCall`, `ContactFailure`); 3.2 the solid shapes (`Solid`, `FlatSolid`, `Plate`,
`Box`, `DomainBox`, `Sphere` and `Cylinder` (A74), `SolidMotion`, `SolidPose`, `PlateSide`, `SolidSide`, `DomainSide`); 3.3 the contact
tolerance; 3.4 `MoveParticle`, `FindFirstHit`, `ComputeDistanceToSolid`, `FindSolidSide`,
`ComputeSolidVelocitySample`, `CheckSolid`; 3.5 the batch `MoveParticles` on plain pointers, `ParticleContact`,
`SolidReaction`, `BuildContactBuffers`, `BuildContactMemoryEstimate`. Used by the step (tolerance checks, stage
`MoveParticles`), solids (faces, coverage, cutting, reaction sums), transfer and particles (`PlateSide`) and
render output (`contact::Solid` through `solids::BuildContactSolids`). Includes only `base/`.

**Grid** (grid design, 3.1 to 3.7). The mixed-size staggered grid. 3.1 ids and indexes, `GridVersion`; 3.2
`GridTables`; 3.3 `GridBuffers`, `BuildGridBuffers`, `BuildGrid(inputs, buffers)` on the run's processor (A17,
g.1), the move-only
`StaggeredGrid`, `TakeBuffers`, `CheckSizeStepRule`; 3.4 `CellBlockRange`, `BuildRanges`, `BorderWidth`,
`FindBorderBlocks`, `GridPiece`, `BuildGridPiece`; 3.5 weight curve and nearby-face weights; 3.6 the value
arrays, `ApplyGridVersion`, `BuildGridMemoryEstimate`; 3.7 helpers added for other modules; the shared cell size
rule `cell_size_rule.h` (g.10.1; A41, A61–A63, 2026-11-15). Every later module uses it; the step builds a version in
`ChooseActiveCells` and fills new cells from the top with `FindNewCellNeighbors` (A118); the cell resize and batches design builds resized
grids and pieces and calls `ComputeWantedCellWidth` per active cell.

**Particles** (particles design, 3.1 to 3.8). The particle arrays. 3.1 the views (`ParticleFields`,
`ParticleFieldsToChange`, `Field`, `FieldList`); 3.2 `Particles`, settings, `ApplyParticleLimits`,
`ComputeAmounts`, `ComputeAmountsPerBlock`; 3.3 `ParticlesTry`, `CommitTry`, `CheckParticles`; 3.4 grouping;
3.5 count changes (`AddParticles`, `RemoveParticles`, `FindNeighbourhood`, `SplitParticle`, `MergeParticles`,
`AdjustParticleCount`); 3.6 split, merge and refill math; 3.7 snapshots and `BuildParticlesMemoryEstimate`; 3.8
`WriteParticleGroups`, `ReadParticleGroups`. Contact, transfer and materials read and change the views; the
step adds, groups and commits; gas markers go through it; cell resize splits and merges inside its
transaction and batches copy groups; I/O and render output read snapshots.

**Transfer** (transfer design, 3). Particles to grid and back, once per material set. Material sets
(`MaterialSetKey`, `ToMaterialSetKey`, `CheckMaterialSets`); `GridFromParticles` and `FaceValueKind`;
`ParticlesToGrid`, `GridSnapshot`, `SaveGridBeforeForces`, `SaveGridAfterPressure`; `ComputeFilledVelocity`,
`GridToParticles`, `MixParticleVelocity`; `SolidsAtFaces` (filled by solids); `CellsFromParticles` with
`ComputeCellsFromParticles`, `ComputeMixtureDensity`, `ComputeSensedVolumeRatio`; `BuildTransferMemoryEstimate`.
The step runs stages `ParticlesToGrid`, `SaveGridBeforeForces`, `SaveGridAfterPressure`, `GridToParticles` per
set and mixes velocities in `MoveParticles`; pressure and viscosity read the liquid set's `CellsFromParticles`;
gas uses the gas set; cell resize writes the grid-only region into the liquid set (not in the first version, A64,
2026-11-15).

**Pressure** (pressure design, 3). Where the liquid is and the pressure that keeps its volume. `FindLiquidSurface`
and `GeometrySnapshot` (with `CellEighthGeometry` and `FacePieceCentre` in viscous scenes);
`ComputeDistanceToSurface` (all liquids, then per touching liquid material; all-zero mask for no liquid);
`FindMaterialBoundaries`; `FindAirPockets`; the open sea (`BuildOpenSea`, `ApplyOpenSeaRelaxation`); the
pressure law; `BuildPressureEquations`, `ComputeEquationIndexes`, `ComputeBlobOfCell`, `ApplyAddedVolumeRate`,
`ComputeBlockedCornerAreas`; `SolvePressure` (scenes with no viscous ingredient, and the gas solve); the pieces
of the joint solve; the blocking-material and sealed-gas terms inside `SolvePressure` (A18, g.2); the shared
solve-batch mechanism `solve_batches.h` (A15, g.3); `ComputeCarriedPressure`; `BuildPressureMemoryEstimate`. Used by the step, viscosity (the
pieces), solids (fills `SolidCoverage`, `FreeSolidPush`, `grid_edge_is_open_to_sky`), gas (fills the gas
arrays, reuses the solve), transfer (`was_solved`), cell resize (remaps carried pressure and pockets) and render
output (`distance_to_surface`).

**Viscosity** (viscosity design, 3). The joint velocity-and-pressure solve for scenes with a viscous ingredient
(A5, A6): `SolveViscosityAndPressure`, `CheckViscosityCanRun(ViscositySceneSummary)`, `BuildViscositySamples`,
`ComputePowerLostToViscosity`, `BuildViscosityMemoryEstimate`; the advice functions of `viscosity_advice.h`
(A57, 2026-11-15). It reads the liquid set's `CellsFromParticles` and
`MaterialTable::ingredient_viscosity()`, and calls pressure's pieces (the `CoupledVelocities` forms and
`ComputeCoupledPush`, A26, A27). Its velocity unknowns include solids' `ViscousMaterialFace` entries and
`FreeSolidForViscousSolve` motions in the guide's `AllowedMotions` (g.9; A59, 2026-11-15); it reads each fixed and
driven solid's `slip_length` (A53) and returns the per-solid viscous impulse (A60, 2026-11-15). Used by the step
(stage 11, capability check) and the planner (`ComputeViscosityToSmoothingRatio`, `ComputeFourCellFeatureChange`,
`ComputeReynoldsNumber`).

**Materials** (materials design, 3). Liquid, gas, snow and springy solid (A9). `MaterialKind`, `Ingredient`,
`MaterialTable` and its accessors, `BuildParticlesSettings`, `stretch_response.h`, `snow_material.h`,
`springy_solid_material.h`, `MaterialBuffers` (`internal_force_per_area` for transfer, `friction` for contact),
`ComputeInternalForces`, `ComputeParticleFriction`, `ApplyInternalForce`, `ComputeInternalForceTotal`,
`ComputeOwnMotionShares`, `ApplyMaterialResponse`, `ComputeCellSqueezeValues` (for pressure),
`CheckMaterialState`, `ComputeWorkRates` (test oracle), `BuildMaterialsMemoryEstimate`. Used by the step, transfer,
viscosity, pressure, gas, I/O and render output (export names). Section 5.9 names the extension points the sand
module uses.

**Solids** (solids design, 3.1 to 3.9). Fixed, driven and free solids and the contact between material sets.
3.1 errors; 3.2 settings and the public limits; 3.3 state and dynamics formulas; 3.4 `Solids`, `SolidsTry`,
`BuildSolids`, `BuildTry`, `CommitTry`, `CheckSolidsCanRun`, `ComputeStepDurationLimit`; 3.5 solids on the grid
(`ComputeSolidsAtFaces` filling `transfer::SolidsAtFaces`, `ComputeSolidCoverage` and `ToSolidCoverage` filling
`pressure::SolidCoverage` over every active cell, `FindGridEdgesOpenToSky` filling `grid_edge_is_open_to_sky`); 3.6 the motion map for free solids; 3.7
contact reactions; 3.8 `CollideSolids`, `DriveWaves`, plate sides and cutting; 3.9 the exchange check; the blocking-material faces of the liquid solve
(`BuildBlockingMaterialFaces`, `ApplyPressureToMaterials`, A18, g.2) and of the viscous solve (`BuildViscousMaterialFaces`,
`ToViscousMaterialFaces`, `ApplyViscousSolveToMaterials`, `ToFreeSolidsForViscousSolve`, `ApplySolvedMotionToSolids`,
A26, A27, g.9). Since 2026-11-15: guides in general form (`GuideKind`, `AllowedMotions`, `ComputeAllowedMotions`,
A59), the force of the liquid on every solid (`LiquidForceOnSolid`, `SolidsTry::liquid_forces()`, A60) and
`SolidSettings::slip_length` (A53) (g.10.4). Also
`BuildContactSolids`, `ComputeRotationVector`. Used by the step (stages `FindSolidSurfaces`, `CollideSolids`,
`DriveWaves`, `MoveSolids`), cell resize (`ComputeSolidsAtFaces` at committed poses), I/O and render output.

**Gas** (gas design, 3). Tracked gas (smoke, fire, sealed and open gas blobs) from the start beside liquid
(A10). Markers (`FindMarkerChanges`), active gas cells (`FindCellsToKeepActive`), the try (`BuildTry`,
`GasAmountsTry`, `CommitTry`), `ComputeGasSeenByLiquid` for pressure, `SolveGas` (stage 13, with pressure's
solve), the step limits (`GasResult::step_limit`; A24 deleted `ComputeSealedGasStepLimit`), keyed sources and wind, the
`ApplyGasStepStart` with the sealed blobs for the liquid solve (A18, g.2), the amounts snapshot for the saved state, `ToGasSnapshot` for volume output, the remap for cell resize
(`BuildResizeTry`, `GasArraysToRemap`, `ToArraysToRemap`), `BuildGasMemoryEstimate`. Used by the step, cell resize,
I/O (saved state, scene file keys) and render output (`ToGasSnapshot`).

**Step** (step design, 3.1 to 3.11). The `Simulation`, the stage order and try-and-commit across every module.
For I/O: `ReadyScene` and `StepSettings` (3.2, 3.3); `CheckSceneCanRun` with every finding, `CapabilityReason`,
`SuggestedChange` (3.4); `BuildRunPlan`, `RunPlan` (3.5); `BuildSimulation`, `ApplyLimits`,
`SimulationSnapshot`, `ReadSnapshot`, `BuildSimulationFromSnapshot` (3.6); `TryStep`, `StepResult`,
`RejectReason` (3.7, 3.1); `ComputeStepDuration`, `StepLimit` (3.8); `AdvanceTo`, `Callback`, `StepCallbacks`,
`FoamAndSprayInputs` (3.9). For cell resize and batches: `GridFields`, `ResizeTry`, `BuildResizeTry`,
`ApplyResizedGrid`, `CommitResize`, `RejectResize` (3.10); `StepTry`, `StageTable`, `TryStepWithStages`,
`StepArray` (3.11); for batched runs (g.6) `ArrayPlace`, `FindCpuGridCopy`, `FindContactReactionBuffers`,
`CopyCounts`, `ReadCopyCounts`, `PlanFit`. Since 2026-11-15: the memory budget (`kDefaultMemoryShareOfFree`,
`kDefaultMemoryLeftFreeBytes`, `StepSettings::memory_left_free_bytes`, `ComputeAllowedMemoryBytes`) and the run's
`RunMemory` with the new `MemoryReport` fields (A58); `CommittedStepLimits::largest_liquid_speed`,
`GridFields::held_velocity_gradient_size`, `merge_ready_step_count` (A41, A63); `RunPlan::viscous_effect_run_duration`,
`reynolds_length` and the A57 `LiquidViscousEffect` (A57); `StepReport::liquid_force_on_solids` (A60); no
`choose_cell_size` callback (A41) and no grid-only callbacks or arrays in the first version (A64);
`StepCallbacks::limits_raised`, `FindRunMemory`, `FindCpuRunMemory`, `BuildRunPlan`'s `forced_mode` (A58 reconcile,
A40); `StartRegionShape`, `StartCells`, `BuildStartCells` and the exact `SceneCounts` start fields (A66);
`StepReport::size_change_kinetic_energy`, `mismatched_particle_count` (A67) (all 2026-11-15; g.11). It calls every
earlier module through its section 3 and includes no cell resize, batches
or render output header: those are reached only through the callback pairs I/O sets.

**Cell resize and batches** (cell resize and batches design, 3.1 to 3.7; folders `cell_resize/`, `batches/`).
3.1 errors; 3.2 where small cells are wanted (moved to grid's `cell_size_rule.h`, A41, 2026-11-15; cell resize keeps
`ComputeVelocityGradientSize` and `ComputeCellSizeChanges`); 3.3 split and merge of cells; 3.4 the resize
transaction and the step callback `ResizeCells` (with a `CellResize*` as its context), `BuildCellResize` (its
run-processor part takes `RunMemory& run_memory` last, A58), `BuildCellResizeMemoryEstimate`; 3.5
`ComputeDepthBelowSurface` and `ComputeLiquidTotals` (over particles only); 3.6 the value remap;
3.7 batches (`BuildBatchPlan`, `BatchPlan`, `BatchedRun`, `ApplyStagesInBatches` with a `BatchedRun*` context). Not in
the first version (A64, 2026-11-15): the grid-only region of 3.4 and 3.5 (`ChooseCellSize` is removed by A41;
`ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep`, `GridOnlyAmounts`, `ReadLargestGridOnlySpeed` go to the
group "Deep liquid without particles (later; needs review)"); the later group "Coarser cells on calm surfaces
(later)" holds A61's curvature work. Used by I/O (sets the callbacks, copies
`BatchPlan` fields into `RunPlan`), the step (through callbacks) and render output (`ComputeLiquidTotals`).

**I/O and CLI** (I/O and CLI design, 3.1 to 3.8). The scene file and its schema, `RunScene` to
`step::ReadyScene` (3.2); the run folder, file list, `WriterLock`, `OutputFile`, `WriteListedFile`, the array
file for `saved_state` and internal records (3.3); fingerprints, `RunReport`, run-folder facts (3.4);
`SavedState`, `KeptFrameInputs`, `ReadParticle` (3.5); the output queue, `BuildRun`, `AdvanceRunTo`,
`RebuildOutputs` (3.6); the `fluidity` CLI, events, exit codes (3.7); `ToMessage` and `ToName` for every
module, with `stable_names.h` for the step's enums (3.1). Since 2026-11-15: `resume` in the recorded mode,
`IoError::kRecordedModeDoesNotFit`, `--choose-mode-again`, `run_mode_changes` (A40); the warnings without margins and
`kMemoryTakenToleranceBytes` (A43, A58); the A57 `liquids:` lines; `liquid_force_on_solids` in `frame_done` and
`run_report.json` (A60); `BuildStartParticles` and the start order of `BuildRun` (A66), the A67 report fields in
`frame_done` and `run_report.json`, the A65 particles line and the A68 plan-area line (g.11); scene fields `wind`, `refinement_*` (A78; were `cell_size_regions`, `cell_size_map`), `slip_length`,
`free_solid.guide`; `RenderSettings::wind` filled from the ready scene (A47). Used by the Blender add-on (CLI,
events, `report`) and by render output (its writers write into `OutputFile` sinks I/O opens and lists).

**Surface, foam and spray** (surface, foam and spray design, 3.1 to 3.6; folders `render_output/`, `surface/`,
`foam_and_spray/`, target `fluidity_interchange`). 3.1 errors; 3.2 lattice and motion sample; 3.3 recording
inside the step (`StepRecorder`, `RecordStep`, `CommitRecordedStep`, `RejectRecordedStep`,
`ReadRecordedInterval`); 3.4 the surface; 3.5 foam and spray particles; 3.6 `BuildRenderFrame`, `FrameInputs`
(with `gas_law`), volumes, output channels, the format table and witness records (one Blender 5.3 result each, g.4), `RenderSettings`,
`CheckRenderSettings`, `RenderOutputSnapshot`, `BuildRenderOutputMemoryEstimate` and the open-format writers; wind on
spray and foam (`RenderSettings::wind`, `kFoamWindDriftDuration`, A47, 2026-11-15). Used by
I/O (scene check, queue, writers, `capabilities`) and the Blender add-on (channels, formats, witness tests).

**Blender add-on** (Blender add-on design, 3.1 to 3.13; Python; tested with Blender 5.3, A16;
`blender_version_min = "5.3.0"`, no maximum, a NOTE on newer versions, A48, 2026-11-15). Provides no C++ names.
It writes the scene file I/O reads (including the `wind` keys it derives from Blender's Wind force fields whenever
one exists, A20, A47, 2026-11-15; the cell size regions and map, A63; slip length and guides, A53, A59), runs the CLI and reads its events and `report`, links products through Blender's native Alembic, USD and
OpenVDB readers (`fluidity.link_outputs`, `LinkPlan`, `LINK_ROUTES`: Blender's word, A22), and runs the witness
tests in Blender 5.3 whose `WitnessRecord` lines decide render output's format table. Its own model:
`ROLE_TABLE`, `SUBSTANCE_PRESETS`, `REFERENCE_SCENES` (wave, syrup, fire, a snow example), `check_scene_quickly`,
`FORCE_FIELD_TYPE_TABLE`, the run state machine (with `needs_new_run`); its pure package is `logic/`.

**Sand** (sand design, 3; last module). Appends `MaterialKind::kSand = 4`, `LimitCase::kShapeReduced` and
`kPulledApart` (count 4), `sand_material.h` with `ComputeSandResponse`, the hit-split fields of the material
response, and plans the additions it needs in viscosity, step, I/O, render output and the Blender add-on
(`sand` in the `ROLE_TABLE` `substance_kinds` column and in the quick check). No other design declares, tests
or depends on sand (A9).

## (d) Material sets

A material set is the group of materials that share one velocity on the grid (one `GridFromParticles`, one
set of snapshots). Materials that do not mix are in separate sets, and `solids::CollideSolids` applies the
contact rule at faces they share (A8).

- **All liquids are in one material set.** Every liquid shares one grid velocity; two liquids that do not mix
  are kept apart by pressure's material boundary, not by separate sets. So every span "one per liquid material
  set" (`FoamAndSprayInputs::after_pressure`, viscosity's `cells_from_particles`, and the like) has exactly one
  entry, or none in a scene without liquid. The span types stay as declared.
- The other sets: snow, springy solids, gas markers. Gas markers are never passed to `CollideSolids` or to
  `ApplyContactReactions`; a set no pressure solve writes (snow, springy solids) passes null `was_solved`.
- `ReadyScene::material_sets` holds the masks; the step visits sets in ascending lowest material index. Any
  other layout is `CapabilityReason::kMaterialSetsNotValid` with `SuggestedChange::kPutAllLiquidsInOneSet`.
- A mask has one 0/1 entry per material with at least one 1 (transfer's stages, `CheckMaterialSets`).
  `pressure::ComputeDistanceToSurface` is the one function that accepts an all-zero mask: no liquid, every open
  cell is air.
- Memory per material set and cell block: 59,856 + 512 x ingredients bytes (48,336 + 512 x ingredients for a
  set with no solve), from `sizeof` of the real types.

Decided 2026-11-14: "snow" and "springy solids" name kinds of set, not a count. Each snow or springy-solid
object, or a group of them the artist chooses to move as one, may be its own set, as the step design's 3.2 and
5.2 say. What is fixed is that every liquid is in one set and gas markers are in one set.

## (e) Step stages

`step::Stage`, declared in the step design 3.1. `kStageCount = 22`. There is no `MoveGas` stage.
`kStages` holds stages 1 to 20; a null `StageTable` entry means "not a stage here". In a batched run the pass
stages run per batch and each `run_whole_grid` part once on the whole grid (a one-piece run calls it right after
`run`); where each buffer lives is the step plan's Appendix A (g.6).

| # | Stage | Main calls |
|---|---|---|
| 0 | `kNone` | the request itself was rejected |
| 1 | `kAddNewParticles` | particles, sources |
| 2 | `kChooseActiveCells` | `grid::BuildGrid` on the run's processor (A17; a batched GPU run builds the list on the CPU, uploads it and keeps a CPU copy of the grid, g.6); new cells come from the fill from the top, `grid::FindNewCellNeighbors`, on the grid's processor (A41, A118); `gas::FindMarkerChanges` (CPU form in batched runs, g.6), `FindCellsToKeepActive`, `BuildTry`, `ComputeGasSeenByLiquid` |
| 3 | `kFindSolidSurfaces` | solids on the grid; `run_whole_grid`: `solids::FindGridEdgesOpenToSky` (g.5) |
| 4 | `kParticlesToGrid` | `transfer::ParticlesToGrid` per set (the callback `apply_grid_only_region` at the end is not in the first version, A64, 2026-11-15) |
| 5 | `kSaveGridBeforeForces` | transfer, per set |
| 6 | `kApplyForces` | gravity, `materials::ApplyInternalForce` |
| 7 | `kDriveWaves` | `solids::DriveWaves` |
| 8 | `kCollideSolids` | `solids::CollideSolids` (every set but gas) |
| 9 | `kFindLiquidSurface` | `pressure::ComputeDistanceToSurface`, `FindLiquidSurface`; `run_whole_grid`: `FindMaterialBoundaries`, `FindAirPockets`, `CheckViscosityCanRun` |
| 10 | `kBuildPressureEquations` | `solids::BuildBlockingMaterialFaces`, `gas::ApplyGasStepStart`, then `pressure::BuildPressureEquations` (A18); whole grid; the material faces and the sealed blob list whatever stage follows (A24); in a viscous scene also `solids::BuildViscousMaterialFaces` after `BuildBlockingMaterialFaces` (scene with a blocking set) and `solids::ToFreeSolidsForViscousSolve(motion_map)` (A26, A27; g.9) |
| 11 | `kSolveViscosityAndPressure` | `viscosity::SolveViscosityAndPressure(ViscosityInputs{..., material_faces, free_solids}, ...)` with `SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure, material_face_velocity_after_solve, free_solid_motion}` (scenes with a viscous ingredient: a liquid with viscosity above 0, chosen once for the run, A25), `pressure::ComputeWasSolved`, `solids::ApplySolvedMotionToSolids(motion_map, free_solid_motion, fields.pressure, solids_try, solids_buffers)` (in place of `ApplyPressureToSolids`), then `solids::ApplyViscousSolveToMaterials(viscous_faces, material_face_velocity_after_solve, blocking_sets, solids_buffers)` (scene with a blocking set; in place of `ApplyPressureToMaterials`); step-owned rows `"step: material face velocity after solve"`, `"step: free solid motion"` (A26, A27; g.9); free solids move in their `AllowedMotions` (A59), solids' slip lengths enter the solve (A53), and the pressure and viscous parts of `SolidsTry::liquid_forces()` are reduced for every solid (A60) (2026-11-15). Replaced (A26, A27): "`solids::ApplyPressureToSolids`, then `solids::ApplyPressureToMaterials`; mirrors stage 12 (A24)" |
| 12 | `kSolvePressure` | `pressure::SolvePressure` (with the blocking-material and sealed-gas terms), `ComputeWasSolved`, `solids::ApplyPressureToSolids`, then `solids::ApplyPressureToMaterials` (liquid, no viscous ingredient); the free-solid term is `(K J) (Jᵀ M J)⁻¹ (K J)ᵀ` (A59) and the pressure part of `SolidsTry::liquid_forces()` is reduced for every solid, viscous parts 0 (A60) (2026-11-15) |
| 13 | `kSolveGas` | `gas::SolveGas` with `sealed_gas_pressure` after stage 11 or 12 (scenes with gas; `ApplyGasStepStart` first when stage 10 does not run) |
| 14 | `kSaveGridAfterPressure` | transfer, per set |
| 15 | `kGridToParticles` | transfer, per set |
| 16 | `kMoveParticles` | `MixParticleVelocity`, `contact::MoveParticles` (the `run_whole_grid` callback `move_grid_only_liquid` is not in the first version, A64, 2026-11-15) |
| 17 | `kMoveSolids` | `solids::ApplyContactReactions`, `ComputeCutWork` (no `ApplyPressureToMaterials`, A18); in a batched run on the CPU with a separate `SolidsBuffers` (`FindContactReactionBuffers`, g.6) |
| 18 | `kMakeFoamAndSpray` | callback `make_foam_and_spray` (pass 2 keeps the CPU piece map, three-argument `ApplyGridPiece`, g.6) |
| 19 | `kCheckStep` | `run_whole_grid`: `CheckSolidExchange`, the gas changes, the next limits |
| 20 | `kCommitStep` | every module's commit; fills `CommittedStepLimits::largest_liquid_speed` (A41) and `StepReport::liquid_force_on_solids` from the committed try (A60); a hinged body's pose rotates about `hinge_point`, a rail body moves along `rail_direction` only (A59) (2026-11-15) |
| 21 | `kResizeCells` | callback `resize_cells`; `ResizeCells` step 2 runs `cell_resize::ComputeCellSizeChanges` on the grid's processor (A41, 2026-11-15) |

## (f) Build order

The programme design's table. Each module may include headers of earlier modules only; later modules are
reached through types an earlier module declares and the step fills, or through the step's callbacks.

| Order | Module | Design file |
|---|---|---|
| 0 | Foundation (with section a) | (programme design) |
| 1 | contact | `01-contact-design.md` |
| 2 | grid | `02-grid-design.md` |
| 2b | particles | `03-particles-design.md` |
| 3 | transfer | `04-transfer-design.md` |
| 4 | pressure | `05-pressure-design.md` |
| 5 | viscosity | `06-viscosity-design.md` |
| 6 | materials | `07-materials-design.md` |
| 7 | solids | `08-solids-design.md` |
| 7b | gas | `09-gas-design.md` |
| 8 | step | `10-step-design.md` |
| 9 | cell resize and batches | `11-cell-resize-and-batches-design.md` |
| 10 | I/O and CLI | `12-io-and-cli-design.md` |
| 11 | surface with foam and spray | `13-surface-foam-and-spray-design.md` |
| 12 | Blender add-on | `14-blender-addon-design.md` |
| 13 | sand | `15-sand-design.md` |

Order numbers and file numbers differ, so designs name each other ("the gas design"), not by number.

How earlier modules avoid needing later ones: contact takes plain pointers and numbers the step passes;
particles' `NeighbourhoodInputs` holds raw arrays (transfer's `SolidsAtFaces::outside_solid`; the cell resize
region is not in the first version, A64, 2026-11-15); transfer declares `SolidsAtFaces` (solids fills it) and takes
`was_solved` as plain arrays; pressure declares `SolidCoverage`, `FreeSolidPush` and the gas arrays (solids and gas
fill them; `is_grid_only_region`, filled by cell resize, is not in the first version, A64, 2026-11-15); grid owns
the cell size rule because both the step and cell resize, later modules, read it (A41, 2026-11-15); viscosity takes a span of ingredient viscosities and a `ViscositySceneSummary` the step fills. An
input left null until its later module exists is tested both null and hand-filled.

## (g) Shared declarations decided while writing the plans

Copied from `plan-decisions.md` for modules not built yet; the owning module's plan carries the details. Where a
design's section 3 differs, these win.

### (g.1) Grid built on the GPU (A17; grid plan Tasks 6 to 11 and 20 to 27)

A run is all CPU or all GPU. The step calls `grid::BuildGrid` on its run's processor each time the cell set
changes; in a GPU run `ChooseActiveCells` hands it a `Buffer<CellId>` on the GPU and the entry count as a CPU
number, and takes the new grid's counts from `grid.counts()`. With GPU buffers `BuildGrid` validates, sorts and
builds every table on the GPU and reads back only a private `GridBuildStatus`; tables, counts and failures equal
the CPU builder's byte for byte (the CPU builder is the oracle). First failure on both processors: the smallest
key `(step << 56) | (position << 8) | detail` (1 id checks, 2 overlaps, 3 siblings, 4 size-step rule, 5 and 6
room); `kTooManyCellBlocks` always has `cell` zero. `ApplyGridVersion` runs on the CPU for every grid; it requires values on
the grid's processor (a precondition checked by an assertion in debug builds, A30). `tables(RunOn::kCpu)` on a GPU grid returns version 0 and null
arrays. `grid::BuildGrid` accepts an empty cell list.

```cpp
namespace fluidity::grid {
struct GridInputs {
  GridFrame frame;
  std::span<const CellId> active_cells;                // A list in CPU memory, any order.
  const Buffer<CellId>* active_cell_buffer = nullptr;  // Or a list in a buffer, any order.
  std::uint32_t active_cell_buffer_count = 0;          // Entries of *active_cell_buffer used.
  GridVersion version;                                 // 1 or more.
};
// With active_cell_buffer set, active_cells must be empty. Requires: a non-empty list on the GridBuffers'
// processor (for a span list: the CPU); a precondition checked by an assertion in debug builds (A30).
std::expected<StaggeredGrid, GridFailure> BuildGrid(const GridInputs& inputs, GridBuffers& buffers);  // Unchanged.
// Checks before building, on either processor, allocating nothing; the buffers hold no grid before or after.
std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs, GridBuffers& buffers);
std::expected<void, GridFailure> CheckSizeStepRule(const GridInputs& inputs, GridBuffers& buffers);
// The one explicit copy of a GPU grid (456 B per block; tests, tools, a caller that accepts the transfer).
std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);
inline constexpr std::uint32_t kMaxCellBlocksLimit = (1u << 26) - 1;   // Largest GridSettings::max_cell_blocks.

// cell_block_range.h: border search on either processor.
class BorderSearchBuffers { public: std::uint32_t max_cell_blocks() const; RunOn run_on() const; };  // Move-only.
std::expected<BorderSearchBuffers, GridFailure> BuildBorderSearchBuffers(RunOn run_on,
                                                                         std::uint32_t max_cell_blocks,
                                                                         RunMemory& run_memory);  // (A58, 2026-11-15)
std::vector<MemoryEstimateEntry> BuildBorderSearchMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks);
// Blocks outside `range` in border_so_far or in the border of range + border_so_far at `width`: one more
// stacked layer, merged with the layers before; ascending; returns the total, writes at most
// border_blocks.size(). All buffers on grid.run_on(); border_blocks must not be border_so_far.
std::expected<std::uint32_t, GridFailure> FindBorderBlocks(const StaggeredGrid& grid, CellBlockRange range,
                                                           const Buffer<CellBlockIndex>& border_so_far,
                                                           std::uint32_t border_so_far_count, BorderWidth width,
                                                           BorderSearchBuffers& search_buffers,
                                                           Buffer<CellBlockIndex>& border_blocks);

// grid_piece.h: pieces on either processor.
std::expected<GridPiece, GridFailure> BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range,
                                                     const Buffer<CellBlockIndex>& border_blocks,
                                                     std::uint32_t border_block_count, GridBuffers& buffers);
// GridPiece member: the piece map on grid().run_on().
const Buffer<CellBlockIndex>& block_in_whole_grid_buffer() const;
}  // namespace fluidity::grid
```

The vector form of `FindBorderBlocks` requires a CPU grid (a precondition checked in debug builds, A30);
`BuildRangeWithBorder` requires a CPU grid and returns `std::expected` with `kBadInput` for a range outside the grid or
an empty range over a grid with blocks (A116);
`BuildRanges` works on either; the span form of `BuildGridPiece` needs a CPU whole grid and
`GridPiece::block_in_whole_grid()` is empty for a GPU piece. Memory (`BuildGridMemoryEstimate`): a CPU grid 456 B per
block on `kRunOn`, 516 B on `kCpu`; a GPU grid everything on `kRunOn`: 1,996 B per block plus
`64 x ComputeOrderBuffersSize(kGpu).bytes_per_item` per block, `fixed_bytes`, 8 B and `sizeof(GridBuildStatus)` per
storage. Cell resize and batches build pieces with the buffer form and find stacked borders one buffer-form
`FindBorderBlocks` call per layer (batches' private `FindStackedBorder`); `ResizeCells` in a GPU run reads a CPU
copy of the committed grid made with `ReadGrid` (for split and merge list edits; the sizing itself runs on the GPU
and reads back only the 1 B `size_change` per cell place, A41, 2026-11-15); `solids::FindGridEdgesOpenToSky` on a GPU grid marks boxes of
the biggest-level block lattice on the GPU and reads back 1 B per box.

### (g.2) Coupling inside the pressure solve (A18)

Sealed gas blobs touching liquid and blocking material sets (snow, springy solids) are solved inside the liquid
pressure solve, not one step late. Since A24 viscosity's joint solve carries both terms as `SolvePressure` does
(declarations at the end of this section); no path reads a sealed blob touching liquid one step late. Air-end precedence: sealed blob, then `holds_tracked_gas` (one step late), then enclosed pocket, then 0.

```cpp
namespace fluidity::pressure {   // pressure_equations.h, shared types.
struct BlockingMaterialFace {          // One face piece where liquid meets a blocking material set.
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
  std::uint32_t material_set = 0;      // Index into the step's list of blocking sets.
  double blocked_area = 0;             // m2, > 0: this set's corner areas (ComputeBlockedCornerAreas) summed.
  double face_mass = 0;                // kg, > 0: the set's GridFromParticles::mass on the piece.
  double velocity_before_pressure = 0; // m/s along +axis, after ApplyForces and CollideSolids.
  double coverage_volume_rate = 0;     // m3/s along +axis: this set's share of the snapshot's solid_volume_rate.
};
struct SealedGasBlob {                 // One sealed tracked-gas blob that touches liquid, not held at atmosphere.
  grid::CellPlaceIndex first_cell_index{grid::kNoIndex};
  double gas_volume = 0;               // m3, > 0, step start.
  double background_pressure = 0;      // Pa absolute, > 0, step start, equalised.
  double heat_ratio = 1.4;
  double pressure_above_atmosphere = 0;  // Pa: the blob's carried value.
  double heat_volume_rate = 0;         // m3/s = ((heat_ratio - 1) / heat_ratio) * sum V q / p0.
  double boundary_volume_rate = 0;     // m3/s of solid moving into the blob through pieces that are not liquid
};                                     //   connections (inflow positive).
// EquationInputs gains:
//   std::span<const BlockingMaterialFace> blocking_material_faces;   // By face_piece_index, then material_set.
//   std::span<const SealedGasBlob> sealed_gas_blobs;                 // By first_cell_index.
//   const grid::CellValues<std::uint32_t>* sealed_gas_blob_of_cell = nullptr;   // kNoBlob elsewhere.
//   const grid::CellValues<double>* gas_pressure_above_background = nullptr;    // Pa, gas solve of the last commit.
// PressureEquations gains blocking_material_face_count, sealed_gas_blob_count.
// SolvedArrays gains std::span<double> blocking_material_velocity_after_pressure (one per face, input order) and
//   std::span<double> sealed_gas_pressure (Pa above atmosphere at the step end, one per blob).
// FlowImbalanceReport gains double max_sealed_gas_volume_error_rate (1/s).
// BufferSizes gains max_blocking_material_faces, max_sealed_gas_blobs.
// cell_equation_math.h (shared):
//   ComputeMaterialFaceCoefficients(blocked_area, face_mass, step_duration) -> ConnectionCoefficients{dt A / m, dt A^2 / m}
//   ComputeGasSqueezeAllowance(gas_volume, background_pressure, heat_ratio, step_duration) = V / (heat_ratio p0 dt)
//   ComputeMaterialVelocityAfterPressure(u, velocity_per_pressure, p_low, p_high) = u + velocity_per_pressure * (p_low - p_high)
}  // namespace fluidity::pressure

namespace fluidity::solids {     // blocking_material_faces.h
struct BlockingSet { std::span<const std::uint8_t> material_is_counted; transfer::GridFromParticles* grid_from_particles; };
struct BlockingMaterialFacesInputs { grid; blocks /* whole grid */; geometry; coverage;
                                     std::span<const BlockingSet> blocking_sets; };
class BlockingMaterialFaces;     // Move-only view into SolidsBuffers; faces_without_mass().
std::expected<BlockingMaterialFaces, SolidsFailure> BuildBlockingMaterialFaces(const BlockingMaterialFacesInputs&,
                                                                               SolidsBuffers&);
std::span<const pressure::BlockingMaterialFace> ToBlockingMaterialFaces(const BlockingMaterialFaces&);
struct PressureOnMaterials { std::span<const Vec3> push; double push_magnitude; };
// Writes each solved face velocity into its set's GridFromParticles::velocity.
std::expected<PressureOnMaterials, SolidsFailure> ApplyPressureToMaterials(const BlockingMaterialFaces&,
    std::span<const double> velocity_after_pressure, std::span<const BlockingSet>, SolidsBuffers&);
}  // namespace fluidity::solids

namespace fluidity::gas {        // solve_gas.h
// A24 deleted SealedGasCoupling and GasStepStartInputs::sealed_gas_coupling.
struct GasStepStartInputs { grid; blocks; liquid_geometry; grid_edge_is_open_to_sky; domain; sources;
                            step_start_time; step_duration; };
struct SealedGasForLiquid { std::span<const pressure::SealedGasBlob> blobs;
                            const grid::CellValues<std::uint32_t>* sealed_blob_of_cell;
                            const grid::CellValues<double>* pressure_above_background; };
// Design 5.3 steps 1 to 4; lists the sealed blobs touching liquid. SolveGas then runs step 1 (pieces) and 5 to 12
// and fails kBadInput without it.
std::expected<SealedGasForLiquid, GasFailure> ApplyGasStepStart(const GasStepStartInputs&, const GasSettings&,
                                                                GasAmountsTry&, GasBuffers&,
                                                                pressure::PressureBuffers&);
// SolveGasInputs gains sealed_gas_blobs, sealed_gas_pressure (after stage 11 or 12); a coupled blob's end background
// pressure is atmosphere_pressure + sealed_gas_pressure[j]. A24 deleted SolveGasInputs::sealed_gas_coupling,
// ComputeSealedGasStepLimit and GasResult::step_limit's sealed term.
}  // namespace fluidity::gas
```

Removed: `solids::MaterialPushFields` and the particle form of `ApplyPressureToMaterials`; `ApplyPressureToSolids`
stores no push owed to particles. A piece whose set has no face mass, or that touches no `kLiquid` cell, is not an
entry. `gas::ComputeGasSeenByLiquid` is unchanged and fills `gas_pressure_above_atmosphere` and
`holds_tracked_gas` where no sealed blob claims a cell (open gas, also above a viscous free surface, slivers, cells
that touch gas, held blobs: still one step late). Both terms are inside the liquid solve whichever of stages 11 and
12 runs (A24).

**Viscous liquids with blocking materials and sealed gas (A24).** From `plan-decisions.md` "Viscous liquids with
blocking materials and sealed gas (A24), and what counts as viscous (A25)". **Replaced (A26): material faces are
velocity unknowns of the joint solve, nothing eliminated but sealed gas never was; see (g.9).** In the joint solve
each material face velocity is eliminated exactly (its weight `dt A^2 / m` joins the outer loop's diagonal, not
scaled) and each sealed gas blob is one more unknown `q_j` of the outer loop (row as A18 with the solved velocity, diagonal `S_j +
sum_c g_c * connection_scale_c`). No piece returns `kBadInput` because the equations hold a material face or a
sealed gas blob.

```cpp
namespace fluidity::pressure {   // coupled_solve_pieces.h, flow_imbalance.h (NEW, A24)
// One value per sealed gas blob of the equations, entries [0, sealed_gas_blob_count), on the equations' processor:
// Pa above atmosphere (pressure), m3/s (volume error), m3/(Pa s) (own weight).
std::expected<void, PressureFailure> ComputePressurePush(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, GivenTerms given_terms, grid::FaceValues<double>& push);
std::expected<void, PressureFailure> ComputeVolumeError(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity, GivenTerms given_terms,
    grid::CellValues<double>& volume_error, Buffer<double>& sealed_gas_volume_error);
std::expected<void, PressureFailure> ComputeOwnWeight(
    const PressureEquations& equations, const grid::FaceValues<double>* connection_scale,
    grid::CellValues<double>& own_weight, Buffer<double>& sealed_gas_own_weight);
std::expected<void, PressureFailure> ComputeStartPressure(const PressureEquations& equations,
    grid::CellValues<double>& pressure, Buffer<double>& sealed_gas_pressure);   // c_j per blob
// Each BlockingMaterialFace's velocity after `pressure` (ComputeMaterialVelocityAfterPressure, unsolved end 0),
// input order; written as SolvePressure writes SolvedArrays::blocking_material_velocity_after_pressure.
std::expected<void, PressureFailure> ComputeBlockingMaterialVelocities(const PressureEquations& equations,
    const grid::CellValues<double>& pressure, std::span<double> velocity_after_pressure);
// The stop rule with the blob pressures given (the joint solve).
std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity,
    const SolveSettings& settings, PressureBuffers& buffers);
// A26, A27: these forms stay (SolvePressure's arithmetic, material faces eliminated); viscosity calls the
// CoupledVelocities forms of (g.9) instead.
}  // namespace fluidity::pressure

namespace fluidity::viscosity {  // (NEW, A24; gains the A26/A27 fields of (g.9))
struct ViscosityBufferSettings {   // gains:
  std::uint32_t max_sealed_gas_blobs = 0;   // >= the equations' sealed_gas_blob_count, else kBadInput.
};
struct SolvedCouplingArrays {      // Sizes equal the equations' counts (kBadInput otherwise).
  std::span<double> blocking_material_velocity_after_pressure;   // m/s along +axis, input order.
  std::span<double> sealed_gas_pressure;                         // Pa above atmosphere at the step end.
};
std::expected<ViscosityResult, ViscosityFailure> SolveViscosityAndPressure(
    const ViscosityInputs& inputs, const ViscositySettings& settings,
    const std::atomic<bool>* cancel_requested, ViscosityBuffers& buffers,
    grid::FaceValues<double>& velocity_after_pressure, grid::CellValues<double>& cell_pressure,
    SolvedCouplingArrays coupling_arrays = {});
}  // namespace fluidity::viscosity
```

The forms without a blob buffer accept material faces and are `kBadInput` only for equations with
`sealed_gas_blob_count > 0`. Memory estimate entry `"viscosity: sealed gas blob vectors"`. The step sets
`max_sealed_gas_blobs` to gas's `max_gas_blobs` (0 without gas). Exclusions (A24): `ViscosityError::
kSnowNotSupported`, `ViscositySceneSummary::has_snow` and `CapabilityReason::kViscousLiquidWithSnow` are removed;
`kFreeSolidNotSupported` / `kViscousLiquidWithFreeSolid` stay (**replaced (A27): lifted, see (g.9)**); `kSpringySolidNotSupported`,
`has_springy_solid_touching_liquid` and `kViscousLiquidWithSpringySolid` are never added.

### (g.3) Batched solves (A15)

A batched run agrees with a one-piece run within the solver's tolerance, not byte for byte; each mode is
repeatable on its own, and a batched result does not depend on the cut. Both modes use the same arithmetic:
grid-wide sums are one partial per cell block, then `math::TreeSum` over the blocks in block order; per-group
totals (each blob's, sealed gas blob's, free solid's, air pocket's sums) are, per cell block, the group's cells in
ascending cell place, then per group `math::TreeSum` over its block partials in ascending block index; labels are
propagated per loaded piece and joined (`JoinLabelsAcrossBatches`), equal to the one-piece labels. No solver
signature changes: a function taking `PressureBuffers&`, `ViscosityBuffers&` or `GasBuffers&` runs batch by batch
when the buffers are in the batched layout. Rule: state and geometry stay whole on the GPU; solver buffers go to
CPU memory (`CpuMemory::kLockedForCopies`) and through the GPU one loaded piece at a time.

```cpp
namespace fluidity::step {       // run_plan.h
enum class RunMode : std::uint8_t { kOnePiece, kBatched, kBatchedWithSolvesInPieces };
// RunPlan gains: std::uint32_t max_solve_loaded_blocks = 0; std::uint64_t allowed_cpu_bytes = 0;
// BuildRunPlan chooses only kOnePiece or kBatched; the caller (I/O Run, tests) copies BuildBatchPlan's choice.
}  // namespace fluidity::step

namespace fluidity::batches {    // batch_plan.h
enum class SolvePlace : std::uint8_t { kWholeGridOnProcessor, kInPieces };   // Replaces kWholeGridOnCpu.
// BatchSettings gains double solve_piece_share = 0.25;
// BatchPlan gains std::uint32_t max_solve_loaded_blocks, solve_batch_count; std::uint64_t bytes_on_processor,
//   bytes_on_cpu; std::uint32_t pass_loads_per_try, solve_loads_per_iteration; std::uint64_t solve_bytes_per_iteration.
// BatchReport gains solve_batch_loads, solve_bytes_to_processor, solve_bytes_from_processor,
//   solve_copy_elapsed_seconds (sums of the solvers' SolveCopyReport of the try).
}  // namespace fluidity::batches

namespace fluidity::pressure {   // solve_batches.h (CPU only), shared by pressure, viscosity and gas.
class SolveBatches {             // Move-only. Held inside buffers built with max_loaded_blocks > 0.
 public:
  RunOn run_on() const;                          // Processor of the loaded piece (the buffers' run_on).
  std::uint32_t max_loaded_blocks() const;       // Range plus stacked border of one batch.
  std::uint32_t batch_count() const;             // For the bound grid; 0 before ApplySolveBatches.
  grid::CellBlockRange range(std::uint32_t batch) const;   // Whole-grid blocks this batch writes.
  std::uint32_t loaded_batch() const;            // grid::kNoIndex when none is loaded.
  const grid::GridPiece& loaded_piece() const;   // Precondition: loaded_batch() != kNoIndex.
};  // Owns the GridBuilderBuffers it lends to the loaded piece's GridBuffers, and an OrderBuffers (A31).
std::expected<SolveBatches, PressureFailure> BuildSolveBatches(RunOn run_on, std::uint32_t max_cell_blocks,
                                                               std::uint32_t max_loaded_blocks,
                                                               std::span<const grid::BorderWidth> stacked_widths,
                                                               RunMemory& run_memory);  // (A58, 2026-11-15)
std::vector<MemoryEstimateEntry> BuildSolveBatchesMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks,
                                                   std::uint32_t max_loaded_blocks);
std::expected<void, PressureFailure> ApplySolveBatches(const grid::StaggeredGrid& grid, SolveBatches& batches);
std::expected<void, PressureFailure> LoadSolveBatch(const std::atomic<bool>* cancel_requested, std::uint32_t batch,
                                                    SolveBatches& batches);
template <typename Value>
std::expected<void, PressureFailure> WriteBatchValues(const Buffer<Value>& whole, std::uint32_t values_per_block,
                                                      SolveBatches& batches, Buffer<Value>& loaded);
template <typename Value>
std::expected<void, PressureFailure> ReadBatchValues(const Buffer<Value>& loaded, std::uint32_t values_per_block,
                                                     SolveBatches& batches, Buffer<Value>& whole);
template <typename Value>
std::expected<std::uint32_t, PressureFailure> WriteBatchGroups(const Buffer<Value>& whole,
                                                               const Buffer<std::uint32_t>& whole_begin_of_block,
                                                               SolveBatches& batches, Buffer<Value>& loaded,
                                                               Buffer<std::uint32_t>& loaded_begin_of_block);
template <typename Value>
std::expected<void, PressureFailure> ReadBatchGroups(const Buffer<Value>& loaded,
                                                     const Buffer<std::uint32_t>& loaded_begin_of_block,
                                                     SolveBatches& batches, Buffer<Value>& whole,
                                                     const Buffer<std::uint32_t>& whole_begin_of_block);
struct LabelJoin { std::uint32_t first = 0; std::uint32_t second = 0; };
std::expected<void, PressureFailure> JoinLabelsAcrossBatches(std::span<const LabelJoin> joins,
                                                             Buffer<std::uint32_t>& final_label);
struct SolveCopyReport {                // Since the last ResetSolveCopyReport (the step resets it per try).
  std::uint32_t batch_loads = 0;
  std::uint32_t sweep_count = 0;        // One sweep = every batch loaded once, in order.
  std::uint64_t bytes_to_processor = 0, bytes_from_processor = 0, bytes_within_processor = 0;
  double copy_elapsed_seconds = 0;
};
const SolveCopyReport& ReadSolveCopyReport(const SolveBatches& batches);
void ResetSolveCopyReport(SolveBatches& batches);
// Pass 1 into whole geometry, both batched modes (called by batches::ReadBatch after pass 1).
std::expected<void, PressureFailure> WriteSnapshotRange(const PressureBuffers& piece_buffers,
                                                        const grid::GridPiece& piece, PressureBuffers& whole_buffers);
}  // namespace fluidity::pressure
```

Buffer settings gain `std::uint32_t max_loaded_blocks = 0;` (0: one piece): pressure `BufferSizes`, viscosity
`ViscosityBufferSettings`, gas `GasBufferSizes`. Accessors `FindSolveBatches(const PressureBuffers&)` and
`ReadSolveCopyReport(const PressureBuffers&)` (also for `ViscosityBuffers`, `GasBuffers`; null in one piece).
`ResetSolveCopyReport(PressureBuffers&)`, `(ViscosityBuffers&)`, `(GasBuffers&)` (reconcile, g.7).
Stacked widths: pressure and gas `{kTouchingCells}`, viscosity `{kTouchingCells, kTouchingCells}`. `BufferSizes`
gains `max_direct_solve_values` (doubles; one piece `32 * max_cells`, batched layout `4 * max_cells`) and
`SolveResult` gains `std::uint32_t blobs_without_direct_solve`. Viscosity orders samples by the cell's block, then
ascending `CellId`, kind, axis pair, quarter, and takes dots by block partials then `math::TreeSum` (not
`ComputeTreeDot` over the buffer), in both modes. Failures: a sweep keeps the smallest failure key; the cancel flag
is read before every batch load; outputs are written only in the last write-back sweep after acceptance.

### (g.4) Witness records and interchange libraries (render output; A16, A19)

```cpp
namespace fluidity::render_output {   // format_table.h
struct WitnessRecord {
  WitnessProperty property;
  WitnessResult blender_5_3 = WitnessResult::kNotRun;  // The result measured in Blender 5.3.
  // Manual page, or "<test>; Blender <version> <build hash>; files <digest>"; empty while kNotRun.
  std::string_view evidence;
};
// ChooseFormat: the first candidate whose every record is kPassed, else the last candidate.
// interchange/interchange_libraries.h
struct InterchangeLibraries {
  std::string_view open_usd, alembic, openvdb;
  std::string_view missing_resource_folder;   // Empty when OpenUSD found its plugin resources.
};
}  // namespace fluidity::render_output
```

A table line is `{WitnessProperty::<P>, WitnessResult::<r>, "<evidence>"},`, the shape the Blender add-on's
`write_witness_records.py` prints from one Blender 5.3 results file. `capabilities` prints each record as
`{"property", "blender_5_3", "evidence"}`, the keys of the add-on's `WitnessRecordFact`. A USD candidate that
passes every witness in Blender 5.3 is chosen; one that fails is removed with its writer and the add-on's link
route. Library versions: the newest release on Conan Center (`https://center2.conan.io`) when each is pinned.

### (g.5) Plan-level shared names (transfer, particles, solids, step, I/O)

From `plan-decisions.md` sections "Transfer", "Particles", "Particles (continued)", "Solids", "Step", "I/O and
CLI", "Cell resize and batches"; the transfer declarations are copied from the transfer plan's Global Constraints.

```cpp
namespace fluidity::transfer {   // transfer_buffers.h (CPU only). Reused by every set and stage in turn. Move-only.
class TransferBuffers {
 public:
  RunOn run_on() const;
  std::size_t max_particle_count() const;
  std::size_t material_count() const;
};
// Bytes depend on material_count (`transfer: material offsets`, 8 B per material; reconcile item 27).
// Every builder below takes RunMemory& run_memory last (A58, 2026-11-15; Requires: run_on == run_memory.run_on()).
std::expected<TransferBuffers, TransferFailure> BuildTransferBuffers(RunOn run_on, std::size_t max_particle_count,
                                                                     std::size_t material_count,
                                                                     RunMemory& run_memory);
std::expected<SolidsAtFaces, TransferFailure> BuildSolidsAtFaces(RunOn run_on, std::uint32_t max_cell_blocks,
                                                                 RunMemory& run_memory);
std::expected<GridFromParticles, TransferFailure> BuildGridFromParticles(RunOn run_on, std::uint32_t max_cell_blocks,
                                                                         RunMemory& run_memory);
enum class FilledFromSolved : std::uint8_t { kNotKept, kKept };  // kNotKept: a set no solve writes.
std::expected<GridSnapshot, TransferFailure> BuildGridSnapshot(RunOn run_on, std::uint32_t max_cell_blocks,
                                                              GridTime time, FilledFromSolved filled_from_solved,
                                                              RunMemory& run_memory);
std::expected<CellsFromParticles, TransferFailure> BuildCellsFromParticles(RunOn run_on,
    std::uint32_t max_cell_blocks, std::uint32_t ingredient_count, RunMemory& run_memory);
// TransferError gains kOutOfMemory as its last value, returned only by these builders (reconcile item 31).
SolidsAtFacesOnProcessor ToSolidsAtFacesOnProcessor(const SolidsAtFaces& solids);
GridSnapshotOnProcessor ToGridSnapshotOnProcessor(const GridSnapshot& snapshot);
// Each stage (ParticlesToGrid, SaveGridBeforeForces, SaveGridAfterPressure, GridToParticles,
// ComputeCellsFromParticles) has one form, with TransferBuffers& transfer_buffers last (A31: tests and tools
// build a TransferBuffers too); it allocates nothing.
}  // namespace fluidity::transfer

namespace fluidity::particles {
// ApplyParticleLimits(Particles&, std::span<const std::size_t> max_particle_counts, std::uint32_t max_cell_blocks)
// ComputeAmounts returns std::expected<ParticleAmounts, ParticlesFailure>; ComputeAmountsPerBlock returns
// std::expected<void, ParticlesFailure>. ParticleFields and ParticleFieldsToChange carry RunOn run_on.
}  // namespace fluidity::particles

namespace fluidity::solids {
std::expected<SolidsBuffers, SolidsFailure> BuildSolidsBuffers(const SolidsSettings& settings,
    std::size_t max_particle_count, std::uint32_t max_cell_blocks, RunMemory& run_memory);   // (A58, 2026-11-15)
std::expected<SolidCoverageValues, SolidsFailure> BuildSolidCoverageValues(RunOn run_on,
    std::uint32_t max_cell_blocks, RunMemory& run_memory);   // The step builds its coverage arrays with it.
// MotionMap: a move-only view into SolidsBuffers; a map from an earlier build on the same buffers gives kWrongTime.
}  // namespace fluidity::solids

namespace fluidity::step {
struct RandomWords { std::uint32_t word[4]; };   // Shared header; Philox 4x32-10. Foam and spray use these too.
// ComputeRandomWords(std::uint64_t child_number, std::uint64_t source_id, std::uint64_t key);
// ComputeSignedUnitValue(std::uint32_t word).
// step_error.h; a private ToStageOutcome(const <Module>Failure&) in stage_outcome_from_modules.h maps module errors.
struct StageOutcome { RejectReason reason = RejectReason::kNone; RejectContext context{}; };
// StageTable: std::array<StageFunction, kStageCount> run_whole_grid{}, non-null only for kFindSolidSurfaces
// (FindGridEdgesOpenToSky, right after pass 1 and before stage 9's whole part), kFindLiquidSurface
// (FindMaterialBoundaries, FindAirPockets, CheckViscosityCanRun), kMoveParticles (the grid-only move; not in the
// first version, A64, 2026-11-15, so null until the later group) and
// kCheckStep (CheckSolidExchange, the gas changes, the next limits). A one-piece run calls it right after run.
// kStages holds stages 1 to 20; a null StageTable entry means "not a stage here"; kResizeCells is recorded only
// by AdvanceTo. Loaded try: FindLoadedTry(Simulation&), FindLoadedGridBuffers(Simulation&),
// struct BlockCopy { from_block; to_block; block_count; },
// WriteStepArrayBlocks(const StepTry& from, std::size_t array_index, std::span<const BlockCopy> copies, StepTry& to).
}  // namespace fluidity::step

namespace fluidity::io {
// io/material_presets.h
enum class MaterialPreset : std::uint8_t { kCustom, kWater, kSyrup, kSnow, kSpringySolid, kAir };  // sand appends kSand.
// struct MaterialPresetValues; inline constexpr std::array<MaterialPresetValues, 6> kMaterialPresets;
// std::optional<MaterialPreset> FindMaterialPreset(std::string_view name).
// Seams (I/O includes no render_output/ header): FrameWriter (function pointers, with start(void* context)),
// FrameOutput and KeptFrameInputs carrying std::int64_t frame and double time, a file_pattern without extension,
// the empty scene table outputs.render: RenderSection, and cli::BuildFrameWriterWithoutProducts() (each declared
// output "not built", reason no_product_writer); render output's Task 47 fills them in (I/O design 3.6).
// RunParts { FrameWriter frame_writer; OutputQueueSettings output; RunCallbacks callbacks; }.
}  // namespace fluidity::io
```

### (g.6) Buffers in batched runs and the planned cost (2026-11-14)

From `plan-decisions.md` "Planned cost shown to the user, and buffers in batched runs (2026-11-14)"; the full
inventory of stages x buffers x processor is the step plan's Appendix A, the step's planned copies its Appendix B.
Principle (A17): only what does not fit (the particles; in `kBatchedWithSolvesInPieces` also the solver buffers)
stays in CPU memory, moved in planned copies the planner counts and the reports measure.

- **Placement rule for whole try arrays** (both batched modes, either processor): a whole array lives on the run's
  processor when a part at `kWholeGridOnProcessor` reads or writes it; else in CPU memory when a `kWholeStateOnCpu`
  stage reads or writes it, the commit keeps it, or a later pass reads it (a carrier between passes); an array read
  and written only inside one pass has no whole instance, only the loaded one. `ReadStepArrays` reports each
  array's place.
- **Loaded instances.** The loaded `StepTry` owns, on the run's processor, one instance of every module buffer a
  pass stage uses (`transfer::TransferBuffers`, `materials::MaterialBuffers`, `contact::ContactBuffers`,
  `solids::SolidsBuffers`, at `max_loaded_particles` per material and `max_loaded_blocks`) beside the piece
  `pressure::PressureBuffers`. No module signature changes: the builders take the smaller counts.
- **Stage 17 in CPU memory** in a batched run: `solids::ApplyContactReactions` and `ComputeCutWork` on a separate
  `SolidsBuffers` built with `run_on = kCpu` (rows `"solids: "`, all `where = kCpu`).
- **CPU copy of the grid** (batched GPU runs): stage 2 builds the cell list on the CPU, uploads it into
  `Buffer<grid::CellId>` (row `"step: active cell list"`) for `grid::BuildGrid` on the GPU, and builds a CPU copy
  with the CPU builder from the same list (rows `"step: grid copy on the CPU"`); after a committed resize the copy is
  the CPU resized grid cell resize built, handed over with `step::ApplyResizedCpuGridCopy` (reconcile, g.7; the
  earlier refresh with `grid::ReadGrid` is withdrawn). Grouping, marker upkeep and `CellResize` read it
  (`choose_cell_size` removed, A41, 2026-11-15).
- **Mirrors of committed values** (both batched modes, `where = kCpu`): `"step: committed gas volume on the CPU"`,
  `"step: committed gas velocity on the CPU"`, `"step: keep active on the CPU"`; `"step: committed grid-only marks on
  the CPU"` is not in the first version (A64, 2026-11-15).

```cpp
namespace fluidity::step {
enum class ArrayPlace : std::uint8_t { kLoadedOnly, kWholeOnProcessor, kWholeInCpuMemory };  // StepArray::whole_place.
solids::SolidsBuffers* FindContactReactionBuffers(Simulation&);         // Stage 17's CPU instance; null in one piece.
const grid::StaggeredGrid* FindCpuGridCopy(const Simulation&);  // Batched GPU: the copy; CPU run: the grid; one-piece
                                                                // GPU: null.
// Replaces the two-argument form of "Cell resize and batches"; keeps the CPU piece map for stage 18 (one entry per
// piece block, else kBadInput). Form from the batches plan Task 36.
std::expected<void, StepFailure> ApplyGridPiece(StepTry& loaded_try, const grid::GridPiece& piece,
                                                std::span<const grid::CellBlockIndex> block_in_whole_grid);
// run_plan.h, shared by RunPlan and StepReport.
struct CopyCounts { std::uint32_t copy_count = 0; std::uint64_t bytes_to_processor = 0; std::uint64_t
  bytes_from_processor = 0; };
// StepReport::copies (this try's step-own copies); CopyCounts ReadCopyCounts(const Simulation&) (cumulative since
// BuildSimulation). RunPlan gains CopyCounts planned_step_copies; and PlanFit fit.
enum class PlanFit : std::uint8_t { kFits, kDoesNotFit };
// BuildRunPlan returns a plan that does not fit (mode kBatched, every estimate filled) instead of failing;
// BuildSimulation (and I/O BuildRun) fail kDoesNotFit on it.
// BuildStepMemoryEstimate(const ReadyScene&, const SceneCounts&, const RunPlan* batched_plan = nullptr).
// MemoryEstimate gains std::uint64_t on_cpu = 0; (rows where = kCpu plus snapshots; on_processor counts only kRunOn
// rows); ModuleBytes gains std::uint64_t on_cpu = 0; MemoryReport gains std::uint64_t allocated_cpu_bytes = 0;.
// (A58, 2026-11-15) MemoryEstimate::on_processor is ComputeRunMemoryBytes of the kRunOn rows (rounded per piece);
// MemoryReport's allocated_bytes becomes handed_out_bytes and it gains the fields of (g.10.2).
}  // namespace fluidity::step

namespace fluidity::solids {
std::expected<void, SolidsFailure> WriteBlockPartialsRange(const SolidsBuffers& piece_buffers, const
    grid::GridPiece& piece, SolidsBuffers& whole_buffers);   // Range blocks only; called by batches::ReadBatch.
}  // namespace fluidity::solids

namespace fluidity::gas {
struct MarkerGridInputs { const grid::StaggeredGrid* grid; const grid::CellValues<double>* gas_volume; const
  grid::FaceValues<double>* committed_velocity; };
// CPU form. Requires: every input in CPU memory (debug assertion, A30); same lists as the amounts form.
std::expected<MarkerChanges, GasFailure> FindMarkerChanges(const MarkerGridInputs& inputs, grid::CellBlockRange blocks,
    const particles::ParticleFields& gas_markers, const GasLaw& law, const MarkerSettings& settings, GasBuffers& buffers);
// MarkerChanges gains std::uint32_t copy_count = 0; std::uint64_t bytes_copied = 0; (rows where = kCpu in every GPU
// layout).
}  // namespace fluidity::gas

// particles::AddParticlesResult gains std::uint32_t copy_count = 0; std::uint64_t bytes_copied = 0;
// cell_resize::ResizeCellsReport gains std::uint32_t copy_count = 0; beside bytes_copied.
// render_output: CopyCounts ReadRecorderCopyCounts(const StepRecorder&) (cumulative), summed by I/O through its seam.
```

The budget and the measured cost are I/O's (CPU only), declared in `io/run_budget.h` (`ModeReason`,
`BudgetVerdict`, `MemoryBudget`, `ModuleBudget`, `CopyBudget`, `FitSuggestion`, `RunBudget`, `BuildRunBudget`,
`BuildBudgetLines`, `BuildSceneBudget`; `RunOverrides::assume_cpu_memory_bytes`, CLI `check --assume-cpu-memory`)
and `io/run_cost.h` (`StageSeconds`, `RunCost`, `ApplyTryToRunCost`, `ApplyStepToRunCost`, `ApplyFrameToRunCost`,
`ActivitySeconds`, `BuildCostLines`, `kMemoryTakenToleranceBytes = 2,097,152` (A58, 2026-11-15; replaces
`kMemoryWarningShare = 1.10` and `kCopyWarningShare = 1.25`, A43), `FrameMark`, `TimeProjection`,
`ComputeTimeProjection`), exactly as `plan-decisions.md` gives them; the warnings and fields of (g.10.2); schema
`schemas/simulate/v2/run_budget.fbs`. The Blender add-on reads `RunPlanFacts.budget: RunBudgetFacts | None` and
draws the warnings in the cost lines of `FLUIDITY_PT_performance` (no new widget).

### (g.7) Reconcile pass names (2026-11-14)

From `plan-decisions.md` "Reconcile pass (2026-11-14)", items 3, 11, 12 and coordinator items (3), (6), (7).

```cpp
namespace fluidity::pressure { void ResetSolveCopyReport(PressureBuffers& buffers); }     // item 3
namespace fluidity::viscosity { void ResetSolveCopyReport(ViscosityBuffers& buffers); }
namespace fluidity::gas { void ResetSolveCopyReport(GasBuffers& buffers); }
// Each zeroes the held SolveBatches' report; does nothing for one-piece buffers; allocates and copies nothing.

namespace fluidity::grid {       // staggered_grid.h (item 11). GridSettings::max_cell_blocks defaults to 0.
class GridBuilderBuffers { public: std::uint32_t max_cell_blocks() const; RunOn run_on() const; };  // Move-only.
std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffers(RunOn run_on, std::uint32_t max_cell_blocks,
                                                                       RunMemory& run_memory);  // (A58, 2026-11-15)
// The only form (A31). Requires: builder_buffers on settings' processor (debug assertion, A30); run_memory on the
// same processor (A58, 2026-11-15).
std::expected<GridBuffers, GridFailure> BuildGridBuffers(const GridSettings& settings,
                                                         GridBuilderBuffers& builder_buffers,
                                                         RunMemory& run_memory);
enum class BuilderBuffersSharing : std::uint8_t { kOnePerGrid, kOneForAllGrids };
std::vector<MemoryEstimateEntry> BuildGridMemoryEstimate(RunOn run_on, std::uint32_t storage_count, std::uint32_t max_cell_blocks,
    BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid);
}  // namespace fluidity::grid

namespace fluidity::step {       // resize_access.h (item 12, coordinator item (6))
struct ResizedCells { std::span<const grid::CellId> cells; const Buffer<grid::CellId>* cell_buffer = nullptr;
                      std::uint32_t cell_buffer_count = 0; const grid::StaggeredGrid* cpu_resized_grid = nullptr; };
std::expected<grid::StaggeredGrid, StepFailure> BuildResizedGrid(ResizeTry& resize_try,
                                                                 const ResizedCells& resized_cells);
std::expected<void, StepFailure> ApplyResizedCpuGridCopy(ResizeTry& resize_try,
                                                         grid::StaggeredGrid&& cpu_resized_grid);
}  // namespace fluidity::step

namespace fluidity::cell_resize {   // coordinator item (6)
// Kept by A41 (2026-11-15): the list edit, BuildResizeMap, grouping and marker upkeep still read CPU tables.
enum class GridCopies : std::uint8_t { kNone, kHeldByCellResize, kHeldByStep };   // CellResizeSettings::grid_copies.
std::vector<MemoryEstimateEntry> BuildCellResizeMemoryEstimate(std::uint32_t ingredient_count,
    RunOn grid_run_on = RunOn::kCpu, GridCopies grid_copies = GridCopies::kNone);
}  // namespace fluidity::cell_resize
```

`GridBuilderBuffers` (the grid's working buffers for building, A31) is lent to every `GridBuffers`, which must not
outlive it; with `kOneForAllGrids` the builder buffers entries have `copies = 1`. The step owns one `GridBuilderBuffers` for the committed, free
and loaded grid buffers. `ApplyResizedGrid(ResizeTry&, const grid::StaggeredGrid&)` stays for a caller that built
the grid itself. In a batched GPU run (`kHeldByStep`) cell resize's two CPU grids are neither allocated nor
counted. I/O: `ToName(step::Module)` (14 lower-snake names) and `ToName(step::RunMode)` with
`"batched_with_solves_in_pieces"` (coordinator item (3)); the add-on's `RunStatus.run_mode` and the `started`
event's `run_mode` take that third value (coordinator item (7)).

### (g.8) What counts as viscous (A25)

From `plan-decisions.md` "A25: what counts as viscous, and the viscous effect per step".

```cpp
namespace fluidity::viscosity {  // deformation_rate.h (shared)
// kSmallestVisibleViscousEffectPerStep (2.5e-4) is removed (A57, 2026-11-15); see viscosity_advice.h below.
// viscosity * step_duration / (density * cell_width * cell_width): no unit (the diffusion number).
FLUIDITY_CPU_AND_GPU double ComputeViscousEffectPerStep(double viscosity, double step_duration, double density,
                                                        double cell_width);
}  // namespace fluidity::viscosity

namespace fluidity::viscosity {  // viscosity_advice.h (CPU only; NEW, A57, 2026-11-15)
inline constexpr double kNegligibleViscosityRatio = 0.1;            // Judgments, literature-practice.md topic 1.
inline constexpr double kSmallestVisibleWavelengthInCells = 4;
inline constexpr double kNegligibleViscousChange = 0.01;
// 4 N / (1 - own_motion_kept); empty when own_motion_kept == 1 (no smoothing figure for pure FLIP).
std::optional<double> ComputeViscosityToSmoothingRatio(double viscous_effect_per_step, double own_motion_kept);
// -expm1(-π² D / 4), D = ν T / h², ν = viscosity / density.
double ComputeFourCellFeatureChange(double viscosity, double density, double run_duration, double cell_width);
// U L / ν with U = sqrt(2 g L).
double ComputeReynoldsNumber(double viscosity, double density, double smallest_domain_extent, double gravity_size);
}  // namespace fluidity::viscosity

namespace fluidity::step {       // run_plan.h
struct LiquidViscousEffect {     // At the smallest cell h, max_step_duration, own_motion_kept, T = end_time.
  std::uint32_t ingredient_index = 0;        // scene ingredient
  double viscosity = 0;                      // Pa s, the setting
  double density = 0;                        // kg/m3, rest density of its material
  double viscous_effect_per_step = 0;        // no unit; 0 when viscosity is 0
  std::optional<double> viscosity_to_smoothing_ratio;   // NEW (A57, 2026-11-15)
  double four_cell_feature_change = 0;       // NEW (A57): no unit, over the run
  double spreading_in_cells = 0;             // NEW (A57): sqrt(ν T) / h
  double reynolds_number = 0;                // NEW (A57)
  bool advises_zero_viscosity = false;       // NEW (A57): ratio < 0.1 or change < 0.01
};
enum class ViscousPath : std::uint8_t { kPressureSolve, kViscousSolve };   // stage 12 or 11, fixed for the run
// RunPlan gains: std::vector<LiquidViscousEffect> liquid_viscous_effects; double viscous_effect_cell_width = 0;
// (m) double viscous_effect_step_duration = 0; (s) and ViscousPath viscous_path;
// NEW (A57, 2026-11-15): double viscous_effect_run_duration = 0; (s) double reynolds_length = 0; (m, the domain
// box's smallest extent).
}  // namespace fluidity::step
```

A liquid ingredient is viscous exactly when its viscosity is above 0 (no threshold); the step fills
`ViscositySceneSummary::has_viscous_ingredient` and chooses the stage table once, in `CheckSceneCanRun` and
`BuildSimulation`; retries, shortened steps, limit raises, cell resizes and batches never change it.
`step::BuildRunPlan` fills the effects from `StepSettings::max_step_duration` and the grid frame's smallest cell
width. The water preset has viscosity 0. I/O's `RunBudget` gains `viscous_path`, `viscous_effect_cell_width_m`,
`viscous_effect_step_seconds`, `liquids` (`io::ViscousEffectOfLiquid`, renamed in the A27 pass from
`LiquidViscousEffect`, with `ingredient_name`, `viscosity_pa_s`, `density_kg_per_m3`, `viscous_effect_per_step`,
`is_viscous`, and since A57 (2026-11-15) `viscosity_to_smoothing_ratio` (absent when none), `four_cell_feature_change`,
`spreading_in_cells`, `reynolds_number`, `advises_zero_viscosity` in place of `effect_not_visible`), and
`viscous_effect_run_seconds`, `reynolds_length_m` (A57), the same in `run_budget.fbs` (table `ViscousEffectOfLiquid`),
and `BuildBudgetLines` prints the `liquids:` lines (A57 form: per viscous liquid grid number, smoothing ratio,
four-cell change, spreading, Reynolds number; `advice: <name>'s viscosity will not be visible: set it to 0` when
advised; `note: at viscosity 0 <name> slips along solids instead of sticking`) before the last `fits:` line; the
add-on's `RunBudgetFacts` gains `liquids` and `viscous_path` and draws the lines verbatim. **Replaced (A57,
2026-11-15):** the threshold `kSmallestVisibleViscousEffectPerStep` and the A25 "note" line.

### (g.9) A viscous liquid sticks, and free solids in the viscous solve (A26, A27)

From `plan-decisions.md` "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)". In the viscous
solve (stage 11) the velocity unknowns are `x = (u, w, xi)`: liquid face velocities, the material face velocities of
each `ViscousMaterialFace` (band faces of a blocking set with face mass within `kFitReach` cell widths of a liquid
cell centre) and six motions per free solid touching liquid; the liquid sticks to both. Nothing is eliminated
except sealed gas, which never was (A24). An inviscid scene (stage 12, `SolvePressure`) is unchanged: it slides,
material faces are eliminated (A18), free solids enter as the free-solid term, `(K J) (Jᵀ M J)⁻¹ (K J)ᵀ` of rank
`m = column_count` with `ξ_0` in the right-hand side (A59, 2026-11-15; was "the rank-one term", plan-decisions says
rank 6 for an unguided solid).

```cpp
namespace fluidity::solids {     // viscous_material_faces.h (NEW, A26)
struct ViscousMaterialFace {           // One band face of one blocking set.
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
  std::uint32_t material_set = 0;        // Index into the step's blocking sets.
  double face_mass = 0;                  // kg, > 0.
  double velocity_before_pressure = 0;   // m/s along +axis, after ApplyForces and CollideSolids.
  std::uint32_t blocking_face_index = 0xFFFFFFFFu;  // Its pressure::BlockingMaterialFace entry, or none.
};
class ViscousMaterialFaces;             // Move-only view into SolidsBuffers, sorted by face_piece_index, then set.
std::expected<ViscousMaterialFaces, SolidsFailure> BuildViscousMaterialFaces(const BlockingMaterialFacesInputs&,
    const BlockingMaterialFaces& pressure_faces, const pressure::GeometrySnapshot& geometry, SolidsBuffers&);
std::span<const ViscousMaterialFace> ToViscousMaterialFaces(const ViscousMaterialFaces&);
// Writes each solved band velocity into its set's GridFromParticles::velocity; push report as ApplyPressureToMaterials.
std::expected<PressureOnMaterials, SolidsFailure> ApplyViscousSolveToMaterials(const ViscousMaterialFaces&,
    std::span<const double> velocity_after_solve, std::span<const BlockingSet>, SolidsBuffers&);
// motion_map.h (NEW, A27). The free solids' inputs to the viscous solve, one per free solid touching liquid.
struct FreeSolidForViscousSolve {
  std::uint32_t solid_index = 0;          // As FreeSolidPush::solid_index.
  double mass = 0;                        // kg
  Mat3 inertia_at_start;                  // kg m2, world axes, about origin_at_start
  Vec3 origin_at_start{};                 // m
  Vec3 velocity_before_pressure{};        // m/s
  Vec3 turn_rate_before_pressure{};       // rad/s
  AllowedMotions allowed_motions{};       // (A59, 2026-11-15) ComputeAllowedMotions(limits, origin_at_start) of the
                                          //   try; the solve's unknowns are q (m = column_count), ξ = J q + ξ_0.
};  // Replaced (A59): "FreeMotionLimits limits{}; forbidden motions are held at their value before pressure".
std::span<const FreeSolidForViscousSolve> ToFreeSolidsForViscousSolve(const MotionMap&);
struct FreeSolidMotion { Vec3 velocity{}; Vec3 turn_rate{}; };   // m/s, rad/s
// Like ApplyPressureToSolids, with the motion solved by the viscous solve (travel and turn limits checked again,
// kSolidMovesTooFar); push report = mass (after - before), pressure part kept separately.
std::expected<void, SolidsFailure> ApplySolvedMotionToSolids(const MotionMap&,
    std::span<const FreeSolidMotion> motion_after_solve, const grid::CellValues<double>& pressure,
    SolidsTry&, SolidsBuffers&);
}  // namespace fluidity::solids

namespace fluidity::viscosity {  // (A26, A27; the viscosity design's section 4 uses the solids types)
// ViscosityInputs gains
//   std::span<const solids::ViscousMaterialFace> material_faces;
//   std::span<const solids::FreeSolidForViscousSolve> free_solids;
// SolvedCouplingArrays gains
//   std::span<double> material_face_velocity_after_solve;      // one per ViscousMaterialFace
//   std::span<solids::FreeSolidMotion> free_solid_motion;      // one per free solid
// (blocking_material_velocity_after_pressure stays filled, for the BlockingMaterialFace subset.)
// ViscosityBufferSettings gains std::uint32_t max_material_faces = 0, max_free_solids = 0.
// Memory estimate entries: "viscosity: material face vectors" (given max_material_faces, 12 x 8 + 16 bytes),
// "viscosity: free solid vectors and blocks" (given max_free_solids, (12 x 6 + 2 x 36) x 8 bytes),
// "viscosity: solid sample coefficients" (52 B = 4 + 6 x 8 per solid sample of any motion kind, kRunOn; A60,
// 2026-11-15, was "viscosity: sample solid coefficients" given max_sample_count).
// "viscosity: free solid vectors and blocks" holds the m x m block Jᵀ (M + Δt Rᵀ E_sᵀ W E_s R) J (A59).
// ViscosityResult gains the per-solid viscous impulse (A60); ViscosityError's kOutOfMemory is its last value
// (reconcile item 31, 2026-11-15).
}  // namespace fluidity::viscosity

namespace fluidity::pressure {   // coupled_solve_pieces.h, flow_imbalance.h (A26, A27), for the joint solve
struct CoupledVelocities {                              // Given velocity unknowns other than the liquid's.
  const Buffer<double>* blocking_material_velocity = nullptr;   // one per BlockingMaterialFace
  const Buffer<double>* free_solid_motion = nullptr;            // 6 per free solid: V then Omega
};
// Material face flow A_k w_k - coverage_volume_rate (kIncluded) or A_k w_k (kLeftOut); a free solid's flow
// sum_e push_e . (xi - xi_before) (kIncluded) or . xi (kLeftOut), sign as SolvePressure's free-solid term.
std::expected<void, PressureFailure> ComputeVolumeError(const PressureEquations&, const grid::CellValues<double>&
    pressure, const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity, CoupledVelocities,
    GivenTerms, grid::CellValues<double>& volume_error, Buffer<double>& sealed_gas_volume_error);
// dt A_k (p_low - p_high) per BlockingMaterialFace; dt sum_e (force, torque)_per_pressure p_e per free solid.
std::expected<void, PressureFailure> ComputeCoupledPush(const PressureEquations&, const grid::CellValues<double>&
    pressure, Buffer<double>& blocking_material_push, Buffer<double>& free_solid_push);
// connection_scale for liquid pieces; material_face_scale_k = face_mass_k / velocity_diagonal_k on g_k; free
// solids left out of the diagonal.
std::expected<void, PressureFailure> ComputeOwnWeight(const PressureEquations&, const grid::FaceValues<double>*
    connection_scale, const Buffer<double>* material_face_scale, grid::CellValues<double>& own_weight,
    Buffer<double>& sealed_gas_own_weight);
std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(const PressureEquations&,
    const grid::CellValues<double>& pressure, const Buffer<double>& sealed_gas_pressure,
    const grid::FaceValues<double>& velocity, CoupledVelocities, const SolveSettings&, PressureBuffers&);
}  // namespace fluidity::pressure
```

The A24 forms without `CoupledVelocities` (g.2) stay (they eliminate material faces from the pressure;
`SolvePressure`'s arithmetic) and are no longer called by viscosity. The step (stage 10, viscous scene) calls
`BuildViscousMaterialFaces` after `BuildBlockingMaterialFaces` and `ToFreeSolidsForViscousSolve(motion_map)`; stage
11 calls `ApplySolvedMotionToSolids` in place of `ApplyPressureToSolids` and `ApplyViscousSolveToMaterials` in place
of `ApplyPressureToMaterials` (section e). Step-owned rows `"step: material face velocity after solve"` (given
solids' `max_motion_map_entries` x 2, 8 B) and `"step: free solid motion"` (given the free solid count, 48 B);
viscosity buffers `max_material_faces` = 2 x `max_motion_map_entries`, `max_free_solids` = the free solid count.
**Exclusions (A27):** `ViscosityError::kFreeSolidNotSupported`, `ViscositySceneSummary::has_free_solid_touching_liquid`,
`CapabilityReason::kViscousLiquidWithFreeSolid` (stable names `free_solid_not_supported`,
`viscous_liquid_with_free_solid`) are lifted: `ViscosityError` 15 values with `kOutOfMemory` (reconcile item 25; the last value, item 31, 2026-11-15) before the viscosity
plan's three lifting tasks (12 if all are lifted), `kCapabilityReasonCount` 16 at most (13). `SuggestedChange::kSetViscosityToZero` is
removed from the main sequence (`kSuggestedChangeCount` 12); the sand module appends it back with
`kViscousLiquidWithSand` (stable name `set_viscosity_to_zero`, the add-on's repair). I/O's per-liquid budget struct
and schema table are `io::ViscousEffectOfLiquid` / `ViscousEffectOfLiquid` (were `LiquidViscousEffect`, the step's
struct's name).

### (g.10) Owner answers A40–A64 (2026-11-15)

From `plan-decisions.md` "Owner answers A40–A64 applied (2026-11-15)"; where an earlier part of this file differs,
this wins. Every row and block below is "(A<n>, 2026-11-15)" with the answer named.

#### (g.10.1) The shared cell size rule (A41, A61, A62, A63; A52 refined)

`native/fluidity/grid/cell_size_rule.h` (shared, `FLUIDITY_CPU_AND_GPU`, C++20 subset, no allocation), namespace
`fluidity::grid`, built in a new grid plan task (numerics). Cell resize design 3.2 becomes "moved to grid".

```cpp
namespace fluidity::grid {
struct CellSizeSettings {                     // Chosen once per scene. Replaces cell_resize::CellSizeSettings.
  SizeLevel smallest_size_level{};            // Width w_min: the surface, pockets, gas and solids.
  double surface_reach_in_cell_widths = 8;    // No unit. Depth term: w_band = w_min * max(1, depth / (8 w_min)).
  double solid_reach_in_cell_widths = 2;      // No unit. Solid term: w_solid = w_min * max(1, d / (2 w_min)).
  double shear_width_factor = 0.25;           // No unit. c_u of literature-cell-size.md 5(a); starting value (A62).
  double merge_width_share = 0.8;             // No unit. Merge only while 2 w <= 0.8 x wanted width.
  double merge_wait_duration = 0.25;          // s of committed time the merge test holds without a break.
  std::uint32_t min_merge_wait_steps = 8;     // Committed steps the merge test holds without a break.
  std::uint32_t max_cell_splits_per_resize = 1u << 16;   // Limit. More wanted splits wait.
};
struct CellWidthsByLevel {                    // m. width[l] = ComputeCellWidth(frame, l), computed once on the CPU.
  double width[kMaxSizeLevels];               // Compared, never recomputed with log2 or pow on either processor.
  std::uint8_t size_level_count = 1;
};
struct CellSizeFacts {                        // What the rule reads about one active cell.
  double cell_width = 0;                      // m.
  double depth_below_surface = 0;             // m. 0 at the surface; +infinity beyond reach.
  double distance_to_solid = 0;               // m. +infinity: none within reach.
  double velocity_gradient_size = 0;          // 1/s. Held maximum; 0 outside liquid.
  SizeLevel artist_size_level{};              // REPLACED (A78, A79) by double cell_size_factor = 1 (row above).
  bool holds_particles_that_keep_their_count = false;
  bool touches_air_pocket = false;
  bool wants_smallest_cells = false;          // Gas in or beside the cell (A10).
  bool touches_air_gas_or_solid = false;      // Merge veto (5.3 rule 5).
};
// m/s. U_ref = max(largest_liquid_speed, sqrt(gravity_size * biggest_cell_width)) (literature-cell-size.md 5(a)).
FLUIDITY_CPU_AND_GPU double ComputeReferenceSpeed(double largest_liquid_speed, double gravity_size,
                                                  double biggest_cell_width);
// m. wanted = clamp(min(w_band, w_solid, w_shear, w_artist), w_min, w_max); w_shear = c_u U_ref / g for g > 0, else
// +infinity; w_artist = widths.width[artist_size_level] when above 0, else +infinity. w_min where
// touches_air_pocket or wants_smallest_cells. No FMA contraction.
FLUIDITY_CPU_AND_GPU double ComputeWantedCellWidth(const CellSizeSettings& settings,
                                                   const CellWidthsByLevel& widths, double reference_speed,
                                                   const CellSizeFacts& facts);
// The biggest level (smallest index) whose width <= wanted_width, by comparison with widths.width.
FLUIDITY_CPU_AND_GPU SizeLevel ComputeWantedSizeLevel(const CellWidthsByLevel& widths, double wanted_width);
// A box that is not in the committed grid (stage ChooseActiveCells; A118): two facts read from the committed cells that
// touch it. The step fills each empty biggest cell from the top, splitting a box into its 8 children while
// touches_much_smaller_cell holds. At most 82 look-ups. Header grid/new_cell_rule.h; grid design 3.8.
struct NewCellNeighbors { bool touches_much_smaller_cell; bool touches_bigger_cell; };
FLUIDITY_CPU_AND_GPU NewCellNeighbors FindNewCellNeighbors(const GridTables& committed, CellId new_cell) noexcept;
// REMOVED 2026-11-15 (A78, A79): CellSizeRegion, ArtistCellSizes, ComputeArtistSizeLevel (history below).
struct CellSizeRegion {                       // An artist region: world axis-aligned box.
  Vec3 low_corner{}, high_corner{};           // m, world.
  SizeLevel size_level{};                     // Asked width rounded at load to the biggest level whose width <= it.
};
struct ArtistCellSizes {                      // Views on the run's processor.
  const CellSizeRegion* regions = nullptr;  std::uint32_t region_count = 0;
  const std::uint8_t* map_levels = nullptr;   // [map_height * map_width], row 0 at the domain's low y. 0: none.
  std::uint32_t map_width = 0, map_height = 0;
  Vec3 domain_low_corner{};  double domain_width_x = 0, domain_width_y = 0;   // m.
};
// Finest level any region containing `center` or the map pixel under it asks for; 0: none. Pixel column =
// min(floor((x - x0) / domain_width_x * map_width), map_width - 1), row likewise; outside the box: none.
FLUIDITY_CPU_AND_GPU SizeLevel ComputeArtistSizeLevel(const ArtistCellSizes& artist, Vec3 center);
}  // namespace fluidity::grid

namespace fluidity::cell_resize {
// velocity_gradient_math.h (shared): Frobenius norm of the full 3x3 gradient at the cell centre from the committed
// velocity_after_pressure; liquid cells only, 0 elsewhere; fixed order, no contraction.
FLUIDITY_CPU_AND_GPU double ComputeVelocityGradientSize(const grid::GridTables& tables, const double* velocity,
                                                        grid::CellPlaceIndex cell);
// ComputeCellSizeChanges (inside ResizeCells step 2, on the grid's processor, all blocks): one byte per cell place,
// grid::CellValues<std::uint8_t> size_change with CellSizeChange::kKeep, kSplit, kMayMerge, kMergeReady.
// CellResizeSettings::cell_size is a grid::CellSizeSettings.
}  // namespace fluidity::cell_resize
```

Hysteresis: split at once; merge eight siblings only after `2 w <= merge_width_share x wanted` held for them and
every touching cell for `merge_wait_duration` **and** `min_merge_wait_steps`; never touching air, a pocket, gas or a
solid, or within `8 w_min` of the surface; one level per cell per resize. Held maximum and wait counters are
`step::GridFields` arrays (section b, "Step grid fields"); remap: split copies, merge gives 0. Memory rows (cell
resize): `"cell resize: size changes"` 1 B per cell place, `"cell resize: cell size regions"` 56 B per region
(`kGivenCount`), `"cell resize: cell size map"` 1 B per pixel (`kGivenCount`), all `kRunOn`, `kWholeRun`. Same bytes on
CPU and GPU and for any batch cut (exact min and max, widths compared with `CellWidthsByLevel`, no `log2`, `pow`,
FMA). Artist inputs (REPLACED 2026-11-15 by the refinement volume, A78, A79; history): scene `cell_size_regions`, `cell_size_map` (top-view PNG, 8- or 16-bit grey, `v == 0` none,
else level `round(v x (size_level_count - 1))`); Blender `cell_size_region_collection`,
`fluidity_cell_size_region.cell_width`, `cell_size_map_image`, finding `cell_size_region_rotated` (NOTE). Removed:
`step::StepCallbacks::choose_cell_size`, `cell_resize::ChooseCellSize`, step 6.5 row O4 and Appendix B's O4 row,
`ResizeCellsReport`'s "ChooseCellSize's first copy". Later group "Coarser cells on calm surfaces (later)" (cell
resize plan, not in the done-criteria): A61's curvature term.

#### (g.10.2) One reserved block per run, the budget and the warnings (A58; A43, A45 refined)

**Pending Foundation addition** (Foundation plan "Additions after Foundation closed", Tasks 17–19; run before
module 1). Section (a) is re-copied from the built headers after Task 19; until then these declarations stand:

```cpp
namespace fluidity {
// base/run_memory.h (CPU only). One block on one processor; hands out pieces at multiples of 256 B. Move-only.
// A piece is never freed alone; a Buffer from a RunMemory frees nothing when destroyed; the RunMemory frees the
// block whole and must outlive its pieces (debug assertion on a live-piece count in a heap cell it owns).
class RunMemory {
 public:
  RunOn run_on() const;
  std::size_t reserved_bytes() const;
  std::size_t handed_out_bytes() const;
};
std::expected<RunMemory, BufferError> BuildRunMemory(RunOn run_on, std::size_t byte_count,
                                                     CpuMemory cpu_memory = CpuMemory::kOrdinary);
// base/buffer.h. A piece that does not fit returns kRunMemoryTooSmall, takes and writes nothing (a planner bug;
// debug builds assert). Requires: run_on == run_memory.run_on().
template <BufferValue Value>
std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size, RunMemory& run_memory);
// BufferError gains kRunMemoryTooSmall, appended last (ToName "run_memory_too_small").
// base/memory_count.h
inline constexpr std::size_t kRunMemoryPieceAlignmentBytes = 256;   // CUDA's cudaMalloc alignment; same on the CPU.
std::size_t ComputeRunMemoryPieceBytes(std::size_t byte_count);     // Rounded up to 256; 0 stays 0.
// Over entries with where == kRunOn: copies x ComputeRunMemoryPieceBytes(item count x bytes_each), saturating.
std::uint64_t ComputeRunMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts);
// base/ordered_sums.h, base/stable_order.h (Task 18); the plain forms stay for tests and tools.
std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count, RunMemory& run_memory);
std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count,
                                                           RunMemory& run_memory);
// Task 19: ReadMemoryAvailable(RunOn::kGpu) makes sure the GPU context exists (cudaFree(0)) before cudaMemGetInfo.
}  // namespace fluidity
```

**The builder rule.** Every module builder (`Build<Module>Buffers`, `BuildGridBuffers`, `BuildGridBuilderBuffers`,
`BuildBorderSearchBuffers`, `Build...Values`, `BuildSolveBatches`, `BuildCellResize`, `BuildRenderOutput`'s
run-processor part, and every other function that builds `kRunOn` buffers) takes `RunMemory& run_memory` as its
**last** parameter; every existing parameter stays; `Requires:` `run_on == run_memory.run_on()` (debug assertion,
A30); a `kCpu` buffer comes from `cpu_run_memory` (below). An entry's `copies` equals the number of buffers it stands
for, each `count x bytes_each` bytes; no `kRunOn` entry is `kOneCall` (a plan that has one makes it `kWholeRun`).
Each module's estimate test: build a `RunMemory` of exactly `ComputeRunMemoryBytes(Build<Module>MemoryEstimate(...),
counts)`, build into it, assert `handed_out_bytes() == reserved_bytes()` (`Build<Module>Buffers,
IntoRunMemory_HandsOutExactlyTheEstimate`). `kRunMemoryTooSmall` maps to the module's `kOutOfMemory` (rule 3).
Swaps of try and committed buffers, `grid::TakeBuffers` and rebuilding a grid into its buffers move `Buffer` objects
and free nothing.

```cpp
namespace fluidity::step {       // step_settings.h / run_plan.h
inline constexpr double kDefaultMemoryShareOfFree = 0.9;                    // Redshift's documented default.
inline constexpr std::uint64_t kDefaultMemoryLeftFreeBytes = 536'870'912;   // 512 MiB; starting value.
// StepSettings: double memory_share_of_free = kDefaultMemoryShareOfFree;
//               std::uint64_t memory_left_free_bytes = kDefaultMemoryLeftFreeBytes;   (memory_share_of_total removed)
// min(memory_share_of_free x free, free - memory_left_free_bytes), 0 when negative; both processors.
std::uint64_t ComputeAllowedMemoryBytes(MemoryAvailable available, double memory_share_of_free,
                                        std::uint64_t memory_left_free_bytes);
// MemoryReport gains (allocated_bytes becomes handed_out_bytes):
//   std::uint64_t planned_bytes = 0;            // ComputeRunMemoryBytes of the plan.
//   std::uint64_t reserved_bytes = 0;           // The current block.
//   std::uint64_t handed_out_bytes = 0;
//   std::uint32_t reserved_block_count = 0;     // Since BuildSimulation: 1 + limit raises.
//   std::uint64_t memory_taken_bytes = 0;       // Free before minus free after the reservation; GPU only, 0 on CPU.
//   std::uint64_t requested_beyond_plan_bytes = 0;   // Reserved after a raise beyond the start plan, plus measured
//                                                    //   kCpu bytes beyond their plan.
}  // namespace fluidity::step
// io/run_cost.h: inline constexpr std::uint64_t kMemoryTakenToleranceBytes = 2'097'152;   // 2 MiB; starting value.
```

`BuildSimulation` reads `ReadMemoryAvailable`, builds `run_memory` from `ComputeRunMemoryBytes(estimate)` before
anything else, reads free again, then builds every buffer into it. `ApplyLimits` builds a new `RunMemory` for the
whole new plan, rebuilds, copies the committed values, swaps and frees the old block whole; its check is `new
reserved + old reserved <= allowed_bytes`. The grid's 25 % headroom stays (A37). Warnings (I/O, no margins):
`warning: <p> memory: <b> requested beyond the planned <b>`; `warning: copies per step <b> (max) are above the
planned <b>` and the same per solve iteration; `warning: gpu memory taken <b> is less than the <b> reserved; ...`
when `memory_taken_bytes + kMemoryTakenToleranceBytes < reserved_bytes` (never when taken is above reserved); cost
line `gpu memory: planned <b>, reserved <b> in <n> block(s), handed out <b>, taken <b>`; on Windows the budget
lines' `note: on Windows a full GPU makes the run slow instead of failing; see "Prefer No Sysmem Fallback" in the
NVIDIA Control Panel`. JSON `RunCost` and `MemoryBudget` fields as in section (b).

**Decided (correction of 2026-11-15, plan-decisions item 3).** One `RunMemory` per processor a run uses: in a GPU
run `cpu_run_memory` (CPU; locked for copies in a batched GPU run) holds the `where = kCpu` entries kept `kWholeRun`
or `kOneStepTry` (stage 17's CPU `SolidsBuffers`, the step's CPU grid copy and mirrors, cell resize's CPU grids, gas
marker lists); a builder called with `run_on = kCpu` gets it as its `run_memory`; a builder building on both takes
`RunMemory& run_memory, RunMemory& cpu_run_memory` last; `ComputeRunMemoryBytes(estimate, counts, MemoryWhere)`;
`MemoryReport::allocated_cpu_bytes` becomes `handed_out_cpu_bytes`, with `reserved_cpu_bytes`. In a CPU run both are
one object. Plain `BuildBuffer` stays for tests, tools and `kOneCall` / `kOneSnapshot` CPU arrays.

#### (g.10.3) Viscosity advice (A57; replaces A44 and A25's threshold)

Declarations in (g.8). The smoothing formula `ν_num = (1 - α) h² / (4 Δt)` is this project's derivation from the
XPIC filter analysis (Hammerquist and Nairn 2017), not a published figure; the ratio uses the largest step, so it
is a lower bound.

#### (g.10.4) Solids: guides, liquid force, slip length (A59, A60, A53)

```cpp
namespace fluidity::solids {
// solid_settings.h (A59). FreeMotionLimits keeps its axis flags and gains (world frame, decided):
enum class GuideKind : std::uint8_t { kAxisLimits, kRail, kHinge };
//   GuideKind guide_kind = GuideKind::kAxisLimits;
//   Vec3 rail_direction{0, 0, 1};   // No unit, world, normalised at load. kRail: slides along it, no turning.
//   Vec3 hinge_point{};             // m, world, at time 0.
//   Vec3 hinge_axis{0, 0, 1};       // No unit, world.
// guide_math.h (shared, A59). Rows (V, Ω) in world axes about origin_at_start; built each try.
struct AllowedMotions {
  double columns[6][6];
  std::uint8_t column_count = 6;
  double prescribed_motion[6] = {};
};
// Axis limits: selection columns (0/1, exact); rail J = [d; 0]; hinge J = [a x (o - p); a], o = origin_at_start.
FLUIDITY_CPU_AND_GPU AllowedMotions ComputeAllowedMotions(const FreeMotionLimits& limits, Vec3 origin_at_start);
// liquid_force.h (A60). N, N m; mean over the step (impulse / step duration); world axes; torque about the body's
// origin at the step start (T_p = T_o + (o - p) x F). Every solid, ascending solid_id, fixed and domain box
// included; viscous parts 0 for an inviscid liquid. Guided bodies report all six components.
struct LiquidForceOnSolid {
  std::uint32_t solid_id = 0;
  Vec3 pressure_force{};
  Vec3 pressure_torque{};
  Vec3 viscous_force{};
  Vec3 viscous_torque{};
};
// SolidsTry::liquid_forces() returns them (span, ascending solid_id).
// SolidSettings gains double slip_length = 0;   // m (A53). 0 full stick (default); +infinity free slip; read for
//   fixed and driven solids and the domain box beside a viscous liquid; free solids always stick (A27), a value
//   there gives the check NOTE slip_ignored_on_free_solid. Tangential coefficient μ / (d + λ).
// SolidsError's kOutOfMemory is its last value (reconcile item 31).
}  // namespace fluidity::solids
// step::StepReport gains std::vector<solids::LiquidForceOnSolid> liquid_force_on_solids;   (from the committed try)
```

`LiquidForceOnSolid` is in `liquid_force.h` by this file's choice (plan-decisions names no header). The pressure part
is `Δt Σ_e (force_per_pressure, torque_per_pressure)_e p_e` over the solid's motion map entries; the viscous part
`-Δt Rᵀ E_sᵀ W (E x + g)` over its samples; one partial per cell block, then `math::TreeSum` in block order.

#### (g.10.5) Wind, Blender versions and the CUDA runtime, resume (A47, A50, A48, A49, A40)

Section (b) rows "Gas sources and wind", "Foam and spray recording", "Blender versions and the CUDA runtime",
"Resume in the recorded mode", "Scene file fields", "CLI". `render_output::RenderSettings` gains `gas::GasWind
wind;` and `inline constexpr double kFoamWindDriftDuration = 0.1;` (s; starting value). `IoError` gains
`kRecordedModeDoesNotFit` (last value); `Command` gains `bool choose_mode_again = false;`.

#### (g.10.6) Particles everywhere and the later groups (A64)

Everything grid-only is "not in the first version (A64)": cell resize design 1 Part A, 3.4 `RegionSettings`,
`kMaxFilledShare`, `ReadLargestGridOnlySpeed`, `ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep`, 3.5 (except
`ComputeDepthBelowSurface` and `ComputeLiquidTotals`, over particles only), 5.5, 5.6; transfer's
`kFromGridOnlyRegion` uses; the step's three grid-only callbacks, `CommittedStepLimits::grid_only_speed`, the
grid-only `GridFields` arrays and CPU mirror, stage 2's grid-only mark; pressure's "a grid-only cell counts as
full"; render output's grid-only surface inputs; the I/O and add-on depth fields; `kGridOnlyCellDrained`,
`kGridOnlyCellTooFull` (`kAmountsDoNotBalance` keeps its meaning over particles only). They move to "Deep liquid
without particles (later; needs review)" in the cell resize and batches plan (excluded from the done-criteria,
opened by "an independent review of this design comes first"), with same-named groups in the other plans pointing
at it. New test `ResizeCells, CalmDeepTank_CoarsensWithParticlesInEveryCell`.

#### (g.10.7) Reconcile items 31–33

31: `kOutOfMemory` last in every module error enum (rule 3; section b "Out of memory"). 32: a span (CPU) form given a
GPU grid, or a CPU list with GPU buffers, is a `Requires:` precondition with a debug assertion, not `kBadInput`
(`FindBorderBlocks`, `BuildGridPiece`, the grid checks). 33: accepted as written: test-only `BuildTestGrid`; span
forms of `ComputeGridCounts` / `CheckSizeStepRule` kept; `ComputeAmounts`, `ComputeWorkRates`, `CheckMaterialState`
keep one allocating form each; `kBadStepInputs`, `step_inputs`; `max_field_parts = 1u << 26` with one frame
`OrderBuffers`; sand Task 7's per-call GPU copy is settled in the sand module.

### (g.11) Owner answers A65–A68, and rows missing after the A40–A64 pass (2026-11-15)

From `plan-decisions.md` "Owner answers A65–A68 applied (2026-11-15)" and, for g.11.5 to g.11.7, from the A40–A64
section's items 3, 5, 10 and the reconcile rulings, whose declarations the designs already carry but this file
lacked. Where an earlier part of this file differs, this wins. Every row and block is "(A<n>, 2026-11-15)".

#### (g.11.1) Particle size follows the cell (A65, 2026-11-15)

The one wording, used by every design, plan, budget line and comment: **"every liquid cell holds about eight
particles of its own size: a particle's `size_level` is its cell's level, its rest volume is one eighth of its cell
(`(w/2)³` for a cell of width `w`), so a cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest
particle's volume."** No text says or implies one particle size everywhere, a lattice of `w_min / 2` over the whole
liquid, or `8 · 8^k` particles in a coarse cell. Memory estimates and the planner count **8 particles per liquid cell
at every level** (the start state's exact count times 1.25); the batch plan's 512 particles per full block holds at
every level. Budget line: `particles: <n> needed (about eight per liquid cell, sized to their cell); <m> of them in
cells deeper than <8 w_min> m below the surface; in this version every liquid cell holds particles (deep liquid
without particles comes later)`. Consequences in the module designs (plan decisions of this pass, A65):

- Pressure: `SurfaceDistanceInputs::particle_radius` (m) is **replaced** by `double particle_radius_in_cell_widths
  = 0.5;` (no unit): a particle counts as a ball of radius `particle_radius_in_cell_widths x
  grid::ComputeCellWidth(frame, size_level_p)`, so a coarse cell's eight particles cover it (a corner of a cell of
  width `w` is `sqrt(3) w / 4 < w / 2` from its nearest particle). The step passes
  `StepSettings::particle_radius_in_cell_widths` unchanged (the step design's stage 9 `FindLiquidSurface` call:
  replaced "`particle_radius`").
- Solids: a blocking material's one `particle_radius` stays right: blocking materials keep their count
  (`kKeepsItsCount`), so their particles are all at `smallest_size_level` (A66 seeding). I/O's `particle_radius =
  particle_spacing / 2` reads `particle_radius = w_min / 4`.
- Surface: a field point wider than the spacing is split into `k^3` parts, `k = ceil(width / spacing)` up to 8,
  beyond that `k = ceil(width / (2 spacing))` (replaced: "at most 8"), so parts are never more than two spacings
  apart (the default smoothing width) for particles of any level; parts stay at most one per lattice point (one per 8
  from 16 spacings wide).
- Gas: markers are 8 per gas cell at the cell's level with `rest_volume = (w/2)³` (unchanged); transfer's
  `min_weight_sum_at_face` (a count of particle shares) means the same at every level.

#### (g.11.2) A run starts on the adaptive grid (A66, 2026-11-15)

```cpp
namespace fluidity::step {   // step/start_cells.h (CPU only)
struct StartRegionShape {              // A start region of the scene: a box or a ball with a pose, world axes.
  enum class Kind : std::uint8_t { kBox, kBall } kind = Kind::kBox;
  contact::SolidPose pose{};           // m, world.
  Vec3 half_size{};                    // m. kBox.
  double radius = 0;                   // m. kBall.
  std::uint32_t material_index = 0;
  particles::CountRule count_rule = particles::CountRule::kMayBeSplitAndMerged;  // kKeepsItsCount: forced smallest.
};
struct StartCells {
  std::vector<grid::CellId> cells;                       // Active start cells, the grid's fixed order.
  std::array<std::uint64_t, grid::kMaxSizeLevels> cells_per_level{};
  std::uint32_t refine_passes = 0, closure_passes = 0;   // For the report and the termination test.
};
// Builds the start grid's cell list from the shapes alone, before any step. Facts per cell: depth and solid
// distance at the centre minus half the cell diagonal (sqrt(3) w / 2), velocity_gradient_size 0, the artist level at
// the cell's box; split while wider than grid::ComputeWantedCellWidth (one level per pass, coarse to fine), then split for
// the one-size-step rule; smallest_size_level wherever a kKeepsItsCount region lies. At most size_level_count - 1
// passes each. Above max_cell_count: kTooManyCellBlocks.
std::expected<StartCells, StepFailure> BuildStartCells(std::span<const StartRegionShape> regions,
    std::span<const contact::Solid> solids, const grid::GridFrame& frame, const grid::CellSizeSettings& cell_size,
    const grid::ArtistCellSizes& artist, std::uint64_t max_cell_count);  // A78: artist -> refinement, factor_table
// SceneCounts gains (exact, from the start state; limits are 1.25 x these plus sources):
//   std::array<std::uint64_t, grid::kMaxSizeLevels> start_cells_per_level{};
//   std::uint64_t start_particle_count = 0;   // From the seeded points.
//   std::uint64_t start_cell_blocks = 0;
// particles_at_depth_levels and start_particles_in_deep_cells are computed from the start cells.
}  // namespace fluidity::step
// io: BuildStartParticles gives each start cell of width w the 8 points at its cell-eighth centres (offsets +-w/4,
// the order of particles::ComputeRefillParticle), kept when inside its start region and outside every solid;
// rest_volume = (w/2)^3, mass = density x rest_volume, size_level = the cell's level, the region's start velocity,
// fitted_velocity_change = 0. Sources keep rest_volume = (w_min/2)^3. Replaces I/O 5.1's particle_spacing lattice.
```

Order in I/O `BuildRun`: read the scene, build `StartRegionShape`s (with `count_rule`), `step::BuildStartCells`,
`io::BuildStartParticles`, set `ReadyScene::start_cells = StartCells::cells` (no new parameter on the planner or the
build; empty only without cell resize; `StepError::kTooManyCellBlocks` sits just before `kOutOfMemory`),
`step::BuildRunPlan`, `step::BuildSimulation` (which builds the grid's first version from
`StartCells::cells`, not from particles at `new_cell_size_level`). A resume reads its saved grid and particles and
never calls `BuildStartCells`. `grid::ComputeWantedCellWidth` takes a negative depth or solid distance (a cell
centre's bound minus half the diagonal) as the surface or the solid: the `max(1, ...)` gives `w_min` (grid plan Task
13a test `ComputeWantedCellWidth, NegativeDepthOrDistance_IsSmallest`).

#### (g.11.3) Particles whose size no longer matches their cell (A67, 2026-11-15)

```cpp
namespace fluidity::particles {
// ParticleFields (and the committed and try copies, the snapshot, saved states) gain, per particle:
//   const std::uint8_t* merge_wait_passes;   // Resize passes waited, too small, for a chunk of eight; saturates
//                                            // at 255; 0 when it matches its cell or is created. Null for
//                                            // kKeepsItsCount materials. Field::kMergeWaitPasses.
inline constexpr std::uint8_t kMaxMergeWaitPasses = 16;   // Starting value, about 16 committed steps.
// AdjustParticleCountResult gains:
//   std::size_t mismatched_particle_count, waiting_particle_count, lone_particle_count, incomplete_split_count;
//   double kinetic_energy_change;   // J, energy(after) - energy(before), energy = (m/2)(|v|^2 + (w^2/4)|C|_F^2).
}  // namespace fluidity::particles
namespace fluidity::step {
// StepReport gains:
//   double size_change_kinetic_energy = 0;        // J, after minus before, the ResizeCells stage of this try.
//   std::uint64_t mismatched_particle_count = 0;
}  // namespace fluidity::step
```

`AdjustParticleCount` runs at **every** resize pass over all blocks; candidates are particles of
`kMayBeSplitAndMerged` materials (liquids, gas markers) whose `size_level` differs from their active cell's level.
Too large: split by the 1-to-8 pattern, repeated in a `smallest_size_level` cell until the children match (at most
`size_level_count - 1` times in one pass), one level per pass elsewhere; never refused for an incomplete
neighbourhood (counted). Too small: merged in chunks of eight of the same cell, material, `plate_side`, `flags` and
`size_level`, in cell place order then ascending id; chunks of two to seven merge after their first particle waited
`kMaxMergeWaitPasses`; a lone particle stays and is counted. Memory row `"particles: merge wait passes"` 1 B per
particle x 2, `kRunOn`. I/O JSON `frame_done` and `run_report.json` carry both new `StepReport` fields. Contact
counts a split child that lands inside a solid as `started_inside_count` at the next `MoveParticles`. Gas's marker
upkeep counts every marker of a cell whatever its `size_level`. Replaced: "a particle that drifts across an existing
size change keeps its `size_level` for good" (particles 5.4, cell resize 5.4).

#### (g.11.4) Order of the later work and the plan-area line (A68, 2026-11-15)

Later groups of the cell resize and batches plan, in this order, none in a done-criterion: (1) "Can the fine
surface band shrink from 8 cells to 4? (later, test only)" (analytic scene tests and the bathtub splash at
`surface_reach_in_cell_widths` 8, 6, 4; the default becomes the smallest reach whose every analytic error is within
its tolerance and within 1.2 x the reach-8 error, and the owner accepts the visual comparison); (2) "Coarser cells
on calm surfaces (later)" together with "Batches cut by columns or tiles (later)" (design sketch only: x-y tiles of
blocks spanning every height, borders of `kBorderLayers` block columns on four sides, solve pieces become tiles,
fixed-order sums unchanged; no task in the first version); (3) "Deep liquid without particles (later; needs
review)", last. The first version keeps slab batches and the pinned surface. When
`ModeReason::kLayerDoesNotFitProcessor` stops a scene, `BuildBudgetLines` prints: `batches: this version cuts batches
in horizontal slabs; one slab of <n> block layers at the smallest cell needs <b> and the <p> has room for <b>; at
this cell size batches hold a plan area up to about <a> m² (this scene: <a> m²); batches cut by columns, which remove
that limit, come later`, with `a_max = room x 16 w_min² / (layers x bytes_per_loaded_block)` (I/O test
`BuildBudgetLines, WideSceneLayerDoesNotFit_NamesPlanAreaLimit`: pool 10 x 5 m at 1 cm, 50 m² against about 3.2 m²).

#### (g.11.5) Solids and pressure: viscous push, guided inertia (A59, A60, 2026-11-15)

```cpp
namespace fluidity::solids {   // motion_map.h
// The viscous push on one solid, one per SolidsTry::solids() entry, as ViscosityResult::viscous_push_on_solids
// gives it: N s and N m s about the solid's origin at the step start, world axes ("viscous push", never "impulse").
struct ViscousPushOnSolid { Vec3 push{}; Vec3 turning_push{}; };
// Stage 11, right after ApplySolvedMotionToSolids: stores the pushes so liquid_forces() holds viscous_force =
// push / dt and viscous_torque = turning_push / dt. Another size: kBadInput; not finite: kNotFinite; on failure the
// try is unchanged. CPU only.
std::expected<void, SolidsFailure> ApplyViscousImpulses(std::span<const ViscousPushOnSolid> viscous_impulse,
                                                        SolidsTry& solids_try);
// One pressure::FreeSolidInertia per free solid, FreeSolidPush::solid_index order, filled by BuildMotionMap with
// allowed_motions = ComputeAllowedMotions(limits, origin_at_start). Replaces ToFreeSolidInverseInertia.
std::span<const pressure::FreeSolidInertia> ToFreeSolidInertia(const MotionMap& motion_map);
}  // namespace fluidity::solids
namespace fluidity::pressure {
struct FreeSolidInertia {                    // pressure_equations.h. M = diag(mass I3, inertia_at_start).
  double mass = 0;                           // kg, > 0.
  Mat3 inertia_at_start{};                   // kg m2, world axes, about origin_at_start.
  solids::AllowedMotions allowed_motions{};  // J = columns, xi_0 = prescribed_motion.
};
// EquationInputs::free_solids is std::span<const FreeSolidInertia> (replaces free_solid_inverse_inertia).
// free_solid_guide_math.h (shared). J^T M J factored by Cholesky in a fixed loop order; kBadInput for mass not > 0,
// a value not finite or a pivot not > 0.
struct GuidedInverseMass { double values[6][6]; };   // G = J (J^T M J)^-1 J^T, rows and columns (V, Omega).
FLUIDITY_CPU_AND_GPU PressureError ComputeGuidedInverseMass(const FreeSolidInertia& solid, GuidedInverseMass& result);
// q = (J^T M J)^-1 J^T (M xi_before + impulse), xi_after = J q + xi_0; solids' ApplyPressureToSolids calls it.
FLUIDITY_CPU_AND_GPU PressureError ComputeGuidedSolidMotion(const FreeSolidInertia& solid, const double xi_before[6],
                                                            const double impulse[6], double xi_after[6]);
}  // namespace fluidity::pressure
```

#### (g.11.6) Step: raises, run memory, resume (A58, A40, reconcile, 2026-11-15)

```cpp
namespace fluidity::step {
// advance_to.h. StepCallbacks gains: called by ApplyLimits after the new blocks are reserved and the step's buffers
// rebuilt, before the old blocks are freed; I/O rebuilds CellResize, the BatchedRun and the render StepRecorder into
// FindRunMemory / FindCpuRunMemory and copies what they keep. false: kDoesNotFit, the simulation unchanged, AdvanceTo
// stops with kLimitsCannotBeRaised. Null: nothing outside the step holds run buffers. BuildSimulation calls none.
//   Callback<bool(void* context, Simulation& simulation, const SimulationLimits& new_limits)> limits_raised;
// simulation.h. The run's blocks, for builders outside the step (only between steps); the same object in a CPU run.
RunMemory& FindRunMemory(Simulation& simulation);
RunMemory& FindCpuRunMemory(Simulation& simulation);
// run_plan.h. forced_mode set (a resume reuses its recorded mode, unless --choose-mode-again): mode = *forced_mode,
// fit = kFits exactly when that mode fits by step 5's test, else kDoesNotFit with the mode kept.
std::expected<RunPlan, StepFailure> BuildRunPlan(const ReadyScene& scene, const StepSettings& settings,
                                                 MemoryAvailable on_processor, MemoryAvailable on_cpu,
                                                 std::span<const MemoryEstimateEntry> output_estimate,
                                                 std::span<const MemoryEstimateEntry> cell_resize_estimate = {},
                                                 std::optional<RunMode> forced_mode = std::nullopt);
}  // namespace fluidity::step
```

#### (g.11.7) Grid estimate builders take the room (A58 reconcile ruling, 2026-11-15)

`grid::BuildGridMemoryEstimate(RunOn run_on, std::uint32_t storage_count, std::uint32_t max_cell_blocks,
BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid)` and
`grid::BuildBorderSearchMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks)`; callers pass
`limits.max_cell_blocks` (the step: `BuildGridMemoryEstimate(run_on, 2, limits.max_cell_blocks, kOneForAllGrids)`). A
buffer of a per-item plus fixed size is one `kGivenCount` entry at its exact bytes for that room. Section (a)'s
`base/memory_count.h` paragraph still quotes the two-parameter forms; it is re-copied after Foundation Task 19.

## Decided conflicts

- **C1. Who runs the joint velocity-and-pressure solve.** Viscosity, as `SolveViscosityAndPressure`; the
  `VelocityResponse` hook is removed from pressure. Owner answers A5, A6; viscosity and pressure designs,
  "Decisions".
- **C2. Split and merge at a cell size that no longer exists.** `SplitParticle`, `MergeParticles` and
  `AdjustParticleCount` take the grids before and after the resize, inside the resize transaction.
  `changes-to-apply.md` (particles); particles and cell resize and batches designs, "Decisions".
- **C3. Liquid volume per cell eighth and face-piece centres.** Pressure provides `CellEighthGeometry` and
  `FacePieceCentre` in viscous scenes. `changes-to-apply.md` (pressure, viscosity); pressure design, "Decisions".
- **C4. The boundary between two liquids that do not mix.** Pressure computes one distance field per touching
  liquid material and `FindMaterialBoundaries`; the pressure plan implements the sharp boundary, so
  `kMaterialBoundaryNotSupported` does not exist. `changes-to-apply.md` (pressure); pressure design,
  "Decisions"; `plan-decisions.md`, "Pressure (continued)".
- **Shared grid velocity across materials.** Materials that do not mix are separate sets; all liquids are in one
  set. Owner answer A8; `changes-to-apply.md` final table (step decision); section (d).
- **Gas with liquid.** Supported from the start. Owner answer A10; gas design.
- **Gas snapshot in the step versus render output.** The step holds `GasAmountsSnapshot`; render output calls
  `gas::ToGasSnapshot`. `changes-to-apply.md` final table; gas and surface, foam and spray designs.
- **Step callbacks.** `Callback<Function>{function, context}` pairs. `changes-to-apply.md` final table; step
  design 3.9.
- **Batched solves.** Pressure, viscosity and gas solves are also batched (A15): `kBatchedWithSolvesInPieces`
  streams solver buffers through the GPU one loaded piece at a time; a batched run agrees with a one-piece run
  within the solver's tolerance; `border_cell_pressure` removed. `plan-decisions.md`, "Batched solves (A15)",
  (g.3); `changes-to-apply.md` (pressure, cell resize and batches).
- **Coupling inside the pressure solve.** Sealed gas blobs touching liquid and blocking material sets are
  solved inside the liquid pressure solve (A18); `plan-decisions.md`, "Coupling inside the pressure solve (A18)";
  (g.2).
- **Grid built on the GPU.** A run is all CPU or all GPU; a GPU run builds its grid on the GPU (A17);
  `plan-decisions.md`, "Grid built on the GPU (A17)"; (g.1).
- **Blocking materials and sealed gas beside a viscous liquid.** Viscosity's joint solve carries both terms as the
  pressure solve does; snow and springy solids beside a viscous liquid run; free solids beside a viscous liquid stay
  excluded (A24, option C of reconcile item 4); `plan-decisions.md`, "Viscous liquids with blocking materials and
  sealed gas (A24)"; (g.2). **Replaced (A26, A27) for the joint solve's material faces and for free solids** by the
  next item.
- **A viscous liquid sticks; free solids in the viscous solve.** Band material faces and free-solid motions are
  velocity unknowns of the viscous solve, the liquid sticks to both; no viscosity exclusion by material remains;
  `kSetViscosityToZero` leaves the main sequence (the sand module adds it back). Owner answers A26, A27;
  `plan-decisions.md`, "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)"; (g.9).
- **What makes a liquid viscous.** Its viscosity setting above 0, fixed for the run (A25); the budget prints, per
  viscous liquid, the grid number, the viscosity-to-transfer-smoothing ratio, the four-cell feature change over the
  run, the spreading and the Reynolds number, and advises viscosity 0 when the ratio is below 0.1 or the change below
  0.01 (A57, 2026-11-15; replaces A25's threshold); (g.8).
- **Deep liquid (A64, 2026-11-15).** Every liquid cell holds particles in the first version; the grid-only region
  is a later group that needs review; (g.10.6).
- **Who owns the cell size rule (A41, A61–A63, 2026-11-15).** Grid (`grid/cell_size_rule.h`), read by stage 2 and
  cell resize; the `choose_cell_size` callback is removed; (g.10.1).
- **GPU memory (A58, 2026-11-15; A43, A45 refined).** One `RunMemory` block per run, a budget of
  `min(0.9 free, free - 512 MiB)` on both processors, warnings without margins; (g.10.2).
- **Guides, liquid force, slip (A59, A60, A53, 2026-11-15).** Guided bodies in general form with `AllowedMotions`;
  the liquid force on every solid from the solve's own terms; a Navier slip length on prescribed-motion solids;
  (g.10.4).
- **Wind (A47, A50, 2026-11-15).** On gas in the solver and on spray and foam in render output, never on the liquid
  body; scene field `wind`.
- **Resume (A40, 2026-11-15).** The recorded run mode is reused; a mode that no longer fits refuses with
  `kRecordedModeDoesNotFit` unless `--choose-mode-again`.
- **Particle size (A65, 2026-11-15).** Option B of `tradeoff-particles.md`: about eight particles per liquid cell,
  sized to their cell, at every level; budgets count 8 per liquid cell; (g.11.1).
- **Start state (A66, 2026-11-15).** The run starts on the adaptive grid from `step::BuildStartCells`, seeded by
  `io::BuildStartParticles` at each cell's own size; the planner counts that state exactly; (g.11.2).
- **Drifted particles (A67, 2026-11-15).** Matched to their cell at every resize pass (split, or merged by eights,
  or two to seven after 16 passes); a lone particle stays, counted; (g.11.3).
- **Later work (A68, 2026-11-15).** First version: slab batches and the pinned surface, with an honest plan-area
  line; the later groups in the order of (g.11.4).

## Replaced on 2026-11-14

- Section (a): the Foundation "plan" declarations (`Transpose(const Mat3&)`, `Determinant(const Mat3&)`,
  `IsFinite(const Mat3&)`, four-value `BufferError`, span parameters without `std::type_identity_t`, no
  `OrderBuffers`, no `TreeSumOf` or `LargerMagnitude`) by the built headers ("Foundation as built" in
  `plan-decisions.md`); the note that `processor_description.h` and the field names of `ProcessorDescription` were
  this file's proposal, by the built header.
- Rule 5 and conflict "Batched solves": "a run in one piece and a batched run give the same bytes" and "the solves
  are not batched", by "Batched solves (A15)".
- (b) "Building a grid version": "`BuildGrid` runs on the CPU, also for GPU runs" (grid decision 8, Q41), by
  "Grid built on the GPU (A17)".
- (b) "Gas seen by the liquid": sealed gas read one step late, and (e) stage 17 calling `ApplyPressureToMaterials`,
  by "Coupling inside the pressure solve (A18)".
- (b) "Boundary between two liquids that do not mix" and conflict C4: the exclusion `kMaterialBoundaryNotSupported`,
  by "Pressure (continued)".
- (b) "Capability names": `kLiquidsThatDoNotMixTouch` and `kCapabilityReasonCount = 19`, by "Pressure
  (continued)" and "Viscosity"; `step::Support` / `kSupported`, by the glossary table (`CapabilityLevel`, `kCanRun`).
- (b) "Gas state and output": the two-argument `ToGasSnapshot`, by "Gas" (three arguments).
- (d) memory per set 48,576 (40,416) + 512 x ingredients bytes, by "Transfer" (59,856 and 48,336, `sizeof`).
- Render output and add-on rows: `WitnessRecord::blender_5_2` and library versions matched to Blender's bundled
  ones, by "Amendments of 2026-11-14 to the Blender and render-output plans" (A16, A19); the add-on's
  `wind_acceleration`, by the same section (A20); `ChannelStorage` / `storage` and `WitnessOutcome`, by the
  glossary tables (`ChannelValueType` / `value_type`, `WitnessCaseResult`).
- Batches and step rows: the two-argument `step::ApplyGridPiece(StepTry&, const grid::GridPiece&)` ("Cell resize and
  batches"), CPU block ids of a GPU grid (solids' `FindGridEdgesOpenToSky` "from the block ids"; the batches
  read the piece map back, 4 B per loaded block) and a `BuildRunPlan` that fails when nothing fits, by "Planned cost shown to the
  user, and buffers in batched runs" and "Batched solves (A15)", "A17 follow-ups" (g.5, g.6).
- (b) "The solve without viscosity" ("the only solve of equations with blocking-material faces or sealed gas
  blobs"), "Pieces of the joint solve" (`kBadInput` for those equations), "Sealed gas inside the liquid solve"
  (`gas::SealedGasCoupling`), "The solve with viscosity" (six-argument form), "Viscosity exclusions" (`has_snow`,
  `kSnowNotSupported`), "Capability names" (`kCapabilityReasonCount = 18`, `kViscousLiquidWithSnow`), "Solids in a
  step" (stages 10 and 12 only); (c) gas's `ComputeSealedGasStepLimit`; (e) stage 11's "sealed gas
  `kReadOneStepLate`"; (g.2) "Equations with either term are solved only by `SolvePressure`; the pieces viscosity's
  joint solve calls return `kBadInput` for them (a viscous liquid keeps reading sealed gas one step late)",
  `SealedGasCoupling`, `GasStepStartInputs::sealed_gas_coupling`, `SolveGasInputs::sealed_gas_coupling`,
  `ComputeSealedGasStepLimit` and "Coupling is `kInsideLiquidSolve` when stage 12 runs, `kReadOneStepLate` when
  stage 11 runs": by "Viscous liquids with blocking materials and sealed gas (A24)".
- (b) "Material presets": the water preset's 1e-3 Pa s, by "A25: what counts as viscous".
- (b) "Memory estimate entries of the grid" (two-parameter form), `GridSettings::max_cell_blocks = 1u << 22`, and
  (g.6) the CPU grid copy "refreshed after a committed resize with `grid::ReadGrid`": by "Reconcile pass
  (2026-11-14)", items 11 and coordinator item (6) (g.7).
- (b) "Pieces of the joint solve" (viscosity calling the forms that eliminate material faces), "Blocking materials
  inside the liquid solve" (faces eliminated in the joint solve), "The solve with viscosity" (A24 fields only),
  "Viscosity exclusions" (`has_free_solid`, `kFreeSolidNotSupported` kept), "Capability names"
  (`kCapabilityReasonCount = 17`, `kViscousLiquidWithFreeSolid` stays, `kSuggestedChangeCount = 13`), "Solids in a
  step" (`ApplyPressureToMaterials` at stages 11 and 12 alike), "What counts as viscous" (`io::LiquidViscousEffect`);
  (e) stage 11 (`ApplyPressureToSolids`, `ApplyPressureToMaterials`, "mirrors stage 12"); (g.2) the A24 elimination
  of material faces in the joint solve and "`kFreeSolidNotSupported` / `kViscousLiquidWithFreeSolid` stay"; conflict
  "Blocking materials and sealed gas beside a viscous liquid" (free solids excluded): by "A viscous liquid sticks,
  and free solids in the viscous solve (A26, A27)" (g.9).
- Owner's review of the Foundation rulings (A28 to A31, 2026-11-14; `plan-decisions.md` "Owner's review of the
  Foundation rulings applied (A28–A33)"): every `...Scratch` name, by its `...Buffers` name (the grid's lent
  `GridScratch` by `GridBuilderBuffers`, `ScratchSharing { kOnePerBuffers, kOneForAllBuffers }` by
  `BuilderBuffersSharing { kOnePerGrid, kOneForAllGrids }`); the forms of a function that built their own working
  buffers (`BuildGridBuffers(const GridSettings&)`, the transfer stages without `TransferBuffers`, the
  `ComputeLiquidTotals` form without `SumBuffers`), by the one form that takes them; `BufferError::kRunOnMismatch`
  and every "on another processor is `kBadInput`" between arguments, by a `Requires:` precondition checked in debug
  builds; `grid::ReleaseBuffers`, by `grid::TakeBuffers`; `StepRequest`, by `StepInputs`.

## Replaced on 2026-11-15

By `plan-decisions.md` "Owner answers A40–A64 applied (2026-11-15)" (g.10):

- (b) "Grid-only region", "Liquid totals" (the `const GridOnlyAmounts&` form), "A complete neighbourhood"
  (`is_grid_only_region`), "Face sums and snapshots" (`kFromGridOnlyRegion`), "Step snapshot" (grid-only arrays),
  "Callbacks" (grid-only callbacks); (c) cell resize's two regions and transfer's grid-only write; (e) stages 4 and 16
  callbacks; (f) the cell resize region and `is_grid_only_region`; (g.5) `run_whole_grid` for `kMoveParticles`;
  (g.6) the grid-only mirror: marked "not in the first version (A64)".
- (b) "Callbacks" (`choose_cell_size`); (c) `ChooseCellSize` and cell resize 3.2; (g.6) `choose_cell_size` as a
  reader of the CPU grid copy: by the grid's shared cell size rule (A41, A61–A63).
- (g.6) `kMemoryWarningShare = 1.10`, `kCopyWarningShare = 1.25`; `StepSettings::memory_share_of_total` and the
  CPU rule min(0.7 total, 0.8 free); `MemoryReport::allocated_bytes`; builders without a `RunMemory&` in (b), (g.1),
  (g.3), (g.5), (g.7): by A58 (A43, A45 refined).
- (b) "What counts as viscous" and (g.8): `kSmallestVisibleViscousEffectPerStep = 2.5e-4`, `effect_not_visible`, the
  A25 note line: by A57.
- (b) "Free solids in the viscous solve" and (g.9): `FreeMotionLimits limits` held at their value before pressure,
  and "free solids enter as the rank-one term": by A59; (g.9) `"viscosity: sample solid coefficients"` (given
  `max_sample_count`): by A60's `"viscosity: solid sample coefficients"` per solid sample.
- (b) "Gas sources and wind": scene field `gas_wind`, wind keys written only with gas: by A47.
- (c) Blender add-on "Blender 5.3 only" as a limit: by A48 (tested with 5.3, no maximum).
- Rule 3: `kOutOfMemory` at any position: by reconcile item 31 (last value).

By `plan-decisions.md` "Owner answers A65–A68 applied (2026-11-15)" (g.11):

- (b) "Split, merge, count changes": "drifted particles keep their level and are counted", and particles 5.4 / cell
  resize 5.4 "keeps its `size_level` for good": by A67.
- (b) "The run plan": the particle estimate from start cells "at the levels the depth term gives, times the seeded
  count" and every limit sized at the smallest cell: by A66's exact `SceneCounts` fields. I/O 5.1's start lattice
  `particle_spacing = smallest_cell_width / 2` and the first grid from particles at `new_cell_size_level`: by
  `BuildStartCells` and `BuildStartParticles`.
- Any wording of one particle size everywhere, a `w_min / 2` lattice over the whole liquid or `8 · 8^k` particles in
  a coarse cell; pressure's `SurfaceDistanceInputs::particle_radius` (m); surface's "at most 8" parts per axis: by
  A65 (g.11.1).
- (b) "Solids in a step": `ToFreeSolidInverseInertia`, by `ToFreeSolidInertia` (A59; g.11.5).
