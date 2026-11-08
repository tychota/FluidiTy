#pragma once
#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_boundary_transfer_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
namespace fluidity {
struct CudaLiquidBoundaryGeometry3D {
  FrozenLiquidBoundaryGeometry3D geometry;
  // Sum of per-query CUDA event times, including the architecture marker.
  // Excludes host topology work, allocations, uploads and downloads.
  CudaOperatorMeasurement measurement;
};
struct CudaLiquidPressureProblem3D {
  LiquidPressureProblem3D problem;
  // Sum of per-query CUDA event times, including the architecture marker.
  // Excludes host topology work, allocations, uploads and downloads.
  CudaOperatorMeasurement measurement;
};
struct CudaLiquidWallContactResult3D {
  LiquidWallContactResult3D contact;
  // CUDA event time for the continuous sweep and both responses.
  CudaOperatorMeasurement measurement;
};
// Device clipping/curvature at the frozen start pose. Matches CPU contracts;
// unavailable hardware or image returns FailedPrecondition. No partial output.
// Host checks topology and stages one fixed POD query at a time; device computes
// clipping, curvature and capillary bounds. This bounded qualification path uses
// at most 32770 sequential kernels and two reusable buffers, without a CPU fallback.
absl::StatusOr<CudaLiquidBoundaryGeometry3D> CudaFreezeLiquidBoundaryGeometry3D(
    const LiquidBoundaryGeometryInputs3D& inputs
);
// Builds links with device theta/capillary arithmetic and checked host topology.
// Matches the CPU link contract; errors publish no problem.
absl::StatusOr<CudaLiquidPressureProblem3D>
CudaBuildLiquidPressureLinks3D(const FrozenLiquidBoundaryGeometry3D& geometry, const LiquidPressureFields3D& fields);
// Continuous device sweep and independent velocity responses. Matches the CPU
// contact contract and publishes no result on failure.
absl::StatusOr<CudaLiquidWallContactResult3D> CudaResolveLiquidWallContact3D(const LiquidWallContactInputs3D& inputs);
struct CudaLiquidBoundaryDeposits3D {
  LiquidBoundaryFaceDeposits3D deposits;
  // One sequential device kernel; excludes staging and synchronization overhead.
  CudaOperatorMeasurement measurement;
};
struct CudaNormalizedLiquidMacFaces3D {
  NormalizedLiquidMacFaces3D faces;
  // CUDA event time for decoding, normalization and validity decisions.
  CudaOperatorMeasurement measurement;
};
struct CudaLiquidBoundaryGatherResults3D {
  LiquidBoundaryGatherResults3D gathered;
  // CUDA event time for both histories, their QR/fallback and full gather.
  CudaOperatorMeasurement measurement;
};
// Checked device compatible deposition, including normalized basis, recentering,
// per-contribution nearest-even conversion and signed reduction. CPU API bounds,
// ownership and error contracts apply. No result publishes on any failure.
// All three transfer APIs use one sequential kernel and at most six allocations;
// they qualify bounded numerical operators, not production throughput.
absl::StatusOr<CudaLiquidBoundaryDeposits3D> CudaDepositCompatibleMacQuanta3D(
    const LiquidBoundaryDepositInputs3D& inputs
);
// Device quanta decoding, velocity normalization and independent mass validity.
// Matches NormalizeLiquidMacFaces3D; unavailable hardware rejects without output.
absl::StatusOr<CudaNormalizedLiquidMacFaces3D> CudaNormalizeLiquidMacFaces3D(
    std::span<const MacFaceDepositQuanta3D> face_quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass
);
// Device old/new ghost QR, deterministic fallback and complete gather. Host only
// validates and resolves support. Matches the CPU contract without CPU fallback.
absl::StatusOr<CudaLiquidBoundaryGatherResults3D> CudaGatherLiquidBoundaryParticles3D(
    const LiquidBoundaryGatherInputs3D& inputs
);
}  // namespace fluidity
