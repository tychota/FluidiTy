#pragma once
#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"
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
}  // namespace fluidity
