#pragma once

#include <cstddef>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/base/types.h"
#include "fluidity/host/grid.h"

// Host FP64 reference transfers for complete quadratic MAC stencils.
namespace fluidity {

struct ParticleView2D {
  // All spans have the same particle count and ordering.
  std::span<const Vec2> positions;  // World positions in m.
  std::span<const double> masses;  // Nonnegative masses in kg.
  std::span<const Vec2> velocities;  // Velocities in m/s.
  std::span<const Mat2> affine;  // Affine fits C in 1/s, one row per velocity component.
};

struct AffineFit2D {
  double value;  // Weighted value in the input units.
  Vec2 row;  // Spatial affine coefficients in input units per m.
};

// Phase selection for the history blend policy.
enum class Phase {
  kLiquid,
  kGas,
  kElastic,
  kGranular,
  kSnow
};

struct VelocityPair {
  Vec2 stored;  // History-blended particle velocity in m/s.
  Vec2 advecting;  // Velocity for position advection in m/s.
};

// Returns the required P2G scratch extent in doubles (two per face). Propagates FaceCount
// errors and returns kOutOfRange if the scratch count overflows.
absl::StatusOr<std::size_t> P2GScratchDoubles(const Grid2D& grid);

// Returns the required G2P scratch extent in doubles (12 per particle, including zero).
// Returns kOutOfRange if the scratch count overflows.
absl::StatusOr<std::size_t> G2PScratchDoubles(std::size_t particle_count);

// Deposits centered APIC mass (kg) and pure momentum (kg*m/s) into exact FaceCount-sized
// outputs in FaceGeometry order. Each particle needs all nine nodes on both staggered
// lattices, including zero-weight nodes. An empty particle batch writes zero to every face.
// Scratch has exactly P2GScratchDoubles entries. Writable spans must be disjoint from
// each other and all inputs. Scratch may change on failure. Outputs change only on success.
// Returns kInvalidArgument for wrong extents, overlap, nonfinite particle inputs, negative
// mass or an incomplete stencil. Propagates grid errors and returns kOutOfRange for
// unrepresentable scratch extents, stencil indices or nonfinite arithmetic.
absl::Status P2G(
    const Grid2D& grid,
    ParticleView2D particles,
    std::span<double> face_mass,
    std::span<double> face_momentum,
    std::span<double> scratch
);

// Gathers PIC and the new-minus-old grid velocity increment (m/s), affine fit C (1/s)
// and true velocity gradient L (1/s) at world positions in m. C and L have Cartesian
// rows in velocity-component order. Each particle needs all nine nodes on both staggered
// lattices, including zero-weight nodes. Both face arrays have exactly FaceCount entries
// in FaceGeometry order and may share storage. Each output has exactly positions.size()
// entries. Scratch has exactly G2PScratchDoubles entries.
// Empty positions are allowed. Writable spans must be disjoint from each other and all
// inputs. Scratch may change on failure. Outputs change only on success.
// Returns kInvalidArgument for wrong extents, overlap, nonfinite inputs or an incomplete
// stencil, kFailedPrecondition for a numerically singular fit and kOutOfRange for
// unrepresentable extents or indices or nonfinite arithmetic. Propagates grid errors.
absl::Status G2P(
    const Grid2D& grid,
    std::span<const Vec2> positions,
    std::span<const double> old_faces,
    std::span<const double> new_faces,
    std::span<Vec2> pic,
    std::span<Vec2> increments,
    std::span<Mat2> affine,
    std::span<Mat2> gradient,
    std::span<double> scratch
);

// Gathers a weighted scalar and fits its affine row using the full centered 2D moment.
// Position and sample positions are in m. Sample positions, weights, and values share
// the same nonzero extent and order and may share storage. Weights are
// dimensionless, nonnegative and sum to one within 2e-12. Returns kInvalidArgument for
// invalid inputs, kFailedPrecondition for a numerically singular moment, and kOutOfRange
// for nonfinite arithmetic. The smaller moment eigenvalue must exceed twice double
// epsilon times the larger one, to working precision.
absl::StatusOr<AffineFit2D> FitAffine2D(
    Vec2 position,
    std::span<const Vec2> samples,
    std::span<const double> weights,
    std::span<const double> values
);

// Sums scalar samples times their true interpolation weight gradients (1/m) in input
// order, returning input units per m. Inputs have the same nonzero extent and may share
// storage. Returns kInvalidArgument for invalid inputs and kOutOfRange for nonfinite arithmetic.
absl::StatusOr<Vec2> VelocityGradient2D(std::span<const double> values, std::span<const Vec2> gradients);

// Blends PIC and the particle velocity minus the old-grid gather (both in m/s).
// Stored velocity adds alpha times the residual to PIC. Advecting velocity adds
// beta times that retained residual. Energy and separation are dimensionless in [0, 1].
// Alpha is 0.99 times energy, and beta is energy times separation. Gas sets both
// factors to zero. Elastic sets beta to zero. Returns kInvalidArgument for invalid
// inputs and kOutOfRange for nonfinite arithmetic.
absl::StatusOr<VelocityPair> BlendHistory2D(Vec2 pic, Vec2 residual, double energy, Phase phase, double separation);

}  // namespace fluidity
