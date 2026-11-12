#include <algorithm>
#include <cmath>
#include <limits>

#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/surface_occupancy_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

using Occupancy = SurfaceOccupancyClass3D;

constexpr Vec3 kCenter{2.0, -3.0, 1.0};
constexpr double kOuterRadius = 1.4;
constexpr double kCavityRadius = 0.49;

struct ManufacturedBand {
  SurfaceLattice3D density{
      .origin = {kCenter.x - 1.75, kCenter.y - 1.75, kCenter.z - 1.75},
      .spacing = 0.25,
      .extent = {15, 15, 15},
      .values = {},
  };
  std::vector<Occupancy> classifications;

  SurfaceOccupancyView3D OccupancyView() const {
    return {
        .world_origin_m = density.origin,
        .spacing_m = density.spacing,
        .extent = density.extent,
        .classifications = classifications,
    };
  }
};

ManufacturedBand MakeBand(bool has_cavity) {
  ManufacturedBand band;
  const auto [nx, ny, nz] = band.density.extent;
  const double spacing = band.density.spacing;
  // Every cube crossed by either true interface lies inside this support halo.
  const double halo = std::sqrt(3.0) * spacing;
  for (std::size_t x = 0; x < nx; ++x) {
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t z = 0; z < nz; ++z) {
        const double radius = std::hypot(x * spacing - 1.75, y * spacing - 1.75, z * spacing - 1.75);
        const double exterior_field = radius - kOuterRadius;
        double particle_field = std::max(exterior_field, 0.79 - radius);
        Occupancy classification = Occupancy::kCertifiedLiquid;
        if (std::abs(exterior_field) <= halo) {
          classification = Occupancy::kSupportedInterface;
        } else if (exterior_field > 0) {
          classification = Occupancy::kCertifiedAir;
        }
        if (has_cavity) {
          const double cavity_field = kCavityRadius - radius;
          if (std::abs(cavity_field) <= halo) {
            classification = Occupancy::kSupportedInterface;
            particle_field = cavity_field;
          } else if (cavity_field > 0) {
            classification = Occupancy::kCertifiedAir;
            particle_field = -0.5;  // Wrong sign where particles have no support.
          }
        }
        band.density.values.push_back(particle_field);
        band.classifications.push_back(classification);
      }
    }
  }
  return band;
}

void ExpectClosedSurface(const SurfaceMetrics3D& metrics, std::size_t components) {
  EXPECT_EQ(metrics.components, components);
  EXPECT_EQ(metrics.euler_characteristic, 2 * components);
  EXPECT_EQ(metrics.boundary_edges, 0);
  EXPECT_EQ(metrics.nonmanifold_edges, 0);
  EXPECT_EQ(metrics.nonmanifold_vertices, 0);
  EXPECT_EQ(metrics.orientation_defects, 0);
  EXPECT_EQ(metrics.degenerate_triangles, 0);
  EXPECT_EQ(metrics.self_intersections, 0);
}

TEST(SurfaceOccupancy3D, MapsAcceptedGradedCoverageAndPreservesManufacturedTopology) {
  for (const bool has_cavity : {false, true}) {
    SCOPED_TRACE(has_cavity);
    const auto band = MakeBand(has_cavity);
    const auto [nx, ny, nz] = band.density.extent;
    std::vector<std::uint8_t> support(band.classifications.size());
    std::vector<AdaptiveMacCellKey3D> keys;
    std::vector<LiquidCellOccupancy3D> occupancies;
    for (std::size_t z = 0; z < nz; ++z) {
      for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
          const auto index = (x * ny + y) * nz + z;
          const auto classification = band.classifications[index];
          if (classification == Occupancy::kSupportedInterface) {
            support[index] = 1;
            continue;
          }
          // The corner is eight unsupported air nodes. Store it as one
          // coarse leaf so the mapped range crosses a real 2:1 transition.
          const bool in_coarse_corner = x < 2 && y < 2 && z < 2;
          if (in_coarse_corner && (x != 0 || y != 0 || z != 0)) {
            EXPECT_EQ(classification, Occupancy::kCertifiedAir);
            continue;
          }
          keys.push_back({
              .lower_corner = {static_cast<int>(x) - 2, static_cast<int>(y) - 2, static_cast<int>(z) - 2},
              .width_exponent = static_cast<std::uint8_t>(in_coarse_corner),
          });
          occupancies.push_back(
              classification == Occupancy::kCertifiedAir ? LiquidCellOccupancy3D::kAir : LiquidCellOccupancy3D::kLiquid
          );
        }
      }
    }
    auto topology = AdaptiveMacTopology3D::Create(keys, band.density.spacing);
    ASSERT_TRUE(topology.ok()) << topology.status();
    const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
    ASSERT_TRUE(ownership.ok()) << ownership.status();
    const auto face_count = ownership->interior_face_slots.size() + ownership->exterior_faces.size();
    LiquidState3D state;
    const auto origin = band.density.origin;
    state.geometry = {{origin.x + 0.5, origin.y + 0.5, origin.z + 0.5}, band.density.spacing};
    auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(keys.size());
    auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
    ASSERT_TRUE(cells.ok());
    ASSERT_TRUE(faces.ok());
    ASSERT_TRUE(cells->Resize(keys.size()).ok());
    ASSERT_TRUE(faces->Resize(face_count).ok());
    for (std::size_t ordinal = 0; ordinal < keys.size(); ++ordinal) {
      auto& cell = (*cells)[ordinal];
      cell.key = keys[ordinal];
      cell.occupancy = occupancies[ordinal];
      if (cell.occupancy == LiquidCellOccupancy3D::kLiquid) {
        cell.inventory.current_volume_m3 = topology->cells()[ordinal].cell_volume;
      }
    }
    state.adaptive_cells = std::move(*cells);
    state.adaptive_faces = std::move(*faces);
    const auto encoded = EncodeLiquidState3D(state);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    const auto decoded = DecodeLiquidState3D(encoded->values());
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    const auto corrected = ApplyAdaptiveSurfaceOccupancy3D(band.density, support, *decoded, 4000);
    ASSERT_TRUE(corrected.ok()) << corrected.status();
    const auto expected = ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000);
    ASSERT_TRUE(expected.ok()) << expected.status();
    EXPECT_EQ(corrected->values, expected->values);
    const auto mesh = ExtractSurfaceMesh3D(*corrected, 20000);
    ASSERT_TRUE(mesh.ok()) << mesh.status();
    const auto metrics = MeasureSurfaceMesh3D(*mesh);
    ASSERT_TRUE(metrics.ok()) << metrics.status();
    ExpectClosedSurface(*metrics, has_cavity ? 2 : 1);
    state.adaptive_cells[0].occupancy = LiquidCellOccupancy3D::kSolid;
    EXPECT_EQ(
        ApplyAdaptiveSurfaceOccupancy3D(band.density, support, state, 4000).status().code(),
        absl::StatusCode::kFailedPrecondition
    );
  }
}

TEST(SurfaceOccupancy3D, DeepLiquidRemovesFalseInnerSurfaceWithoutMovingExterior) {
  const auto band = MakeBand(false);
  auto raw = ExtractSurfaceMesh3D(band.density, 20000);
  ASSERT_TRUE(raw.ok()) << raw.status();
  const auto raw_metrics = MeasureSurfaceMesh3D(*raw);
  ASSERT_TRUE(raw_metrics.ok()) << raw_metrics.status();
  ExpectClosedSurface(*raw_metrics, 2);
  EXPECT_EQ(raw_metrics->positive_volume_components, 1);
  EXPECT_EQ(raw_metrics->negative_volume_components, 1);

  const auto merged = ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000);
  ASSERT_TRUE(merged.ok()) << merged.status();
  for (std::size_t index = 0; index < band.classifications.size(); ++index) {
    if (band.classifications[index] == Occupancy::kSupportedInterface) {
      EXPECT_DOUBLE_EQ(merged->values[index], band.density.values[index]);
    }
  }
  auto mesh = ExtractSurfaceMesh3D(*merged, 20000);
  ASSERT_TRUE(mesh.ok()) << mesh.status();
  const auto metrics = MeasureSurfaceMesh3D(*mesh);
  ASSERT_TRUE(metrics.ok()) << metrics.status();
  ExpectClosedSurface(*metrics, 1);
  EXPECT_EQ(metrics->positive_volume_components, 1);
  EXPECT_EQ(metrics->negative_volume_components, 0);
  EXPECT_GT(metrics->signed_volume, raw_metrics->signed_volume);

  std::vector<std::array<double, 3>> raw_exterior;
  for (const auto& vertex : raw->vertices) {
    const auto point = vertex.position;
    if (std::hypot(point.x - kCenter.x, point.y - kCenter.y, point.z - kCenter.z) > 1.1) {
      raw_exterior.push_back({point.x, point.y, point.z});
    }
  }
  std::vector<std::array<double, 3>> merged_exterior;
  for (const auto& vertex : mesh->vertices) {
    const auto point = vertex.position;
    merged_exterior.push_back({point.x, point.y, point.z});
  }
  EXPECT_EQ(merged_exterior, raw_exterior);
}

TEST(SurfaceOccupancy3D, PreservesClosedCavityWithLiquidToAirOrientation) {
  const auto band = MakeBand(true);
  const auto merged = ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000);
  ASSERT_TRUE(merged.ok()) << merged.status();
  const std::size_t center = (7 * 15 + 7) * 15 + 7;
  EXPECT_LT(band.density.values[center], 0);
  EXPECT_GT(merged->values[center], 0);
  auto mesh = ExtractSurfaceMesh3D(*merged, 20000);
  ASSERT_TRUE(mesh.ok()) << mesh.status();
  const auto metrics = MeasureSurfaceMesh3D(*mesh);
  ASSERT_TRUE(metrics.ok()) << metrics.status();
  ExpectClosedSurface(*metrics, 2);
  EXPECT_EQ(metrics->positive_volume_components, 1);
  EXPECT_EQ(metrics->negative_volume_components, 1);

  std::size_t cavity_vertices = 0;
  std::size_t exterior_vertices = 0;
  for (const auto& vertex : mesh->vertices) {
    const Vec3 radial{vertex.position.x - kCenter.x, vertex.position.y - kCenter.y, vertex.position.z - kCenter.z};
    const double orientation = radial.x * vertex.normal.x + radial.y * vertex.normal.y + radial.z * vertex.normal.z;
    if (std::hypot(radial.x, radial.y, radial.z) < 0.75) {
      EXPECT_LT(orientation, 0);
      ++cavity_vertices;
    } else {
      EXPECT_GT(orientation, 0);
      ++exterior_vertices;
    }
  }
  EXPECT_GT(cavity_vertices, 0);
  EXPECT_GT(exterior_vertices, 0);
  for (const auto triangle : mesh->triangles) {
    const Vec3 a = mesh->vertices[triangle[0]].position;
    const Vec3 b = mesh->vertices[triangle[1]].position;
    const Vec3 c = mesh->vertices[triangle[2]].position;
    const Vec3 ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 ac{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3 normal{ab.y * ac.z - ab.z * ac.y, ab.z * ac.x - ab.x * ac.z, ab.x * ac.y - ab.y * ac.x};
    const Vec3 radial{a.x - kCenter.x, a.y - kCenter.y, a.z - kCenter.z};
    const double orientation = radial.x * normal.x + radial.y * normal.y + radial.z * normal.z;
    if (std::hypot(radial.x, radial.y, radial.z) < 0.75) {
      EXPECT_LT(orientation, 0);
    } else {
      EXPECT_GT(orientation, 0);
    }
  }
}

TEST(SurfaceOccupancy3D, RejectsMissingRefusedAndMisalignedCoverage) {
  auto band = MakeBand(false);
  auto occupancy = band.OccupancyView();
  occupancy.classifications = {};
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(band.density, occupancy, 4000).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  occupancy = band.OccupancyView();
  occupancy.world_origin_m.x += 0.25;
  EXPECT_EQ(ApplySurfaceOccupancy3D(band.density, occupancy, 4000).status().code(), absl::StatusCode::kInvalidArgument);
  occupancy = band.OccupancyView();
  occupancy.spacing_m *= 2;
  EXPECT_EQ(ApplySurfaceOccupancy3D(band.density, occupancy, 4000).status().code(), absl::StatusCode::kInvalidArgument);
  occupancy = band.OccupancyView();
  --occupancy.extent[2];
  EXPECT_EQ(ApplySurfaceOccupancy3D(band.density, occupancy, 4000).status().code(), absl::StatusCode::kInvalidArgument);
  for (const auto refused : {Occupancy::kUnknown, Occupancy::kUnsupportedInterface}) {
    band.classifications.back() = refused;
    EXPECT_EQ(
        ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000).status().code(),
        absl::StatusCode::kFailedPrecondition
    );
  }
  band.classifications.back() = static_cast<Occupancy>(255);
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(SurfaceOccupancy3D, RejectsUnqualifiedCrossingsAndMalformedFields) {
  auto band = MakeBand(false);
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 10).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  auto collapsed = band;
  collapsed.density.origin.x = 1e20;
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(collapsed.density, collapsed.OccupancyView(), 4000).status().code(),
      absl::StatusCode::kOutOfRange
  );
  auto open = band;
  open.classifications.front() = Occupancy::kCertifiedLiquid;
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(open.density, open.OccupancyView(), 4000).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  const auto original = band.density.values;
  band.classifications[(7 * 15 + 7) * 15 + 7] = Occupancy::kCertifiedAir;
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(band.density.values, original);
  band.density.values.back() = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  band.density.values.pop_back();
  EXPECT_EQ(
      ApplySurfaceOccupancy3D(band.density, band.OccupancyView(), 4000).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

}  // namespace
}  // namespace fluidity
