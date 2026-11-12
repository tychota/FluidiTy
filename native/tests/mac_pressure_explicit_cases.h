#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "../src/mac_pressure_3d_system.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/viscosity.h"

namespace fluidity::pressure_test {

struct AdaptiveCutPressureFixture {
  AdaptiveMacTopology3D topology;
  std::array<MacPressureCell3D, 2> cells;
  std::vector<pressure_internal::AdaptiveMacPressureSubface3D> subfaces;
  std::size_t internal_slot;
  std::size_t outlet_slot;
};

// A source feeds a coarse leaf through a partly blocked subface. The moving
// wall supplies part of the interior flux, and an open fine face drains it.
inline absl::StatusOr<AdaptiveCutPressureFixture> MakeAdaptiveCutPressureFixture() {
  const std::array keys{
      AdaptiveMacCellKey3D{.lower_corner = {0, 0, 0}, .width_exponent = 1},
      AdaptiveMacCellKey3D{.lower_corner = {2, 0, 0}, .width_exponent = 0},
  };
  auto topology = AdaptiveMacTopology3D::Create(keys, 0.5);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  if (ownership->interior_face_slots.size() != 1 || ownership->exterior_faces.size() != 13) {
    return absl::InternalError("unexpected adaptive pressure fixture ownership");
  }
  const auto outlet = std::ranges::find_if(ownership->exterior_faces, [](const auto& face) {
    return face.cell_ordinal == 1 && face.cell_is_tail;
  });
  if (outlet == ownership->exterior_faces.end()) {
    return absl::InternalError("missing adaptive pressure fixture outlet");
  }
  AdaptiveCutPressureFixture fixture{
      .topology = std::move(*topology),
      .cells = {{
          {.cell_index = {0, 0, 0},
           .fluid_volume = 1,
           .target_divergence = 0.09375,
           .bulk_modulus = std::numeric_limits<double>::infinity(),
           .previous_pressure = 0},
          {.cell_index = {2, 0, 0},
           .fluid_volume = 0.125,
           .target_divergence = 0,
           .bulk_modulus = std::numeric_limits<double>::infinity(),
           .previous_pressure = 0},
      }},
      .subfaces = std::vector<pressure_internal::AdaptiveMacPressureSubface3D>(14),
      .internal_slot = ownership->interior_face_slots.front(),
      .outlet_slot = outlet->face_slot,
  };
  fixture.subfaces[fixture.internal_slot] = {
      .open_area_m2 = 0.125,
      .inverse_density_m3_per_kg = 1,
      .prescribed_wall_flux_m3_per_s = 0.0625,
  };
  fixture.subfaces[fixture.outlet_slot] = {
      .open_area_m2 = outlet->face_area,
      .inverse_density_m3_per_kg = 1,
      .exterior_pressure_distance_m = 0.25,
      .exterior_pressure_pa = 1.0,
  };
  return fixture;
}
// Gauge-fixed pressures from the independent 3D NumPy fixture.
struct ExplicitPressureCase {
  std::array<std::size_t, 3> axes;
  std::size_t plane_count;
  bool reversed;
  std::array<double, 11> reference_pressure;
};

inline constexpr std::array kExplicitPressureCases{
    ExplicitPressureCase{
        .axes = {0, 1, 2},
        .plane_count = 2,
        .reversed = false,
        .reference_pressure =
            {
                0,
                0,
                0.4898695054945057,
                -0.02111950549450526,
                0.32194368131868156,
                -0.040693681318681146,
                0.3219436813186816,
                -0.040693681318681035,
                0.2312843406593409,
                0.04996565934065957,
            },
    },
    ExplicitPressureCase{
        .axes = {1, 2, 0},
        .plane_count = 2,
        .reversed = true,
        .reference_pressure =
            {
                0,
                0,
                0.39766483516483525,
                0.4203296703296704,
                0.42032967032967045,
                0.06387362637362647,
                0.3523351648351649,
                0.32967032967032983,
                0.3296703296703298,
                -0.06387362637362629,
            },
    },
    ExplicitPressureCase{
        .axes = {0, 1, 2},
        .plane_count = 3,
        .reversed = false,
        .reference_pressure =
            {
                0,
                0.06617647058823546,
                0.13235294117647078,
                0.6034899034240563,
                0.01316944688323106,
                0.21169885864793697,
                -0.07835820895522373,
                0.4102282704126429,
                -0.012181738366988364,
                0.0539947322212469,
                -0.012181738366988406,
            },
    },
    ExplicitPressureCase{
        .axes = {1, 2, 0},
        .plane_count = 3,
        .reversed = true,
        .reference_pressure =
            {
                0,
                -0.06617647058823489,
                0.0661764705882357,
                0.4533582089552242,
                0.387181738366989,
                0.4533582089552241,
                0.030948200175592924,
                0.5195346795434596,
                0.229477611940299,
                0.4280070237050047,
                -0.16231343283582053,
            },
    },
};

struct ExplicitPressureFixture {
  std::vector<MacPressureCell3D> cells;
  std::vector<pressure_internal::ExplicitMacPressureFace3D> faces;
};

inline ExplicitPressureFixture MakeExplicitPressureFixture(const ExplicitPressureCase& pressure_case) {
  ExplicitPressureFixture fixture;
  const std::size_t coarse_count = pressure_case.plane_count;
  for (std::size_t cell = 0; cell < coarse_count + 8; ++cell) {
    fixture.cells.push_back({
        .cell_index = {static_cast<std::int32_t>(cell) - 8, -2, 0},
        .fluid_volume = cell < coarse_count ? 1.0 : 0.125,
        .target_divergence = 0.0,
        .bulk_modulus = std::numeric_limits<double>::infinity(),
        .previous_pressure = 0.0,
    });
  }
  const auto add_face = [&](std::size_t tail, std::size_t head, double distance, double flux) {
    if (pressure_case.reversed) {
      std::swap(tail, head);
      flux = -flux;
    }
    fixture.faces.push_back({
        .face_slot = fixture.faces.size(),
        .tail_cell_ordinal = tail,
        .head_cell_ordinal = head,
        .open_area = 0.25,
        .inverse_density = 1.0,
        .pressure_gradient_distance = distance,
        .predicted_normal_velocity = flux / 0.25,
    });
  };
  constexpr std::array predicted_subface_fluxes{0.375, -0.125, 0.25, -0.25};
  for (std::size_t coarse = 0; coarse < coarse_count; ++coarse) {
    const std::size_t normal_bit = std::size_t{1} << (2 - pressure_case.axes[coarse]);
    std::size_t subface = 0;
    for (std::size_t fine = 0; fine < 8; ++fine) {
      if (((fine & normal_bit) != 0) == pressure_case.reversed) {
        add_face(coarse, coarse_count + fine, 0.75, predicted_subface_fluxes[subface++]);
      }
    }
  }
  for (std::size_t tail = 0; tail < 8; ++tail) {
    for (std::size_t head = tail + 1; head < 8; ++head) {
      const std::size_t differing_coordinates = tail ^ head;
      if (std::has_single_bit(differing_coordinates)) {
        add_face(coarse_count + tail, coarse_count + head, 0.5, 0.0);
      }
    }
  }
  return fixture;
}

inline std::array<MacStrainRow3D, 2> ConfigureAdaptiveNormalTraction(AdaptiveCutPressureFixture& fixture) {
  const auto interior = fixture.internal_slot;
  const auto outlet = fixture.outlet_slot;
  // Manufacture u = (1/4, 3/8) and p = (2, 5/2) by balancing the open
  // subface inertia, pressure and strain terms. The blocked flux is unchanged.
  fixture.subfaces[interior].predicted_normal_velocity_m_per_s = 0.25;
  fixture.subfaces[outlet].predicted_normal_velocity_m_per_s = -3.625;
  return {
      MacStrainRow3D{
          .low_cell = {0, 0, 0},
          .first_axis = FaceNormalAxis::kX,
          .second_axis = FaceNormalAxis::kX,
          .terms = {{{interior, 1.0}}},
          .term_count = 1,
          .quadrature_weight = 0.25,
      },
      MacStrainRow3D{
          .low_cell = {2, 0, 0},
          .first_axis = FaceNormalAxis::kX,
          .second_axis = FaceNormalAxis::kX,
          .terms = {{{interior, -2.0}, {outlet, 2.0}}},
          .term_count = 2,
          .quadrature_weight = 0.25,
      },
  };
}

}  // namespace fluidity::pressure_test
