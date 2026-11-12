#include "fluidity/host/surface_cache_3d.h"

#include <map>
#include <set>

#include "liquid_cache_3d_internal.h"
#include "surface_cache_3d_internal.h"
#include "surface_frame_generated.h"

namespace fluidity {
namespace {
namespace detail = liquid_cache_internal;
namespace wire = effects::v1;
absl::Status CheckCounts(std::size_t vertices, std::size_t triangles, SurfaceCacheLimits3D limits) {
  if (vertices > std::min(limits.maximum_vertices, std::size_t{1} << 20)
      || triangles > std::min(limits.maximum_triangles, std::size_t{1} << 20)) {
    return absl::ResourceExhaustedError("surface record budget exceeded");
  }
  return absl::OkStatus();
}
bool Safe(const SurfaceMetrics3D& m) {
  return m.boundary_edges == 0 && m.nonmanifold_edges == 0 && m.nonmanifold_vertices == 0 && m.orientation_defects == 0
      && m.degenerate_triangles == 0 && m.self_intersections == 0;
}
absl::Status ValidatePhaseInterface(
    const SurfaceFrame3D& frame,
    SurfaceCacheLimits3D limits,
    std::array<SurfaceMetrics3D, 2>* measured_phases = nullptr
) {
  if (!frame.phase_interface) {
    return absl::OkStatus();
  }
  const auto& sheet = *frame.phase_interface;
  const auto vertex_count = frame.mesh.vertices.size() + sheet.interior_positions.size();
  const auto counts = CheckCounts(vertex_count, frame.mesh.triangles.size() + sheet.triangles.size(), limits);
  if (!counts.ok()) {
    return counts;
  }
  if (sheet.phase_a == 0 || sheet.phase_a >= sheet.phase_b || sheet.triangles.empty()
      || sheet.exterior_sides.size() != frame.mesh.triangles.size()) {
    return absl::InvalidArgumentError("invalid shared surface phase sheet");
  }
  for (const auto side : sheet.exterior_sides) {
    if (side > 1) {
      return absl::InvalidArgumentError("invalid exterior phase side");
    }
  }
  for (const auto& triangle : sheet.triangles) {
    if (*std::max_element(triangle.begin(), triangle.end()) >= vertex_count) {
      return absl::InvalidArgumentError("shared surface index outside vertices");
    }
  }
  if (!sheet.side_compositions.empty()) {
    std::vector<bool> on_sheet(vertex_count);
    for (const auto triangle : sheet.triangles) {
      for (const auto vertex : triangle) {
        on_sheet[vertex] = true;
      }
    }
    std::map<std::uint64_t, std::uint64_t> substance_phases;
    std::size_t record = 0;
    for (std::size_t ordinal = 0; ordinal < on_sheet.size(); ++ordinal) {
      if (!on_sheet[ordinal]) {
        continue;
      }
      for (const auto phase : {sheet.phase_a, sheet.phase_b}) {
        if (record >= sheet.side_compositions.size()) {
          return absl::InvalidArgumentError("shared surface composition is incomplete");
        }
        const auto& entry = sheet.side_compositions[record++];
        if (entry.vertex_ordinal != ordinal || entry.phase_id != phase || entry.constituent_count == 0
            || entry.constituent_count > entry.constituents.size()
            || !ValidLiquidComposition3D(phase, std::span(entry.constituents).first(entry.constituent_count))) {
          return absl::InvalidArgumentError("invalid shared surface composition");
        }
        for (const auto& constituent : std::span(entry.constituents).first(entry.constituent_count)) {
          const auto [found, inserted] = substance_phases.try_emplace(constituent.substance_id, phase);
          if (!inserted && found->second != phase) {
            return absl::InvalidArgumentError("shared surface substance appears in both phases");
          }
        }
      }
    }
    if (record != sheet.side_compositions.size()) {
      return absl::InvalidArgumentError("shared surface composition has unused records");
    }
  }
  for (std::uint8_t side = 0; side < 2; ++side) {
    SurfaceMesh3D phase;
    std::vector<std::uint32_t> local_indices(vertex_count, UINT32_MAX);
    const auto append = [&](std::array<std::uint32_t, 3> triangle) {
      for (auto& index : triangle) {
        auto& local = local_indices[index];
        if (local == UINT32_MAX) {
          const auto position = index < frame.mesh.vertices.size()
              ? frame.mesh.vertices[index].position
              : sheet.interior_positions[index - frame.mesh.vertices.size()];
          local = static_cast<std::uint32_t>(phase.vertices.size());
          phase.vertices.push_back({.position = position});
        }
        index = local;
      }
      phase.triangles.push_back(triangle);
    };
    for (std::size_t index = 0; index < frame.mesh.triangles.size(); ++index) {
      if (sheet.exterior_sides[index] == side) {
        append(frame.mesh.triangles[index]);
      }
    }
    for (auto triangle : sheet.triangles) {
      if (side == 1) {
        std::swap(triangle[1], triangle[2]);
      }
      append(triangle);
    }
    const auto interior = std::span(local_indices).subspan(frame.mesh.vertices.size());
    if (std::ranges::find(interior, UINT32_MAX) != interior.end()) {
      return absl::InvalidArgumentError("unused shared surface interior vertex");
    }
    const auto metrics = MeasureSurfaceMesh3D(phase);
    if (!metrics.ok()) {
      return metrics.status();
    }
    if (!Safe(*metrics) || !detail::Positive(metrics->signed_volume)) {
      return absl::InvalidArgumentError("shared surface phase is open, intersecting or incorrectly oriented");
    }
    if (measured_phases != nullptr) {
      (*measured_phases)[side] = *metrics;
    }
  }
  return absl::OkStatus();
}
absl::Status Validate(const SurfaceFrame3D& frame, SurfaceCacheLimits3D limits) {
  const auto counts = CheckCounts(frame.mesh.vertices.size(), frame.mesh.triangles.size(), limits);
  if (!counts.ok()) {
    return counts;
  }
  if (!detail::ValidClock(frame.clock) || frame.clock.substep_ordinal != 0
      || (frame.clock.frame_ordinal == 0 && frame.clock != LiquidClock3D{}) || !detail::Positive(frame.cell_width_m)
      || frame.mesh.next_id == 0) {
    return absl::InvalidArgumentError("invalid surface clock or width");
  }
  std::set<std::uint64_t> ids;
  std::map<std::uint64_t, SurfaceMesh3D> components;
  std::vector<std::uint32_t> local_indices;
  for (const auto& v : frame.mesh.vertices) {
    if (v.id == 0 || v.id >= frame.mesh.next_id || !ids.insert(v.id).second || v.component == 0
        || !detail::Finite(v.position) || !detail::Finite(v.geometric_velocity) || !detail::Finite(v.fluid_velocity)
        || !detail::Finite(v.normal) || std::abs(std::hypot(v.normal.x, v.normal.y, v.normal.z) - 1) > 1e-8
        || !detail::Nonnegative(v.age_seconds) || !detail::Nonnegative(v.wetness) || v.wetness > 1
        || !detail::Nonnegative(v.thickness_m) || v.parent_id >= frame.mesh.next_id
        || v.constituent_count > v.constituents.size()
        || !ValidLiquidComposition3D(v.phase_id, std::span(v.constituents).first(v.constituent_count))) {
      return absl::InvalidArgumentError("invalid persistent surface vertex");
    }
    auto& component = components[v.component];
    local_indices.push_back(static_cast<std::uint32_t>(component.vertices.size()));
    component.vertices.push_back(v);
  }
  for (const auto triangle : frame.mesh.triangles) {
    if (*std::max_element(triangle.begin(), triangle.end()) >= frame.mesh.vertices.size()) {
      return absl::InvalidArgumentError("surface index outside vertices");
    }
    const auto component = frame.mesh.vertices[triangle[0]].component;
    if (frame.mesh.vertices[triangle[1]].component != component
        || frame.mesh.vertices[triangle[2]].component != component) {
      return absl::InvalidArgumentError("surface triangle crosses material components");
    }
    components[component].triangles.push_back(
        {local_indices[triangle[0]], local_indices[triangle[1]], local_indices[triangle[2]]}
    );
  }
  if (frame.volumes.size() != components.size()) {
    return absl::InvalidArgumentError("surface component volume coverage differs");
  }
  auto component = components.begin();
  for (const auto& volume : frame.volumes) {
    if (volume.component != component->first || !detail::Positive(volume.material_m3)
        || !detail::Positive(volume.raw_m3) || !detail::Positive(volume.corrected_m3)) {
      return absl::InvalidArgumentError("invalid surface component volumes");
    }
    auto measured = MeasureSurfaceMesh3D(component->second);
    if (!measured.ok()) {
      return measured.status();
    }
    if (!Safe(*measured) || std::abs(measured->signed_volume - volume.corrected_m3) > 1e-10 * volume.corrected_m3
        || std::abs(volume.corrected_m3 - volume.material_m3) > 0.005 * volume.material_m3) {
      return absl::InvalidArgumentError("surface component fails measured topology or volume gate");
    }
    ++component;
  }
  if (components.size() > 1) {
    const auto measured = MeasureSurfaceMesh3D(frame.mesh);
    if (!measured.ok()) {
      return measured.status();
    }
    if (!Safe(*measured)) {
      return absl::InvalidArgumentError("surface components intersect");
    }
  }
  return ValidatePhaseInterface(frame, limits);
}
}  // namespace

absl::Status surface_cache_internal::CheckSharedSurfaceTopology3D(
    const SurfaceFrame3D& previous,
    const SurfaceFrame3D& current,
    const SurfaceMetrics3D& current_exterior,
    SurfaceCacheLimits3D limits
) {
  const auto same_topology = [](const SurfaceMetrics3D& first, const SurfaceMetrics3D& second) {
    return first.components == second.components
        && first.positive_volume_components == second.positive_volume_components
        && first.negative_volume_components == second.negative_volume_components
        && first.euler_characteristic == second.euler_characteristic;
  };
  const auto previous_exterior = MeasureSurfaceMesh3D(previous.mesh);
  if (!previous_exterior.ok()) {
    return previous_exterior.status();
  }
  if (!same_topology(*previous_exterior, current_exterior)) {
    return absl::FailedPreconditionError("shared surface exterior topology change requires lineage");
  }

  std::array<SurfaceMetrics3D, 2> previous_phases;
  const auto previous_status = ValidatePhaseInterface(previous, limits, &previous_phases);
  if (!previous_status.ok()) {
    return previous_status;
  }
  std::array<SurfaceMetrics3D, 2> current_phases;
  const auto current_status = ValidatePhaseInterface(current, limits, &current_phases);
  if (!current_status.ok()) {
    return current_status;
  }
  for (std::size_t side = 0; side < previous_phases.size(); ++side) {
    if (!same_topology(previous_phases[side], current_phases[side])) {
      return absl::FailedPreconditionError("shared surface phase topology change requires lineage");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<LiquidCacheBytes3D> EncodeSurfaceFrame3D(const SurfaceFrame3D& frame, SurfaceCacheLimits3D limits) {
  const auto status = Validate(frame, limits);
  if (!status.ok()) {
    return status;
  }
  std::vector<wire::SurfaceVertex> vertices;
  for (const auto& v : frame.mesh.vertices) {
    vertices.emplace_back(
        v.id,
        detail::Pack(v.position),
        detail::Pack(v.geometric_velocity),
        detail::Pack(v.fluid_velocity),
        detail::Pack(v.normal),
        v.age_seconds,
        v.wetness,
        v.thickness_m,
        v.component,
        v.parent_id,
        static_cast<std::uint8_t>((v.contact ? 1 : 0) | (v.unresolved_sheet ? 2 : 0))
    );
  }
  std::vector<wire::SurfaceTriangle> triangles;
  for (const auto t : frame.mesh.triangles) {
    triangles.emplace_back(t[0], t[1], t[2]);
  }
  std::vector<wire::SurfaceVolume> volumes;
  for (const auto& v : frame.volumes) {
    volumes.emplace_back(v.component, v.material_m3, v.raw_m3, v.corrected_m3);
  }
  std::vector<std::uint32_t> composed_vertices;
  for (std::size_t ordinal = 0; ordinal < frame.mesh.vertices.size(); ++ordinal) {
    if (frame.mesh.vertices[ordinal].constituent_count != 0) {
      composed_vertices.push_back(static_cast<std::uint32_t>(ordinal));
    }
  }
  std::vector<wire::SurfaceTriangle> shared_triangles;
  std::vector<cache::v1::Vector3> interior_positions;
  if (frame.phase_interface) {
    for (const auto triangle : frame.phase_interface->triangles) {
      shared_triangles.emplace_back(triangle[0], triangle[1], triangle[2]);
    }
    for (const auto position : frame.phase_interface->interior_positions) {
      interior_positions.push_back(detail::Pack(position));
    }
  }
  const auto bytes = vertices.size() * sizeof(wire::SurfaceVertex) + triangles.size() * sizeof(wire::SurfaceTriangle)
      + volumes.size() * sizeof(wire::SurfaceVolume) + shared_triangles.size() * sizeof(wire::SurfaceTriangle)
      + interior_positions.size() * sizeof(cache::v1::Vector3)
      + (frame.phase_interface ? frame.phase_interface->exterior_sides.size() : 0);
  return detail::Build<wire::SurfaceFrame>(
      2 + (bytes + 511) / 512 + composed_vertices.size() + frame.phase_interface.has_value()
          + (frame.phase_interface ? frame.phase_interface->side_compositions.size() : 0),
      limits.cache,
      "FLMS",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::SurfaceFrame>> {
        const auto vertex_data = builder.CreateVectorOfStructs(vertices);
        const auto triangle_data = builder.CreateVectorOfStructs(triangles);
        const auto volume_data = builder.CreateVectorOfStructs(volumes);
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<wire::SurfaceComposition>>> compositions;
        if (!composed_vertices.empty()) {
          const auto records =
              detail::BuildVector<wire::SurfaceComposition>(builder, composed_vertices.size(), [&](std::size_t index) {
                const auto ordinal = composed_vertices[index];
                const auto packed = detail::PackComposition(builder, frame.mesh.vertices[ordinal]);
                return wire::CreateSurfaceComposition(builder, ordinal, packed.phase_id, packed.constituents);
              });
          if (!records.ok()) {
            return records.status();
          }
          compositions = *records;
        }
        flatbuffers::Offset<wire::SurfacePhaseInterface> phase_interface;
        if (frame.phase_interface) {
          const auto& sheet = *frame.phase_interface;
          const auto sides = builder.CreateVector(sheet.exterior_sides);
          const auto positions = builder.CreateVectorOfStructs(interior_positions);
          const auto shared = builder.CreateVectorOfStructs(shared_triangles);
          flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<wire::SurfaceComposition>>> side_compositions;
          if (!sheet.side_compositions.empty()) {
            const auto records = detail::BuildVector<wire::SurfaceComposition>(
                builder,
                sheet.side_compositions.size(),
                [&](std::size_t index) {
                  const auto& entry = sheet.side_compositions[index];
                  const auto packed = detail::PackComposition(builder, entry);
                  return wire::CreateSurfaceComposition(
                      builder,
                      entry.vertex_ordinal,
                      packed.phase_id,
                      packed.constituents
                  );
                }
            );
            if (!records.ok()) {
              return records.status();
            }
            side_compositions = *records;
          }
          phase_interface = wire::CreateSurfacePhaseInterface(
              builder,
              sheet.phase_a,
              sheet.phase_b,
              sides,
              positions,
              shared,
              side_compositions
          );
        }
        const auto clock = detail::Pack(frame.clock);
        return wire::CreateSurfaceFrame(
            builder,
            1,
            0,
            &clock,
            frame.cell_width_m,
            frame.mesh.next_id,
            vertex_data,
            triangle_data,
            volume_data,
            compositions,
            phase_interface
        );
      }
  );
}

absl::StatusOr<SurfaceFrame3D> DecodeSurfaceFrame3D(std::span<const std::uint8_t> bytes, SurfaceCacheLimits3D limits) {
  const auto verified = detail::Verify<wire::SurfaceFrame>(bytes, "FLMS", limits.cache);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* source = *verified;
  const auto counts = CheckCounts(source->vertices()->size(), source->triangles()->size(), limits);
  if (!counts.ok()) {
    return counts;
  }
  if (source->volumes()->size() > source->vertices()->size()) {
    return absl::DataLossError("surface volume count exceeds vertices");
  }
  SurfaceFrame3D frame{.clock = detail::Unpack(source->clock()), .cell_width_m = source->cell_width_m()};
  frame.mesh.next_id = source->next_id();
  for (const auto* v : *source->vertices()) {
    if (v->flags() > 3) {
      return absl::DataLossError("unknown surface vertex flags");
    }
    frame.mesh.vertices.push_back({
        .id = v->id(),
        .position = detail::Unpack(&v->position()),
        .geometric_velocity = detail::Unpack(&v->geometric_velocity()),
        .fluid_velocity = detail::Unpack(&v->fluid_velocity()),
        .normal = detail::Unpack(&v->normal()),
        .age_seconds = v->age_seconds(),
        .wetness = v->wetness(),
        .thickness_m = v->thickness_m(),
        .component = v->component(),
        .parent_id = v->parent_id(),
        .contact = (v->flags() & 1) != 0,
        .unresolved_sheet = (v->flags() & 2) != 0,
    });
  }
  if (const auto* compositions = source->compositions()) {
    if (compositions->size() > frame.mesh.vertices.size()) {
      return absl::DataLossError("surface composition count exceeds vertices");
    }
    for (flatbuffers::uoffset_t index = 0; index < compositions->size(); ++index) {
      const auto* composition = compositions->Get(index);
      const auto ordinal = composition->vertex_ordinal();
      if (ordinal >= frame.mesh.vertices.size()
          || (index > 0 && compositions->Get(index - 1)->vertex_ordinal() >= ordinal)
          || !detail::ValidCompositionEncoding(composition)) {
        return absl::DataLossError("invalid surface composition record");
      }
      detail::UnpackComposition(composition, frame.mesh.vertices[ordinal]);
    }
  }
  for (const auto* t : *source->triangles()) {
    frame.mesh.triangles.push_back({t->a(), t->b(), t->c()});
  }
  if (const auto* sheet = source->phase_interface()) {
    const auto shared_counts = CheckCounts(
        frame.mesh.vertices.size() + sheet->interior_positions()->size(),
        frame.mesh.triangles.size() + sheet->triangles()->size(),
        limits
    );
    if (!shared_counts.ok()) {
      return shared_counts;
    }
    if (sheet->exterior_sides()->size() != frame.mesh.triangles.size()) {
      return absl::DataLossError("invalid exterior phase side count");
    }
    auto& target = frame.phase_interface.emplace();
    target.phase_a = sheet->phase_a();
    target.phase_b = sheet->phase_b();
    target.exterior_sides.assign(sheet->exterior_sides()->begin(), sheet->exterior_sides()->end());
    for (const auto* position : *sheet->interior_positions()) {
      target.interior_positions.push_back(detail::Unpack(position));
    }
    if (const auto* records = sheet->side_compositions()) {
      const auto vertex_count = frame.mesh.vertices.size() + sheet->interior_positions()->size();
      if (records->size() > 2 * vertex_count) {
        return absl::DataLossError("shared surface composition exceeds vertex coverage");
      }
      for (const auto* record : *records) {
        if (!detail::ValidCompositionEncoding(record)) {
          return absl::DataLossError("invalid shared surface composition record");
        }
        SurfaceInterfaceComposition3D entry{};
        entry.vertex_ordinal = record->vertex_ordinal();
        detail::UnpackComposition(record, entry);
        target.side_compositions.push_back(entry);
      }
    }
    for (const auto* triangle : *sheet->triangles()) {
      target.triangles.push_back({triangle->a(), triangle->b(), triangle->c()});
    }
  }
  for (const auto* v : *source->volumes()) {
    frame.volumes.push_back({v->component(), v->material_m3(), v->raw_m3(), v->corrected_m3()});
  }
  const auto status = Validate(frame, limits);
  if (!status.ok()) {
    return absl::IsResourceExhausted(status) ? status : absl::DataLossError(status.message());
  }
  return frame;
}
}  // namespace fluidity
