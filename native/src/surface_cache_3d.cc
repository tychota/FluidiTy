#include "fluidity/host/surface_cache_3d.h"

#include <map>
#include <set>

#include "liquid_cache_3d_internal.h"
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
        || !detail::Nonnegative(v.thickness_m) || v.parent_id >= frame.mesh.next_id) {
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
  return absl::OkStatus();
}
}  // namespace

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
  const auto bytes = vertices.size() * sizeof(wire::SurfaceVertex) + triangles.size() * sizeof(wire::SurfaceTriangle)
      + volumes.size() * sizeof(wire::SurfaceVolume);
  return detail::Build<wire::SurfaceFrame>(
      2 + (bytes + 511) / 512,
      limits.cache,
      "FLMS",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::SurfaceFrame>> {
        const auto vertex_data = builder.CreateVectorOfStructs(vertices);
        const auto triangle_data = builder.CreateVectorOfStructs(triangles);
        const auto volume_data = builder.CreateVectorOfStructs(volumes);
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
            volume_data
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
  for (const auto* t : *source->triangles()) {
    frame.mesh.triangles.push_back({t->a(), t->b(), t->c()});
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
