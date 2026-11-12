#include "fluidity/host/surface_mesh_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

#include "surface_predicates_3d_internal.h"

namespace fluidity {
namespace {
Vec3 Add(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 Sub(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 Scale(Vec3 a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}
double Dot(Vec3 a, Vec3 b) {
  return (a.x * b.x + a.y * b.y) + a.z * b.z;
}
Vec3 Cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double Norm(Vec3 a) {
  return std::hypot(a.x, a.y, a.z);
}
bool Finite(Vec3 a) {
  return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

struct DisjointSet {
  explicit DisjointSet(std::size_t n) : parent(n) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  std::size_t Find(std::size_t index) {
    while (parent[index] != index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  }
  void Join(std::size_t a, std::size_t b) {
    parent[Find(b)] = Find(a);
  }
  std::vector<std::size_t> parent;
};

bool Intersects(
    const std::array<Vec3, 3>& a,
    const std::array<Vec3, 3>& b,
    const std::array<bool, 3>& shared_a,
    const std::array<bool, 3>& shared_b
) {
  using surface_internal::Orient2D;
  using surface_internal::Orient3D;
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const auto next = (edge + 1) % 3;
    if ((!shared_a[edge] && !shared_a[next] && surface_internal::SegmentTriangle(a[edge], a[next], b))
        || (!shared_b[edge] && !shared_b[next] && surface_internal::SegmentTriangle(b[edge], b[next], a))) {
      return true;
    }
  }
  if (Orient3D(a[0], a[1], a[2], b[0]) != 0 || Orient3D(a[0], a[1], a[2], b[1]) != 0
      || Orient3D(a[0], a[1], a[2], b[2]) != 0) {
    return false;
  }
  int drop = 0;
  while (drop < 3 && Orient2D(a[0], a[1], a[2], drop) == 0) {
    ++drop;
  }
  if (drop == 3) {
    return false;  // Degenerate triangles are reported separately.
  }
  const auto inside = [&](Vec3 p, const auto& t) {
    return surface_internal::SameSide(
        Orient2D(t[0], t[1], p, drop),
        Orient2D(t[1], t[2], p, drop),
        Orient2D(t[2], t[0], p, drop)
    );
  };
  if (std::all_of(shared_a.begin(), shared_a.end(), [](bool shared) {
        return shared;
      })) {
    return true;
  }
  for (std::size_t vertex = 0; vertex < 3; ++vertex) {
    if ((!shared_a[vertex] && inside(a[vertex], b)) || (!shared_b[vertex] && inside(b[vertex], a))) {
      return true;
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      if (Orient2D(a[i], a[(i + 1) % 3], b[j], drop) * Orient2D(a[i], a[(i + 1) % 3], b[(j + 1) % 3], drop) < 0
          && Orient2D(b[j], b[(j + 1) % 3], a[i], drop) * Orient2D(b[j], b[(j + 1) % 3], a[(i + 1) % 3], drop) < 0) {
        return true;
      }
    }
  }
  return false;
}

void UpdateNormals(SurfaceMesh3D& mesh) {
  for (auto& vertex : mesh.vertices) {
    vertex.normal = {};
  }
  for (const auto triangle : mesh.triangles) {
    const auto normal = Cross(
        Sub(mesh.vertices[triangle[1]].position, mesh.vertices[triangle[0]].position),
        Sub(mesh.vertices[triangle[2]].position, mesh.vertices[triangle[0]].position)
    );
    for (const auto index : triangle) {
      mesh.vertices[index].normal = Add(mesh.vertices[index].normal, normal);
    }
  }
  for (auto& vertex : mesh.vertices) {
    const double length = Norm(vertex.normal);
    if (length > 0) {
      vertex.normal = Scale(vertex.normal, 1 / length);
    }
  }
}

bool Safe(const SurfaceMetrics3D& metrics) {
  return metrics.boundary_edges == 0 && metrics.nonmanifold_edges == 0 && metrics.nonmanifold_vertices == 0
      && metrics.orientation_defects == 0 && metrics.degenerate_triangles == 0 && metrics.self_intersections == 0;
}
}  // namespace

absl::StatusOr<SurfaceMesh3D> ExtractSurfaceMesh3D(const SurfaceLattice3D& lattice, std::size_t maximum_vertices) {
  const auto dimensions = lattice.extent;
  if (!Finite(lattice.origin) || !std::isfinite(lattice.spacing) || lattice.spacing <= 0
      || *std::min_element(dimensions.begin(), dimensions.end()) < 2 || maximum_vertices > UINT32_MAX
      || dimensions[0] > lattice.values.size() / dimensions[1] / dimensions[2]
      || dimensions[0] * dimensions[1] * dimensions[2] != lattice.values.size()) {
    return absl::InvalidArgumentError("invalid surface lattice extents or geometry");
  }
  const auto node_id = [&](std::size_t x, std::size_t y, std::size_t z) {
    return (x * dimensions[1] + y) * dimensions[2] + z;
  };
  for (std::size_t x = 0; x < dimensions[0]; ++x) {
    for (std::size_t y = 0; y < dimensions[1]; ++y) {
      for (std::size_t z = 0; z < dimensions[2]; ++z) {
        const double value = lattice.values[node_id(x, y, z)];
        if (!std::isfinite(value)) {
          return absl::InvalidArgumentError("surface lattice is nonfinite");
        }
        if (value <= 0
            && (x == 0 || y == 0 || z == 0 || x + 1 == dimensions[0] || y + 1 == dimensions[1] || z + 1 == dimensions[2]
            )) {
          return absl::FailedPreconditionError("surface reaches the lattice boundary");
        }
      }
    }
  }
  constexpr std::array<std::array<std::size_t, 3>, 8> corners{{
      {0, 0, 0},
      {1, 0, 0},
      {0, 1, 0},
      {1, 1, 0},
      {0, 0, 1},
      {1, 0, 1},
      {0, 1, 1},
      {1, 1, 1},
  }};
  constexpr std::array<std::array<int, 4>, 6> tetrahedra{{
      {0, 1, 3, 7},
      {0, 3, 2, 7},
      {0, 2, 6, 7},
      {0, 6, 4, 7},
      {0, 4, 5, 7},
      {0, 5, 1, 7},
  }};
  SurfaceMesh3D mesh;
  std::vector<Vec3> field_normals;
  std::map<std::pair<std::size_t, std::size_t>, std::uint32_t> edges;
  for (std::size_t x = 0; x + 1 < dimensions[0]; ++x) {
    for (std::size_t y = 0; y + 1 < dimensions[1]; ++y) {
      for (std::size_t z = 0; z + 1 < dimensions[2]; ++z) {
        std::array<std::size_t, 8> nodes;
        std::array<double, 8> values;
        std::array<Vec3, 8> positions;
        std::size_t liquid_count = 0;
        for (std::size_t corner = 0; corner < corners.size(); ++corner) {
          const auto offset = corners[corner];
          nodes[corner] = node_id(x + offset[0], y + offset[1], z + offset[2]);
          values[corner] = lattice.values[nodes[corner]];
          // Roundoff on either side of a zero node must use the same tie rule;
          // otherwise adjacent edges create numerically coincident vertices.
          if (std::abs(values[corner]) <= lattice.spacing * 1e-8) {
            values[corner] = lattice.spacing * 1e-8;
          }
          liquid_count += values[corner] < 0;
          positions[corner] = {
              lattice.origin.x + (x + offset[0]) * lattice.spacing,
              lattice.origin.y + (y + offset[1]) * lattice.spacing,
              lattice.origin.z + (z + offset[2]) * lattice.spacing,
          };
        }
        if (liquid_count == 0 || liquid_count == 8) {
          continue;
        }
        const auto crossing = [&](int a, int b) -> absl::StatusOr<std::uint32_t> {
          const double fraction = values[a] / (values[a] - values[b]);
          // Share a lattice-node identity when a crossing lies at the symbolic
          // tie. Keeping separate edge vertices creates arbitrarily short
          // edges whose unrelated normals cannot safely carry an offset.
          const int snapped_node = fraction <= 1e-7 ? a : (fraction >= 1 - 1e-7 ? b : -1);
          const std::pair<std::size_t, std::size_t> key = snapped_node >= 0
              ? std::pair{nodes[snapped_node], nodes[snapped_node]}
              : std::pair{std::min(nodes[a], nodes[b]), std::max(nodes[a], nodes[b])};
          const auto found = edges.find(key);
          if (found != edges.end()) {
            return found->second;
          }
          if (mesh.vertices.size() == maximum_vertices) {
            return absl::ResourceExhaustedError("surface vertex budget exceeded");
          }
          const Vec3 position = snapped_node >= 0 ? positions[snapped_node]
                                                  : Add(positions[a], Scale(Sub(positions[b], positions[a]), fraction));
          if (!Finite(position)) {
            return absl::OutOfRangeError("surface crossing is nonfinite");
          }
          const auto index = static_cast<std::uint32_t>(mesh.vertices.size());
          edges[key] = index;
          mesh.vertices.push_back({.id = mesh.next_id++, .position = position});
          const auto gradient_at = [&](int corner) {
            const std::array node{x + corners[corner][0], y + corners[corner][1], z + corners[corner][2]};
            std::array<double, 3> gradient{};
            for (std::size_t axis = 0; axis < 3; ++axis) {
              auto lower = node;
              auto upper = node;
              lower[axis] -= lower[axis] > 0;
              upper[axis] += upper[axis] + 1 < dimensions[axis];
              gradient[axis] = (lattice.values[node_id(upper[0], upper[1], upper[2])]
                                - lattice.values[node_id(lower[0], lower[1], lower[2])])
                  / ((upper[axis] - lower[axis]) * lattice.spacing);
            }
            return Vec3{gradient[0], gradient[1], gradient[2]};
          };
          const auto first_gradient = gradient_at(a);
          const auto last_gradient = gradient_at(b);
          const auto gradient = snapped_node >= 0
              ? gradient_at(snapped_node)
              : Add(first_gradient, Scale(Sub(last_gradient, first_gradient), fraction));
          const double gradient_length = Norm(gradient);
          field_normals.push_back(
              gradient_length > 0 && std::isfinite(gradient_length) ? Scale(gradient, 1 / gradient_length) : Vec3{}
          );
          return index;
        };
        for (const auto& tetrahedron : tetrahedra) {
          std::vector<int> liquid;
          std::vector<int> air;
          for (const int node : tetrahedron) {
            (values[node] < 0 ? liquid : air).push_back(node);
          }
          if (liquid.empty() || air.empty()) {
            continue;
          }
          Vec3 outward{};
          for (const int node : air) {
            outward = Add(outward, Scale(positions[node], 1.0 / air.size()));
          }
          for (const int node : liquid) {
            outward = Sub(outward, Scale(positions[node], 1.0 / liquid.size()));
          }
          std::vector<std::uint32_t> polygon;
          if (liquid.size() == 2) {
            for (const auto& pair :
                 {std::pair{liquid[0], air[0]},
                  std::pair{liquid[0], air[1]},
                  std::pair{liquid[1], air[1]},
                  std::pair{liquid[1], air[0]}}) {
              const auto index = crossing(pair.first, pair.second);
              if (!index.ok()) {
                return index.status();
              }
              polygon.push_back(*index);
            }
          } else {
            for (const int a : liquid) {
              for (const int b : air) {
                const auto index = crossing(a, b);
                if (!index.ok()) {
                  return index.status();
                }
                polygon.push_back(*index);
              }
            }
          }
          for (std::size_t fan = 1; fan + 1 < polygon.size(); ++fan) {
            std::array triangle{polygon[0], polygon[fan], polygon[fan + 1]};
            if (triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0]) {
              continue;
            }
            const auto normal = Cross(
                Sub(mesh.vertices[triangle[1]].position, mesh.vertices[triangle[0]].position),
                Sub(mesh.vertices[triangle[2]].position, mesh.vertices[triangle[0]].position)
            );
            if (Dot(normal, outward) < 0) {
              std::swap(triangle[1], triangle[2]);
            }
            mesh.triangles.push_back(triangle);
          }
        }
      }
    }
  }
  UpdateNormals(mesh);
  for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
    if (Dot(mesh.vertices[index].normal, field_normals[index]) > 0) {
      mesh.vertices[index].normal = field_normals[index];
    }
  }
  return mesh;
}

absl::StatusOr<SurfaceMetrics3D> MeasureSurfaceMesh3D(const SurfaceMesh3D& mesh) {
  return MeasureSurfaceMesh3D(mesh, nullptr);
}
absl::StatusOr<SurfaceMetrics3D>
MeasureSurfaceMesh3D(const SurfaceMesh3D& mesh, std::vector<std::array<std::size_t, 2>>* crossings) {
  SurfaceMetrics3D metrics;
  DisjointSet sets(mesh.vertices.size());
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::pair<std::size_t, int>> edges;
  std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> links(mesh.vertices.size());
  std::vector<std::array<Vec3, 3>> positions;
  const Vec3 origin = mesh.vertices.empty() ? Vec3{} : mesh.vertices.front().position;
  for (const auto& vertex : mesh.vertices) {
    if (!Finite(vertex.position)) {
      return absl::InvalidArgumentError("surface vertex is nonfinite");
    }
  }
  for (const auto triangle : mesh.triangles) {
    if (*std::max_element(triangle.begin(), triangle.end()) >= mesh.vertices.size()) {
      return absl::InvalidArgumentError("surface triangle index is outside vertices");
    }
    const std::array p{
        mesh.vertices[triangle[0]].position,
        mesh.vertices[triangle[1]].position,
        mesh.vertices[triangle[2]].position
    };
    positions.push_back(p);
    const double twice_area = Norm(Cross(Sub(p[1], p[0]), Sub(p[2], p[0])));
    metrics.degenerate_triangles += twice_area == 0;
    metrics.area += 0.5 * twice_area;
    metrics.signed_volume += Dot(Sub(p[0], origin), Cross(Sub(p[1], origin), Sub(p[2], origin))) / 6;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto a = triangle[i];
      const auto b = triangle[(i + 1) % 3];
      auto& edge = edges[std::minmax(a, b)];
      ++edge.first;
      edge.second += a < b ? 1 : -1;
      sets.Join(a, b);
      links[a].emplace_back(b, triangle[(i + 2) % 3]);
    }
  }
  std::set<std::size_t> roots;
  for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
    roots.insert(sets.Find(index));
    std::map<std::uint32_t, std::vector<std::uint32_t>> link;
    for (const auto& [a, b] : links[index]) {
      link[a].push_back(b);
      link[b].push_back(a);
    }
    std::set<std::uint32_t> visited;
    std::vector<std::uint32_t> pending;
    if (!link.empty()) {
      pending.push_back(link.begin()->first);
    }
    while (!pending.empty()) {
      const auto node = pending.back();
      pending.pop_back();
      if (!visited.insert(node).second) {
        continue;
      }
      for (const auto neighbor : link[node]) {
        pending.push_back(neighbor);
      }
    }
    const bool cycle =
        !link.empty() && visited.size() == link.size() && std::all_of(link.begin(), link.end(), [](const auto& node) {
          return node.second.size() == 2;
        });
    metrics.nonmanifold_vertices += !cycle;
  }
  metrics.components = roots.size();
  // Use each component's own origin for orientation, including an inner
  // cavity far from the outer boundary's vertices.
  std::vector<double> component_volumes(mesh.vertices.size(), 0.0);
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    const auto root = sets.Find(mesh.triangles[index][0]);
    const auto local_origin = mesh.vertices[root].position;
    const auto& p = positions[index];
    component_volumes[root] +=
        Dot(Sub(p[0], local_origin), Cross(Sub(p[1], local_origin), Sub(p[2], local_origin))) / 6.0;
  }
  for (const auto root : roots) {
    const double volume = component_volumes[root];
    if (!std::isfinite(volume)) {
      return absl::OutOfRangeError("surface component volume overflow");
    }
    metrics.positive_volume_components += volume > 0.0;
    metrics.negative_volume_components += volume < 0.0;
  }
  metrics.euler_characteristic = static_cast<std::int64_t>(mesh.vertices.size())
      - static_cast<std::int64_t>(edges.size()) + static_cast<std::int64_t>(mesh.triangles.size());
  for (const auto& [key, edge] : edges) {
    metrics.boundary_edges += edge.first == 1;
    metrics.nonmanifold_edges += edge.first > 2;
    metrics.orientation_defects += edge.first == 2 && edge.second != 0;
  }
  std::vector<std::size_t> order(positions.size());
  std::iota(order.begin(), order.end(), 0);
  const auto minimum_x = [&](std::size_t i) {
    return std::min({positions[i][0].x, positions[i][1].x, positions[i][2].x});
  };
  std::sort(order.begin(), order.end(), [&](auto a, auto b) {
    return std::tuple(minimum_x(a), a) < std::tuple(minimum_x(b), b);
  });
  for (std::size_t i = 0; i < order.size(); ++i) {
    const auto a = order[i];
    const auto& p = positions[a];
    const double maximum_x = std::max({p[0].x, p[1].x, p[2].x});
    for (std::size_t j = i + 1; j < order.size() && minimum_x(order[j]) <= maximum_x; ++j) {
      const auto b = order[j];
      std::array<bool, 3> shared_a{};
      std::array<bool, 3> shared_b{};
      for (std::size_t va = 0; va < 3; ++va) {
        for (std::size_t vb = 0; vb < 3; ++vb) {
          if (mesh.triangles[a][va] == mesh.triangles[b][vb]) {
            shared_a[va] = true;
            shared_b[vb] = true;
          }
        }
      }
      const auto& q = positions[b];
      if (std::max({p[0].y, p[1].y, p[2].y}) < std::min({q[0].y, q[1].y, q[2].y})
          || std::max({q[0].y, q[1].y, q[2].y}) < std::min({p[0].y, p[1].y, p[2].y})
          || std::max({p[0].z, p[1].z, p[2].z}) < std::min({q[0].z, q[1].z, q[2].z})
          || std::max({q[0].z, q[1].z, q[2].z}) < std::min({p[0].z, p[1].z, p[2].z})) {
        continue;
      }
      if (Intersects(p, q, shared_a, shared_b)) {
        ++metrics.self_intersections;
        if (crossings && crossings->size() < 256) {
          crossings->push_back({a, b});
        }
      }
    }
  }
  if (!std::isfinite(metrics.signed_volume) || !std::isfinite(metrics.area)) {
    return absl::OutOfRangeError("surface measurement overflowed");
  }
  return metrics;
}

namespace {
absl::StatusOr<SurfaceMetrics3D> CorrectVolume(
    SurfaceMesh3D& mesh,
    double target,
    double width,
    double tolerance,
    std::size_t maximum_iterations,
    bool use_stored_normals
) {
  if (!std::isfinite(target) || target <= 0 || !std::isfinite(width) || width <= 0 || !std::isfinite(tolerance)
      || tolerance <= 0 || tolerance > 0.005 || maximum_iterations == 0) {
    return absl::InvalidArgumentError("invalid surface volume correction controls");
  }
  SurfaceMesh3D corrected = mesh;
  auto measured = MeasureSurfaceMesh3D(corrected);
  if (!measured.ok()) {
    return measured.status();
  }
  if (!Safe(*measured)) {
    return absl::FailedPreconditionError("volume correction requires a safe closed surface");
  }
  if (use_stored_normals) {
    auto geometric = mesh;
    UpdateNormals(geometric);
    for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
      const auto normal = mesh.vertices[index].normal;
      if (!Finite(normal) || std::abs(Norm(normal) - 1) > 1e-8 || Dot(normal, geometric.vertices[index].normal) <= 0) {
        return absl::InvalidArgumentError("volume correction needs unit outward stored normals");
      }
    }
  }
  for (std::size_t iteration = 0; iteration <= maximum_iterations; ++iteration) {
    if (std::abs(measured->signed_volume - target) <= tolerance * target) {
      UpdateNormals(corrected);
      mesh = std::move(corrected);
      return *measured;
    }
    if (iteration == maximum_iterations) {
      break;
    }
    if (!use_stored_normals) {
      UpdateNormals(corrected);
    }
    double mobile_area = 0;
    for (const auto triangle : corrected.triangles) {
      const auto area = 0.5
          * Norm(Cross(
              Sub(corrected.vertices[triangle[1]].position, corrected.vertices[triangle[0]].position),
              Sub(corrected.vertices[triangle[2]].position, corrected.vertices[triangle[0]].position)
          ));
      for (const auto index : triangle) {
        if (!corrected.vertices[index].contact && !corrected.vertices[index].unresolved_sheet) {
          mobile_area += area / 3;
        }
      }
    }
    if (mobile_area <= 0) {
      break;
    }
    double offset = std::clamp((target - measured->signed_volume) / mobile_area, -0.25 * width, 0.25 * width);
    bool accepted = false;
    for (int backtrack = 0; backtrack < 16; ++backtrack) {
      auto candidate = corrected;
      bool inside_bound = true;
      for (std::size_t index = 0; index < candidate.vertices.size(); ++index) {
        auto& vertex = candidate.vertices[index];
        if (!vertex.contact && !vertex.unresolved_sheet) {
          vertex.position = Add(vertex.position, Scale(vertex.normal, offset));
          inside_bound = inside_bound && Norm(Sub(vertex.position, mesh.vertices[index].position)) <= 0.25 * width;
        }
      }
      if (!inside_bound) {
        offset *= 0.5;
        continue;
      }
      auto metrics = MeasureSurfaceMesh3D(candidate);
      if (metrics.ok() && Safe(*metrics)
          && std::abs(metrics->signed_volume - target) < std::abs(measured->signed_volume - target)) {
        corrected = std::move(candidate);
        measured = *metrics;
        accepted = true;
        break;
      }
      offset *= 0.5;
    }
    if (!accepted) {
      break;
    }
  }
  return absl::FailedPreconditionError("surface volume constraint is infeasible within the correction budget");
}
}  // namespace

absl::StatusOr<SurfaceMetrics3D> CorrectSurfaceVolume3D(
    SurfaceMesh3D& mesh,
    double target,
    double width,
    double tolerance,
    std::size_t maximum_iterations
) {
  return CorrectVolume(mesh, target, width, tolerance, maximum_iterations, false);
}

absl::StatusOr<SurfaceMetrics3D> CorrectSurfaceVolumeWithNormals3D(
    SurfaceMesh3D& mesh,
    double target,
    double width,
    double tolerance,
    std::size_t maximum_iterations
) {
  return CorrectVolume(mesh, target, width, tolerance, maximum_iterations, true);
}
}  // namespace fluidity
