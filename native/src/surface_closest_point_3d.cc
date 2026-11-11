#include "fluidity/host/surface_closest_point_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#include "surface_predicates_3d_internal.h"

namespace fluidity {
namespace {
bool Finite(Vec3 p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
Vec3 Sub(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 Add(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 Scale(Vec3 a, double scale) {
  return {a.x * scale, a.y * scale, a.z * scale};
}
double Dot(Vec3 a, Vec3 b) {
  return (a.x * b.x + a.y * b.y) + a.z * b.z;
}
double Norm(Vec3 a) {
  return std::hypot(a.x, a.y, a.z);
}
Vec3 Minimum(Vec3 a, Vec3 b) {
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
Vec3 Maximum(Vec3 a, Vec3 b) {
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}
double BoxDistance(Vec3 p, Vec3 lower, Vec3 upper) {
  const double x = std::max({lower.x - p.x, 0.0, p.x - upper.x});
  const double y = std::max({lower.y - p.y, 0.0, p.y - upper.y});
  const double z = std::max({lower.z - p.z, 0.0, p.z - upper.z});
  // Conservative pruning allowance for bounded binary64 distance arithmetic.
  return std::hypot(x, y, z) * (1 - 32 * std::numeric_limits<double>::epsilon());
}
double Area2D(Vec3 a, Vec3 b, Vec3 c, int drop) {
  const std::array av{a.x, a.y, a.z};
  const std::array bv{b.x, b.y, b.z};
  const std::array cv{c.x, c.y, c.z};
  const int x = (drop + 1) % 3;
  const int y = (drop + 2) % 3;
  const double left = (bv[x] - av[x]) * (cv[y] - av[y]);
  const double right = (bv[y] - av[y]) * (cv[x] - av[x]);
  const double approximate = left - right;
  if (std::abs(approximate) > 1e-14 * (std::abs(left) + std::abs(right))) {
    return approximate;
  }
  auto exact =
      surface_internal::Product(surface_internal::Difference(bv[x], av[x]), surface_internal::Difference(cv[y], av[y]));
  surface_internal::Add(
      exact,
      surface_internal::Product(surface_internal::Difference(bv[y], av[y]), surface_internal::Difference(cv[x], av[x])),
      -1
  );
  return std::accumulate(exact.begin(), exact.end(), 0.0);
}
Vec3 ClosestOnSegment(Vec3 p, Vec3 a, Vec3 b) {
  const auto edge = Sub(b, a);
  const double fraction = std::clamp(Dot(Sub(p, a), edge) / Dot(edge, edge), 0.0, 1.0);
  if (fraction == 0) {
    return a;
  }
  if (fraction == 1) {
    return b;
  }
  return Add(a, Scale(edge, fraction));
}
// Tests the three edge minima and the plane foot. Exact-sign area fallback
// avoids a cancelled Gram determinant on skinny but valid triangles.
Vec3 ClosestOnTriangle(Vec3 p, const std::array<Vec3, 3>& triangle, Vec3 normal) {
  Vec3 closest = triangle[0];
  double best = Norm(Sub(p, closest));
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const auto candidate = ClosestOnSegment(p, triangle[edge], triangle[(edge + 1) % 3]);
    const double distance = Norm(Sub(p, candidate));
    if (distance < best) {
      closest = candidate;
      best = distance;
    }
  }
  if (best == 0) {
    return closest;
  }
  const auto projection = Sub(p, Scale(normal, Dot(Sub(p, triangle[0]), normal)));
  const int drop = std::abs(normal.x) > std::abs(normal.y) ? (std::abs(normal.x) > std::abs(normal.z) ? 0 : 2)
                                                           : (std::abs(normal.y) > std::abs(normal.z) ? 1 : 2);
  const double denominator = Area2D(triangle[0], triangle[1], triangle[2], drop);
  const double a = Area2D(triangle[1], triangle[2], projection, drop);
  const double b = Area2D(triangle[2], triangle[0], projection, drop);
  const double c = Area2D(triangle[0], triangle[1], projection, drop);
  if (denominator != 0
      && ((denominator > 0 && a >= 0 && b >= 0 && c >= 0) || (denominator < 0 && a <= 0 && b <= 0 && c <= 0))) {
    const auto candidate =
        Add(triangle[0],
            Add(Scale(Sub(triangle[1], triangle[0]), b / denominator),
                Scale(Sub(triangle[2], triangle[0]), c / denominator)));
    if (Norm(Sub(p, candidate)) < best) {
      closest = candidate;
    }
  }
  return closest;
}
}  // namespace

absl::StatusOr<SurfaceClosestPoint3D>
SurfaceClosestPoint3D::Create(const SurfaceMesh3D& mesh, std::size_t maximum_triangles) {
  const auto budget = std::min(maximum_triangles, std::size_t{1} << 20);
  if (mesh.triangles.size() > budget || mesh.vertices.size() > 3 * budget) {
    return absl::ResourceExhaustedError("surface closest-point triangle budget exceeded");
  }
  const auto metrics = MeasureSurfaceMesh3D(mesh);
  if (!metrics.ok()) {
    return metrics.status();
  }
  if (mesh.triangles.empty() || metrics->signed_volume <= 0 || metrics->boundary_edges != 0
      || metrics->nonmanifold_edges != 0 || metrics->nonmanifold_vertices != 0 || metrics->orientation_defects != 0
      || metrics->degenerate_triangles != 0 || metrics->self_intersections != 0) {
    return absl::FailedPreconditionError("closest-point scaffold must be safe closed outward geometry");
  }
  SurfaceClosestPoint3D result;
  for (const auto triangle : mesh.triangles) {
    const auto& first = mesh.vertices[triangle[0]];
    if (first.component == 0 || mesh.vertices[triangle[1]].component != first.component
        || mesh.vertices[triangle[2]].component != first.component) {
      return absl::InvalidArgumentError("closest-point triangle has inconsistent material labels");
    }
    const std::array vertices{first.position, mesh.vertices[triangle[1]].position, mesh.vertices[triangle[2]].position};
    for (std::size_t edge = 0; edge < 3; ++edge) {
      const auto delta = Sub(vertices[(edge + 1) % 3], vertices[edge]);
      for (const double value : {delta.x, delta.y, delta.z}) {
        if (std::abs(value) > 1e50 || (value != 0 && std::abs(value) < 1e-50)) {
          return absl::OutOfRangeError("closest-point triangle exceeds numerical scale limits");
        }
      }
    }
    const auto a = Sub(vertices[1], vertices[0]);
    const auto b = Sub(vertices[2], vertices[0]);
    const Vec3 normal{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    result.faces_.push_back({
        .vertices = vertices,
        .normal = Scale(normal, 1 / Norm(normal)),
        .lower = Minimum(vertices[0], Minimum(vertices[1], vertices[2])),
        .upper = Maximum(vertices[0], Maximum(vertices[1], vertices[2])),
        .component = first.component,
        .index = static_cast<std::uint32_t>(result.faces_.size()),
    });
  }
  result.order_.resize(result.faces_.size());
  std::iota(result.order_.begin(), result.order_.end(), 0);
  result.nodes_.reserve(2 * result.faces_.size());
  result.Build(0, static_cast<std::uint32_t>(result.faces_.size()));
  return result;
}

std::uint32_t SurfaceClosestPoint3D::Build(std::uint32_t begin, std::uint32_t end) {
  const auto index = static_cast<std::uint32_t>(nodes_.size());
  Node node{.lower = faces_[order_[begin]].lower, .upper = faces_[order_[begin]].upper};
  for (auto offset = begin + 1; offset < end; ++offset) {
    node.lower = Minimum(node.lower, faces_[order_[offset]].lower);
    node.upper = Maximum(node.upper, faces_[order_[offset]].upper);
  }
  nodes_.push_back(node);
  if (end - begin <= 8) {
    nodes_[index].begin = begin;
    nodes_[index].count = end - begin;
    return index;
  }
  const auto span = Sub(node.upper, node.lower);
  const int axis = span.x >= span.y && span.x >= span.z ? 0 : (span.y >= span.z ? 1 : 2);
  const auto center = [&](std::uint32_t face) {
    const auto lower = faces_[face].lower;
    const auto upper = faces_[face].upper;
    const std::array values{
        lower.x + 0.5 * (upper.x - lower.x),
        lower.y + 0.5 * (upper.y - lower.y),
        lower.z + 0.5 * (upper.z - lower.z)
    };
    return values[axis];
  };
  const auto middle = begin + (end - begin) / 2;
  std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end, [&](auto a, auto b) {
    return std::tuple{center(a), a} < std::tuple{center(b), b};
  });
  const auto left = Build(begin, middle);
  const auto right = Build(middle, end);
  nodes_[index].left = left;
  nodes_[index].right = right;
  return index;
}

absl::StatusOr<SurfaceClosestSample3D> SurfaceClosestPoint3D::Sample(Vec3 position, std::uint64_t component) const {
  if (!Finite(position) || component == 0) {
    return absl::InvalidArgumentError("invalid closest-point query or component");
  }
  if (nodes_.empty()) {
    return absl::FailedPreconditionError("closest-point scaffold is empty");
  }
  if (Norm(Sub(position, nodes_[0].lower)) > 1e50) {
    return absl::OutOfRangeError("closest-point query exceeds numerical scale limits");
  }
  std::array<std::uint32_t, 64> pending{};
  std::size_t count = 1;
  double best = std::numeric_limits<double>::infinity();
  SurfaceClosestSample3D result{};
  bool found = false;
  while (count > 0) {
    const auto& node = nodes_[pending[--count]];
    if (BoxDistance(position, node.lower, node.upper) > best) {
      continue;
    }
    if (node.count != 0) {
      for (std::uint32_t offset = node.begin; offset < node.begin + node.count; ++offset) {
        const auto& face = faces_[order_[offset]];
        if (face.component != component) {
          continue;
        }
        const auto closest = ClosestOnTriangle(position, face.vertices, face.normal);
        const double distance = Norm(Sub(position, closest));
        if (!found || std::tuple{distance, face.index} < std::tuple{best, result.triangle}) {
          result = {
              .position = closest,
              .outward_normal = face.normal,
              .distance_m = distance,
              .component = component,
              .triangle = face.index
          };
          best = distance;
          found = true;
        }
      }
    } else {
      if (count + 2 > pending.size()) {
        return absl::ResourceExhaustedError("closest-point traversal stack exceeded");
      }
      const auto& left = nodes_[node.left];
      const auto& right = nodes_[node.right];
      const bool left_first =
          BoxDistance(position, left.lower, left.upper) <= BoxDistance(position, right.lower, right.upper);
      pending[count++] = left_first ? node.right : node.left;
      pending[count++] = left_first ? node.left : node.right;
    }
  }
  if (!found) {
    return absl::FailedPreconditionError("closest-point scaffold has no requested material component");
  }
  if (!std::isfinite(result.distance_m) || !Finite(result.position)) {
    return absl::OutOfRangeError("closest-point arithmetic overflowed");
  }
  return result;
}
}  // namespace fluidity
