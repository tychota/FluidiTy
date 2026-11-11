#pragma once

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "fluidity/base/mac_types_3d.h"

namespace fluidity::surface_internal {

// Nonoverlapping expansions preserve the exact sign of binary64 determinants.
// std::fma supplies the product residual; no geometric tolerance turns a gap
// between skinny triangles into a contact. Inputs must stay in the normal
// product range (the checked mesh API rejects numerical overflow).
using Expansion = std::vector<double>;

inline void Accumulate(Expansion& expansion, double value) {
  Expansion result;
  result.reserve(expansion.size() + 1);
  for (const double term : expansion) {
    const double sum = value + term;
    const double virtual_term = sum - value;
    const double residual = (value - (sum - virtual_term)) + (term - virtual_term);
    if (residual != 0) {
      result.push_back(residual);
    }
    value = sum;
  }
  if (value != 0 || result.empty()) {
    result.push_back(value);
  }
  expansion = std::move(result);
}

inline Expansion Difference(double a, double b) {
  Expansion result{a};
  Accumulate(result, -b);
  return result;
}

inline Expansion Product(const Expansion& a, const Expansion& b) {
  Expansion result;
  for (const double x : a) {
    for (const double y : b) {
      const double product = x * y;
      Accumulate(result, std::fma(x, y, -product));
      Accumulate(result, product);
    }
  }
  return result;
}

inline void Add(Expansion& a, const Expansion& b, double sign = 1) {
  for (const double term : b) {
    Accumulate(a, sign * term);
  }
}

inline int Sign(const Expansion& expansion) {
  const double value = expansion.empty() ? 0 : expansion.back();
  return (value > 0) - (value < 0);
}

inline int Orient3D(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
  const std::array ad{a.x - d.x, a.y - d.y, a.z - d.z};
  const std::array bd{b.x - d.x, b.y - d.y, b.z - d.z};
  const std::array cd{c.x - d.x, c.y - d.y, c.z - d.z};
  double approximate = 0;
  double permanent = 0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const auto next = (axis + 1) % 3;
    const auto last = (axis + 2) % 3;
    approximate += ad[axis] * (bd[next] * cd[last] - bd[last] * cd[next]);
    permanent += std::abs(ad[axis]) * (std::abs(bd[next] * cd[last]) + std::abs(bd[last] * cd[next]));
  }
  // Conservative binary64 roundoff filter. Uncertain signs use expansions.
  if (std::isfinite(permanent) && std::abs(approximate) > 1e-14 * permanent) {
    return (approximate > 0) - (approximate < 0);
  }
  const std::array left{Difference(a.x, d.x), Difference(a.y, d.y), Difference(a.z, d.z)};
  const std::array middle{Difference(b.x, d.x), Difference(b.y, d.y), Difference(b.z, d.z)};
  const std::array right{Difference(c.x, d.x), Difference(c.y, d.y), Difference(c.z, d.z)};
  Expansion determinant;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const auto next = (axis + 1) % 3;
    const auto last = (axis + 2) % 3;
    auto minor = Product(middle[next], right[last]);
    Add(minor, Product(middle[last], right[next]), -1);
    Add(determinant, Product(left[axis], minor));
  }
  return Sign(determinant);
}

inline int Orient2D(Vec3 a, Vec3 b, Vec3 c, int drop) {
  const std::array av{a.x, a.y, a.z};
  const std::array bv{b.x, b.y, b.z};
  const std::array cv{c.x, c.y, c.z};
  const int x = (drop + 1) % 3;
  const int y = (drop + 2) % 3;
  const double left = (av[x] - cv[x]) * (bv[y] - cv[y]);
  const double right = (av[y] - cv[y]) * (bv[x] - cv[x]);
  const double approximate = left - right;
  if (std::isfinite(left) && std::isfinite(right)
      && std::abs(approximate) > 1e-14 * (std::abs(left) + std::abs(right))) {
    return (approximate > 0) - (approximate < 0);
  }
  auto determinant = Product(Difference(av[x], cv[x]), Difference(bv[y], cv[y]));
  Add(determinant, Product(Difference(av[y], cv[y]), Difference(bv[x], cv[x])), -1);
  return Sign(determinant);
}

inline bool SameSide(int a, int b, int c) {
  return (a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0);
}

inline bool SegmentTriangle(Vec3 a, Vec3 b, const std::array<Vec3, 3>& triangle) {
  const int first = Orient3D(triangle[0], triangle[1], triangle[2], a);
  const int last = Orient3D(triangle[0], triangle[1], triangle[2], b);
  if ((first == 0 && last == 0) || first * last > 0) {
    return false;
  }
  return SameSide(
      Orient3D(a, b, triangle[0], triangle[1]),
      Orient3D(a, b, triangle[1], triangle[2]),
      Orient3D(a, b, triangle[2], triangle[0])
  );
}
}  // namespace fluidity::surface_internal
