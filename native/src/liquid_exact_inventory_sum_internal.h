#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace fluidity::liquid_exchange_internal {

// Ordered FastTwoSum retains each residual in a nonoverlapping expansion.
// The largest nonzero partial determines the sign of the exact binary64 sum.
template <std::size_t MaximumTerms>
class ExactInventorySum {
 public:
  bool Add(double term) {
    std::size_t retained = 0;
    for (std::size_t ordinal = 0; ordinal < count_; ++ordinal) {
      double partial = partials_[ordinal];
      if (std::abs(term) < std::abs(partial)) {
        std::swap(term, partial);
      }
      const double sum = term + partial;
      if (!std::isfinite(sum)) {
        return false;
      }
      const double remainder = partial - (sum - term);
      if (remainder != 0.0) {
        partials_[retained++] = remainder;
      }
      term = sum;
    }
    if (retained == partials_.size()) {
      return false;
    }
    partials_[retained] = term;
    count_ = retained + 1;
    return true;
  }

  int sign() const {
    for (std::size_t ordinal = count_; ordinal > 0; --ordinal) {
      const double partial = partials_[ordinal - 1];
      if (partial != 0.0) {
        return partial < 0.0 ? -1 : 1;
      }
    }
    return 0;
  }

 private:
  std::array<double, MaximumTerms> partials_{};
  std::size_t count_ = 0;
};

}  // namespace fluidity::liquid_exchange_internal
