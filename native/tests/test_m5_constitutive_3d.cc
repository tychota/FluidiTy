#include <cstdio>

#include "m5_contact_3d_cases.h"

int main() {
  int result = fluidity::m5_test::CheckConstitutiveCases();
  if (result == 0) {
    result = fluidity::m5_test::CheckContactCases();
  }
  if (result != 0) {
    std::fprintf(stderr, "M5 invariant %d failed\n", result);
  }
  return result;
}
