fluidity_add_headers(
  axis.h
  cpu_and_gpu.h
  mat3.h
  scalar_math.h
  vec3.h
)

fluidity_add_test(vec3_test SOURCES vec3_test.cc LABEL cpu)
fluidity_add_test(mat3_test SOURCES mat3_test.cc LABEL cpu)
fluidity_add_test(scalar_math_test SOURCES scalar_math_test.cc LABEL cpu)
fluidity_add_test(axis_test SOURCES axis_test.cc LABEL cpu)
