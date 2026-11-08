cmake_minimum_required(VERSION 3.31)
project(FluidiTyPointCacheWriter LANGUAGES CXX)

find_package(Alembic 1.8.8 EXACT CONFIG REQUIRED)

add_executable(fluidity_write_point_cache write_point_cache.cc)
target_compile_features(fluidity_write_point_cache PRIVATE cxx_std_20)
set_target_properties(fluidity_write_point_cache PROPERTIES CXX_EXTENSIONS OFF CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(fluidity_write_point_cache PRIVATE Alembic::Alembic)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME fluidity_point_cache_readback
    COMMAND fluidity_write_point_cache "${CMAKE_CURRENT_BINARY_DIR}/FluidiTyPoints.abc")
endif()
