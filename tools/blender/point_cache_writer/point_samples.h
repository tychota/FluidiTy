#pragma once

#include <Alembic/AbcGeom/All.h>

#include <cstddef>
#include <cstdint>

namespace fluidity::blender {
// Convert Blender Z-up meters or meters per second to Alembic Y-up.
inline Alembic::Abc::V3f ToAlembicYUp(const Alembic::Abc::V3f& blender_vector) {
  return {blender_vector.x, blender_vector.z, -blender_vector.y};
}

// All pointers have at least count elements. Even with count zero, pass nonnull pointers
// so Alembic writes an explicit empty sample rather than repeating the preceding sample.
inline void WritePointSample(
    Alembic::AbcGeom::OPointsSchema& schema,
    const Alembic::Abc::V3f* positions,
    const std::uint64_t* ids,
    const Alembic::Abc::V3f* velocities,
    const float* diameters,
    std::size_t count
) {
  namespace abc = Alembic::Abc;
  namespace geom = Alembic::AbcGeom;
  const geom::OFloatGeomParam::Sample widths(abc::FloatArraySample(diameters, count), geom::kVaryingScope);
  schema.set(geom::OPointsSchema::Sample(
      abc::P3fArraySample(positions, count),
      abc::UInt64ArraySample(ids, count),
      abc::V3fArraySample(velocities, count),
      widths
  ));
}
}  // namespace fluidity::blender
