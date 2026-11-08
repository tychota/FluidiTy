#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/IPoints.h>
#include <Alembic/AbcGeom/OPoints.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace {

namespace abc = Alembic::Abc;
namespace geom = Alembic::AbcGeom;
namespace ogawa = Alembic::AbcCoreOgawa;

// Read the carrier from disk and compare it to the fixture in Alembic coordinates.
bool VerifyPointCacheArchive(const std::string& archive_path) {
  const abc::IArchive archive(ogawa::ReadArchive(), archive_path);
  const abc::IObject archive_root = archive.getTop();
  if (archive_root.getNumChildren() != 1) {
    std::cerr << "Archive must contain exactly one Points object\n";
    return false;
  }

  const abc::ObjectHeader& point_cache_object_header = archive_root.getChildHeader(0);
  if (point_cache_object_header.getName() != "FluidiTyPoints" || !geom::IPoints::matches(point_cache_object_header)) {
    std::cerr << "Expected a Points object named FluidiTyPoints\n";
    return false;
  }

  const geom::IPoints point_cache_object(archive_root, point_cache_object_header.getName());
  if (point_cache_object.getNumChildren() != 0) {
    std::cerr << "FluidiTyPoints must be a leaf object\n";
    return false;
  }

  constexpr std::array<std::size_t, 4> expected_point_counts = {3, 5, 0, 4};
  const geom::IPointsSchema& point_cache_schema = point_cache_object.getSchema();
  const auto positions_property = point_cache_schema.getPositionsProperty();
  const auto ids_property = point_cache_schema.getIdsProperty();
  const auto velocities_property = point_cache_schema.getVelocitiesProperty();
  const auto point_width_parameter = point_cache_schema.getWidthsParam();
  if (!velocities_property.valid() || !point_width_parameter.valid()) {
    std::cerr << "FluidiTyPoints must contain velocities and widths\n";
    return false;
  }
  if (point_width_parameter.getScope() != geom::kVaryingScope || point_width_parameter.getArrayExtent() != 1) {
    std::cerr << "Expected one varying width per point\n";
    return false;
  }

  const std::array property_sample_counts = {
      positions_property.getNumSamples(),
      ids_property.getNumSamples(),
      velocities_property.getNumSamples(),
      point_width_parameter.getNumSamples(),
  };
  for (const std::size_t sample_count : property_sample_counts) {
    if (sample_count != expected_point_counts.size()) {
      std::cerr << "Expected four samples in each point attribute\n";
      return false;
    }
  }

  const Alembic::AbcCoreAbstract::TimeSampling expected_sampling(1.0 / 24.0, 1.0 / 24.0);
  const std::array property_time_samplings = {
      positions_property.getTimeSampling(),
      ids_property.getTimeSampling(),
      velocities_property.getTimeSampling(),
      point_width_parameter.getTimeSampling(),
  };
  for (const auto& time_sampling : property_time_samplings) {
    if (*time_sampling != expected_sampling) {
      std::cerr << "Expected uniform 24 fps sampling starting at frame 1\n";
      return false;
    }
  }

  for (std::size_t sample_index = 0; sample_index < expected_point_counts.size(); ++sample_index) {
    const std::size_t timeline_frame = sample_index + 1;
    const auto alembic_sample_index = static_cast<Alembic::AbcCoreAbstract::index_t>(sample_index);
    const double sample_time_seconds = point_cache_schema.getTimeSampling()->getSampleTime(alembic_sample_index);
    const double expected_time_seconds = static_cast<double>(timeline_frame) / 24.0;
    const bool sample_time_matches = std::abs(sample_time_seconds - expected_time_seconds) <= 1e-12;
    if (!sample_time_matches) {
      std::cerr << "Unexpected sample time at frame " << timeline_frame << '\n';
      return false;
    }

    const abc::ISampleSelector alembic_sample_selector(alembic_sample_index);
    const auto alembic_point_sample = point_cache_schema.getValue(alembic_sample_selector);
    const auto sample_positions_y_up_m = alembic_point_sample.getPositions();
    const auto sample_point_ids = alembic_point_sample.getIds();
    const auto sample_velocities_y_up_m_per_s = alembic_point_sample.getVelocities();
    const auto sample_point_widths_m = point_width_parameter.getExpandedValue(alembic_sample_selector).getVals();
    const std::size_t point_count = expected_point_counts[sample_index];
    const bool point_attribute_counts_match = sample_positions_y_up_m && sample_point_ids
        && sample_velocities_y_up_m_per_s && sample_point_widths_m && sample_positions_y_up_m->size() == point_count
        && sample_point_ids->size() == point_count && sample_velocities_y_up_m_per_s->size() == point_count
        && sample_point_widths_m->size() == point_count;
    if (!point_attribute_counts_match) {
      std::cerr << "Expected " << point_count << " entries in each point attribute at frame " << timeline_frame << '\n';
      return false;
    }

    for (std::size_t point_index = 0; point_index < point_count; ++point_index) {
      const abc::V3f expected_position_y_up_m(
          static_cast<float>(timeline_frame),
          -static_cast<float>(timeline_frame + point_index),
          -static_cast<float>(point_index)
      );
      const abc::V3f expected_velocity_y_up_m_per_s(1.0f, 3.0f, 2.0f);
      // Fixture positions, velocities and diameters are exactly representable in binary32.
      const float expected_diameter_m = static_cast<float>(point_index + 1) / 8.0f;
      const bool point_attributes_match = (*sample_positions_y_up_m)[point_index] == expected_position_y_up_m
          && (*sample_velocities_y_up_m_per_s)[point_index] == expected_velocity_y_up_m_per_s
          && (*sample_point_widths_m)[point_index] == expected_diameter_m
          && (*sample_point_ids)[point_index] == 100 + point_index;
      if (!point_attributes_match) {
        std::cerr << "Unexpected point attributes at frame " << timeline_frame << ", point index " << point_index
                  << '\n';
        return false;
      }
    }
  }
  return true;
}

// A quarter turn around X converts Blender Z-up point_position_sample and vectors to Alembic Y-up.
abc::V3f ToAlembicYUp(const abc::V3f& blender_z_up_vector) {
  return {blender_z_up_vector.x, blender_z_up_vector.z, -blender_z_up_vector.y};
}

// Close the archive before the caller reopens it for readback.
void WritePointCacheArchive(const std::string& archive_path) {
  abc::OArchive archive(ogawa::WriteArchive(), archive_path);
  constexpr double sample_step_seconds = 1.0 / 24.0;
  const auto time_sampling_index = archive.addTimeSampling({sample_step_seconds, sample_step_seconds});
  geom::OPoints point_cache_object(archive.getTop(), "FluidiTyPoints", time_sampling_index);
  geom::OPointsSchema& point_cache_schema = point_cache_object.getSchema();

  constexpr std::array<std::size_t, 4> point_counts = {3, 5, 0, 4};
  constexpr std::size_t max_point_count = std::ranges::max(point_counts);
  const abc::V3f velocity_blender_m_per_s(1.0f, -2.0f, 3.0f);
  for (std::size_t sample_index = 0; sample_index < point_counts.size(); ++sample_index) {
    const std::size_t point_count = point_counts[sample_index];
    const float timeline_frame_x_m = static_cast<float>(sample_index + 1);
    std::array<abc::V3f, max_point_count> positions_alembic_m;
    std::array<abc::V3f, max_point_count> velocities_alembic_m_per_s;
    std::array<std::uint64_t, max_point_count> point_ids;
    std::array<float, max_point_count> diameters_m;
    for (std::size_t point_index = 0; point_index < point_count; ++point_index) {
      const float point_index_y_m = static_cast<float>(point_index);
      const abc::V3f position_blender_m(timeline_frame_x_m, point_index_y_m, -(timeline_frame_x_m + point_index_y_m));
      positions_alembic_m[point_index] = ToAlembicYUp(position_blender_m);
      velocities_alembic_m_per_s[point_index] = ToAlembicYUp(velocity_blender_m_per_s);
      point_ids[point_index] = 100 + point_index;
      const float display_radius_m = static_cast<float>(point_index + 1) / 16.0f;
      diameters_m[point_index] = 2.0f * display_radius_m;
    }

    // Alembic treats a null position pointer as a partial sample. At frame 3
    // these typed views have valid backing pointers and all four lengths are zero.
    const abc::P3fArraySample point_position_sample(positions_alembic_m.data(), point_count);
    const abc::UInt64ArraySample point_id_sample(point_ids.data(), point_count);
    const abc::V3fArraySample point_velocity_sample(velocities_alembic_m_per_s.data(), point_count);
    const abc::FloatArraySample point_width_sample(diameters_m.data(), point_count);
    const geom::OFloatGeomParam::Sample varying_width_sample(point_width_sample, geom::kVaryingScope);
    point_cache_schema.set(
        geom::OPointsSchema::Sample(point_position_sample, point_id_sample, point_velocity_sample, varying_width_sample)
    );
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 2 || argv[1][0] == '\0') {
    std::cerr << "Usage: fluidity_write_point_cache <output.abc>\n";
    return 2;
  }

  try {
    const std::string archive_path = argv[1];
    WritePointCacheArchive(archive_path);
    if (!VerifyPointCacheArchive(archive_path)) {
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << "Point cache failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "PASS: FluidiTyPoints Ogawa archive readback\n";
  return 0;
}
