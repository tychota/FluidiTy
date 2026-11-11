import copy
import json

import pytest

from tools.native.m2_native_surface_baseline import parse_measurement, validate_measurement


def sample():
    return {
        "frame": 0, "spacing_m": 0.0125, "source_points_digest": "a" * 64,
        "material_volume_m3": 1.0, "raw_volume_m3": 0.9, "corrected_volume_m3": 1.001,
        "raw_volume_error": -0.1, "corrected_volume_error": 0.001,
        "maximum_offset_m": 0.001, "offset_limit_m": 0.0025,
        "vertices": 4, "triangles": 4, "components": 1, "euler": 2,
        "self_intersections": 0, "peak_process_rss_bytes": 10000, "runtime_seconds": 0.1,
        "raw_radial_rms_m": 0.01, "raw_radial_max_m": 0.02,
        "corrected_radial_rms_m": 0.001, "corrected_radial_max_m": 0.002,
    }


def oracle():
    return {"signed_volume_m3": 0.9, "reference_volume_m3": 1.0, "vertices": 4, "triangles": 4,
            "vertex_radial_error_rms_m": 0.01, "vertex_radial_error_max_m": 0.02}


def test_requires_exactly_one_native_measurement():
    line = "native_surface_measurement " + json.dumps(sample())
    assert parse_measurement("[ RUN ]\n" + line + "\n[ OK ]") == sample()
    with pytest.raises(ValueError, match="exactly one"):
        parse_measurement("[ SKIPPED ]")
    with pytest.raises(ValueError, match="exactly one"):
        parse_measurement(line + "\n" + line)


@pytest.mark.parametrize("field,value", [
    ("source_points_digest", "b" * 64), ("corrected_volume_m3", 1.1),
    ("corrected_volume_error", 0.0), ("self_intersections", 1),
    ("maximum_offset_m", 0.003), ("raw_volume_m3", 0.8), ("runtime_seconds", float("nan")),
])
def test_rejects_lineage_geometry_false_claims_and_nonfinite_receipts(field, value):
    record = copy.deepcopy(sample())
    record[field] = value
    with pytest.raises(ValueError):
        validate_measurement(record, 0, 0.0125, "a" * 64, oracle())


def test_accepts_independently_matching_raw_geometry_and_measured_correction():
    validate_measurement(sample(), 0, 0.0125, "a" * 64, oracle())
