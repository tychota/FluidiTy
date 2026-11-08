"""Compile hashed schema-1 reference fixtures into a test-only C++ header."""

import argparse
import hashlib
import json
import math
import ntpath
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "manifest.json"
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class FixtureArray:
    name: str
    dtype: str
    shape: tuple[int, ...]
    flat_values: tuple[float | int, ...]
    units: str


@dataclass(frozen=True)
class Fixture:
    name: str
    arrays: tuple[FixtureArray, ...]
    rtol: float
    atol: float
    provenance: str
    default_policy: str


def object_from_unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"JSON numbers must be finite, found {value}")


def parse_schema1_json_object(document_bytes):
    document = json.loads(
        document_bytes.decode("utf-8"), object_pairs_hook=object_from_unique_keys, parse_constant=reject_constant,
    )
    if not isinstance(document, dict):
        # The bytes are valid input. The decoded JSON violates the fixture schema.
        raise ValueError("expected a JSON object")  # noqa: TRY004
    version = document.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError("expected schema_version 1")
    return document


def regular_file(path):
    return path.is_file() and not path.is_symlink()


def read_fixtures(directory):
    manifest_path = directory / MANIFEST
    if not regular_file(manifest_path):
        raise ValueError(f"{MANIFEST} is missing or is not a regular file")
    try:
        fixture_digests = parse_schema1_json_object(manifest_path.read_bytes())["fixtures"]
        if not isinstance(fixture_digests, dict) or not fixture_digests:
            raise ValueError("expected a nonempty fixtures object")
        seen_filenames = {MANIFEST}
        for filename, digest in fixture_digests.items():
            safe_basename = (
                filename.endswith(".json") and filename != ".json" and ntpath.basename(filename) == filename
                and not ntpath.isreserved(filename)
            )
            if not safe_basename or filename.casefold() in seen_filenames:
                raise ValueError(f"unsafe or duplicate fixture path {filename!r}")
            seen_filenames.add(filename.casefold())
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"malformed SHA256 for {filename!r}")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{MANIFEST}: {exc}") from exc

    fixture_paths = {path.name: path for path in directory.iterdir() if path.suffix.lower() == ".json"}
    expected_filenames = set(fixture_digests) | {MANIFEST}
    if fixture_paths.keys() != expected_filenames:
        missing_filenames = sorted(expected_filenames - fixture_paths.keys())
        extra_filenames = sorted(fixture_paths.keys() - expected_filenames)
        raise ValueError(f"fixture file set mismatch: missing {missing_filenames}, extra {extra_filenames}")

    verified_fixture_bytes = {}
    for filename in sorted(fixture_digests):
        fixture_path = fixture_paths[filename]
        if not regular_file(fixture_path):
            raise ValueError(f"{filename}: expected a regular fixture file")
        fixture_bytes = fixture_path.read_bytes()
        if hashlib.sha256(fixture_bytes).hexdigest() != fixture_digests[filename].lower():
            raise ValueError(f"{filename}: SHA256 mismatch")
        verified_fixture_bytes[filename] = fixture_bytes

    # Check the manifest hashes before parsing any fixture content.
    fixtures = []
    for filename, fixture_bytes in verified_fixture_bytes.items():
        try:
            document = parse_schema1_json_object(fixture_bytes)
            fixtures.append(parse_fixture(document, filename.removesuffix(".json")))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{filename}: {exc}") from exc
    return fixtures


def nonempty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def finite_float(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a float64 number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite float64") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite float64")
    return number


def flatten_values(data, shape, dtype):
    if shape:
        if not isinstance(data, list) or len(data) != shape[0]:
            raise ValueError("data does not match declared shape")
        for item in data:
            yield from flatten_values(item, shape[1:], dtype)
    else:
        if dtype == "float64":
            yield finite_float(data, "float64 data")
        elif type(data) is int and INT64_MIN <= data <= INT64_MAX:
            yield data
        else:
            raise ValueError("data must be an int64 integer in range")


def parse_array(name, document):
    nonempty_string(name, "array name")
    if not isinstance(document, dict):
        # Invalid schema content uses the same error path as an invalid array extent.
        raise ValueError("expected an array object")  # noqa: TRY004
    dtype = document["dtype"]
    if dtype not in ("float64", "int64"):
        raise ValueError("dtype must be float64 or int64")
    dimensions = document["shape"]
    if (not isinstance(dimensions, list)
            or any(type(extent) is not int or not 0 <= extent <= INT64_MAX for extent in dimensions)):
        raise ValueError("shape must contain nonnegative int64 extents")
    shape = tuple(dimensions)
    flat_values = tuple(flatten_values(document["data"], shape, dtype))
    if len(flat_values) != math.prod(shape):
        raise ValueError("data count does not match declared shape")
    return FixtureArray(name, dtype, shape, flat_values, nonempty_string(document["units"], "units"))


def parse_fixture(document, name):
    if document["name"] != name:
        raise ValueError(f"fixture name must match filename stem {name!r}")
    if not isinstance(document["arrays"], dict) or not document["arrays"]:
        raise ValueError("expected a nonempty arrays object")
    arrays = {}
    for key, value in sorted(document["arrays"].items()):
        try:
            arrays[key] = parse_array(key, value)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{key}: {exc}") from exc

    is_pressure_fixture = name in ("compliant_pressure", "two_cell_pressure")
    if is_pressure_fixture:
        grid = arrays["grid"]
        if grid.dtype != "int64" or grid.shape != (2,):
            raise ValueError("grid must have dtype int64 and shape (2,)")
        if any(extent <= 0 or extent > UINT32_MAX for extent in grid.flat_values):
            raise ValueError("grid extents must be positive and fit in Grid2D uint32_t dimensions")

    if is_pressure_fixture or {"grid", "spacing", "origin"} & arrays.keys():
        for key, shape in (("spacing", (1,)), ("origin", (2,))):
            array = arrays[key]
            if array.dtype != "float64" or array.shape != shape:
                raise ValueError(f"{key} must have dtype float64 and shape {shape}")
        if arrays["spacing"].flat_values[0] <= 0:
            raise ValueError("spacing must be positive")

    # These checks report invalid JSON schema content through the fixture diagnostic.
    metadata = document["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("expected a metadata object")  # noqa: TRY004
    tolerances = metadata["tolerances"]
    if not isinstance(tolerances, dict):
        raise ValueError("expected a tolerances object")  # noqa: TRY004
    rtol = finite_float(tolerances["rtol"], "rtol")
    atol = finite_float(tolerances["atol"], "atol")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be nonnegative")
    return Fixture(name, tuple(arrays.values()), rtol, atol,
                   nonempty_string(metadata["provenance"], "provenance"),
                   nonempty_string(metadata["default_policy"], "default_policy"))


def cpp_string(value):
    raw = value.encode("utf-8")
    escaped = []
    for byte in raw:
        if byte in (ord('"'), ord("\\")):
            escaped.append("\\" + chr(byte))
        elif 0x20 <= byte <= 0x7e:
            escaped.append(chr(byte))
        else:
            escaped.append(f"\\{byte:03o}")
    return f'std::string_view{{"{"".join(escaped)}", {len(raw)}}}'


def cpp_int(value):
    if value == INT64_MIN:
        return "(-9223372036854775807LL - 1LL)"
    return f"{value}LL"


HEADER = """// Generated by compile_fixtures.py. Do not edit.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace fluidity::fixtures {

enum class DType { kFloat64, kInt64 };

struct ArrayRecord {
  std::string_view name;
  DType dtype;
  std::span<const std::size_t> shape;
  std::span<const double> f64_data;
  std::span<const std::int64_t> i64_data;
  std::string_view units;
};

struct FixtureRecord {
  std::string_view name;
  std::span<const ArrayRecord> arrays;
  double rtol;
  double atol;
  std::string_view provenance;
  std::string_view default_policy;
};

namespace detail {
"""


def render_header(fixtures):
    lines = [HEADER]
    for fixture_index, fixture in enumerate(fixtures):
        array_initializers = []
        for array_index, array in enumerate(fixture.arrays):
            array_symbol_prefix = f"kFixture{fixture_index}Array{array_index}"
            shape_literals = ", ".join(f"{extent}ULL" for extent in array.shape)
            lines.append(f"inline constexpr std::array<std::size_t, {len(array.shape)}> "
                         f"{array_symbol_prefix}Shape = {{{shape_literals}}};")
            if array.dtype == "float64":
                kind, cpp_type = "kFloat64", "double"
                value_literals = [value.hex() for value in array.flat_values]
                f64_data, i64_data = f"{array_symbol_prefix}Values", "{}"
            else:
                kind, cpp_type = "kInt64", "std::int64_t"
                value_literals = [cpp_int(value) for value in array.flat_values]
                f64_data, i64_data = "{}", f"{array_symbol_prefix}Values"
            lines.append(f"inline constexpr std::array<{cpp_type}, {len(value_literals)}> "
                         f"{array_symbol_prefix}Values = {{")
            lines.extend(f"    {value}," for value in value_literals)
            lines.append("};")
            array_initializers.append(
                f"    {{{cpp_string(array.name)}, DType::{kind}, {array_symbol_prefix}Shape, "
                f"{f64_data}, {i64_data}, {cpp_string(array.units)}}},"
            )
        lines.append(
            f"inline constexpr std::array<ArrayRecord, {len(array_initializers)}> kFixture{fixture_index} = {{{{"
        )
        lines.extend(array_initializers)
        lines.append("}};\n")

    lines.append(f"inline constexpr std::array<FixtureRecord, {len(fixtures)}> kFixtures = {{{{")
    for fixture_index, fixture in enumerate(fixtures):
        lines.append(f"    {{{cpp_string(fixture.name)}, kFixture{fixture_index}, "
                     f"{fixture.rtol.hex()}, {fixture.atol.hex()},")
        lines.append(f"     {cpp_string(fixture.provenance)}, {cpp_string(fixture.default_policy)}}},")
    lines.append("""}};

}  // namespace detail

inline std::span<const FixtureRecord> AllFixtures() {
  return detail::kFixtures;
}

inline absl::StatusOr<const ArrayRecord*> FindArray(const FixtureRecord& fixture,
                                                    std::string_view name) {
  for (const ArrayRecord& array : fixture.arrays) {
    if (array.name == name) {
      return &array;
    }
  }
  return absl::NotFoundError("fixture array not found");
}

}  // namespace fluidity::fixtures
""")
    return "\n".join(lines).encode("ascii")


def write_header(output, data):
    if output.is_file() and output.read_bytes() == data:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Never let cleanup or publication clobber a fixture or its manifest.
    if args.output.suffix.lower() == ".json":
        parser.exit(1, "compile_fixtures: output must be a header, not a JSON file\n")
    try:
        write_header(args.output, render_header(read_fixtures(args.input)))
    except (OSError, ValueError, RecursionError) as exc:
        try:
            args.output.unlink(missing_ok=True)
        except OSError as cleanup:
            parser.exit(1, f"compile_fixtures: {exc}; removing failed output: {cleanup}\n")
        parser.exit(1, f"compile_fixtures: {exc}\n")


if __name__ == "__main__":
    main()
