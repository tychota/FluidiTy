"""Compile hashed schema-1 reference fixtures into a test-only C++ header."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path
import re
import tempfile


MANIFEST = "manifest.json"
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class Array:
    name: str
    dtype: str
    shape: tuple[int, ...]
    values: tuple[float | int, ...]
    units: str


@dataclass(frozen=True)
class Fixture:
    name: str
    arrays: tuple[Array, ...]
    rtol: float
    atol: float
    provenance: str
    default_policy: str


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"JSON numbers must be finite, found {value}")


def schema_document(raw):
    doc = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                     parse_constant=reject_constant)
    if not isinstance(doc, dict):
        raise ValueError("expected a JSON object")
    version = doc.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError("expected schema_version 1")
    return doc


def regular_file(path):
    return path.is_file() and not path.is_symlink()


def read_fixtures(directory):
    manifest = directory / MANIFEST
    if not regular_file(manifest):
        raise ValueError(f"{MANIFEST} is missing or is not a regular file")
    try:
        entries = schema_document(manifest.read_bytes())["fixtures"]
        if not isinstance(entries, dict) or not entries:
            raise ValueError("expected a nonempty fixtures object")
        seen = {MANIFEST}
        for name, digest in entries.items():
            if (not name.endswith(".json") or name == ".json" or ntpath.basename(name) != name
                    or ntpath.isreserved(name)
                    or name.casefold() in seen):
                raise ValueError(f"unsafe or duplicate fixture path {name!r}")
            seen.add(name.casefold())
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"malformed SHA256 for {name!r}")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{MANIFEST}: {exc}") from exc

    actual = {p.name: p for p in directory.iterdir() if p.suffix.lower() == ".json"}
    expected = set(entries) | {MANIFEST}
    if actual.keys() != expected:
        raise ValueError(f"fixture file set mismatch: missing {sorted(expected - actual.keys())}, "
                         f"extra {sorted(actual.keys() - expected)}")

    contents = {}
    for name in sorted(entries):
        path = actual[name]
        if not regular_file(path):
            raise ValueError(f"{name}: expected a regular fixture file")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entries[name].lower():
            raise ValueError(f"{name}: SHA256 mismatch")
        contents[name] = raw

    fixtures = []
    for name, raw in contents.items():
        try:
            fixtures.append(parse_fixture(schema_document(raw), name.removesuffix(".json")))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
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


def parse_array(name, doc):
    nonempty_string(name, "array name")
    if not isinstance(doc, dict):
        raise ValueError("expected an array object")
    dtype = doc["dtype"]
    if dtype not in ("float64", "int64"):
        raise ValueError("dtype must be float64 or int64")
    dimensions = doc["shape"]
    if (not isinstance(dimensions, list)
            or any(type(n) is not int or not 0 <= n <= INT64_MAX for n in dimensions)):
        raise ValueError("shape must contain nonnegative int64 extents")
    shape = tuple(dimensions)
    values = tuple(flatten_values(doc["data"], shape, dtype))
    if len(values) != math.prod(shape):
        raise ValueError("data count does not match declared shape")
    return Array(name, dtype, shape, values, nonempty_string(doc["units"], "units"))


def parse_fixture(doc, name):
    if doc["name"] != name:
        raise ValueError(f"fixture name must match filename stem {name!r}")
    if not isinstance(doc["arrays"], dict) or not doc["arrays"]:
        raise ValueError("expected a nonempty arrays object")
    arrays = {}
    for key, value in sorted(doc["arrays"].items()):
        try:
            arrays[key] = parse_array(key, value)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{key}: {exc}") from exc

    if {"grid", "spacing", "origin"} & arrays.keys():
        for key, shape in (("spacing", (1,)), ("origin", (2,))):
            array = arrays[key]
            if array.dtype != "float64" or array.shape != shape:
                raise ValueError(f"{key} must have dtype float64 and shape {shape}")
        if arrays["spacing"].values[0] <= 0:
            raise ValueError("spacing must be positive")

    metadata = doc["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("expected a metadata object")
    tolerances = metadata["tolerances"]
    if not isinstance(tolerances, dict):
        raise ValueError("expected a tolerances object")
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
    for i, fixture in enumerate(fixtures):
        records = []
        for j, array in enumerate(fixture.arrays):
            identifier = f"kFixture{i}Array{j}"
            shape = ", ".join(f"{n}ULL" for n in array.shape)
            lines.append(f"inline constexpr std::array<std::size_t, {len(array.shape)}> "
                         f"{identifier}Shape = {{{shape}}};")
            if array.dtype == "float64":
                kind, cpp_type = "kFloat64", "double"
                values = [value.hex() for value in array.values]
                f64_data, i64_data = f"{identifier}Values", "{}"
            else:
                kind, cpp_type = "kInt64", "std::int64_t"
                values = [cpp_int(value) for value in array.values]
                f64_data, i64_data = "{}", f"{identifier}Values"
            lines.append(f"inline constexpr std::array<{cpp_type}, {len(values)}> "
                         f"{identifier}Values = {{")
            lines.extend(f"    {value}," for value in values)
            lines.append("};")
            records.append(f"    {{{cpp_string(array.name)}, DType::{kind}, {identifier}Shape, "
                           f"{f64_data}, {i64_data}, {cpp_string(array.units)}}},")
        lines.append(f"inline constexpr std::array<ArrayRecord, {len(records)}> kFixture{i} = {{{{")
        lines.extend(records)
        lines.append("}};\n")

    lines.append(f"inline constexpr std::array<FixtureRecord, {len(fixtures)}> kFixtures = {{{{")
    for i, fixture in enumerate(fixtures):
        lines.append(f"    {{{cpp_string(fixture.name)}, kFixture{i}, "
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
