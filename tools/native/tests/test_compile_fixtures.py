import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


COMPILER = Path(__file__).resolve().parents[1] / "compile_fixtures.py"
CASE = "example.json"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def rewrite_fixture(directory, change):
    path = directory / CASE
    document = json.loads(path.read_bytes())
    change(document)
    write_json(path, document)
    update_digest(directory)


def update_digest(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["fixtures"][CASE] = hashlib.sha256((directory / CASE).read_bytes()).hexdigest()
    write_json(manifest_path, manifest)


def compile_fixtures(directory, header):
    return subprocess.run(
        [sys.executable, str(COMPILER), "--input", str(directory), "--output", str(header)],
        capture_output=True, text=True, encoding="utf-8",
    )


@pytest.fixture
def fixture_input(tmp_path):
    directory = tmp_path / "reference data"
    directory.mkdir()
    write_json(directory / CASE, {
        "schema_version": 1,
        "name": "example",
        "arrays": {
            "grid": {"dtype": "int64", "shape": [2], "data": [1, 2], "units": "1"},
            "spacing": {"dtype": "float64", "shape": [1], "data": [1.0], "units": "1"},
            "origin": {"dtype": "float64", "shape": [2], "data": [0.0, -0.0], "units": "1"},
            "x": {"dtype": "float64", "shape": [1, 2], "data": [[1.0, 0.1]], "units": "1"},
        },
        "metadata": {
            "tolerances": {"rtol": 1e-12, "atol": 1e-14},
            "provenance": "Hand evaluation.",
            "default_policy": "Unit grid.",
        },
    })
    write_json(directory / "manifest.json", {"schema_version": 1, "fixtures": {CASE: ""}})
    update_digest(directory)
    return directory, tmp_path / "generated headers" / "fixture_data.h"


def assert_rejected(directory, header, message):
    result = compile_fixtures(directory, header)
    assert result.returncode == 1, result
    assert message in result.stderr, result.stderr
    assert not header.exists()


def test_manifest_tamper_rejected(fixture_input):
    directory, header = fixture_input
    result = compile_fixtures(directory, header)
    assert result.returncode == 0, result.stderr
    assert header.is_file()
    # An invalid JSON byte must be diagnosed by its hash, before the JSON is parsed.
    (directory / CASE).write_bytes(b"!")
    assert_rejected(directory, header, "SHA256 mismatch")


@pytest.mark.parametrize("field", ["default_policy", "provenance", "tolerances", "origin", "spacing"])
def test_missing_field_rejected(fixture_input, field):
    directory, header = fixture_input
    container = "arrays" if field in {"origin", "spacing"} else "metadata"
    rewrite_fixture(directory, lambda doc: doc[container].pop(field))
    assert_rejected(directory, header, field)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("dtype", "float32", "dtype"),
    ("shape", [1, 3], "shape"),
    ("shape", [True, 2], "shape"),
    ("shape", [-1, 2], "shape"),
    ("data", [1.0, 0.1], "shape"),
    ("data", [[True, 0.1]], "float64"),
    ("units", "", "units"),
])
def test_wrong_dtype_or_shape_rejected(fixture_input, field, value, message):
    directory, header = fixture_input
    rewrite_fixture(directory, lambda doc: doc["arrays"]["x"].update({field: value}))
    assert_rejected(directory, header, message)


@pytest.mark.parametrize("data", [[1.5, 2], [True, 2], [-(2**63) - 1, 2], [1, 2**63]])
def test_invalid_int64_rejected(fixture_input, data):
    directory, header = fixture_input
    rewrite_fixture(directory, lambda doc: doc["arrays"]["grid"].update(data=data))
    assert_rejected(directory, header, "int64")


@pytest.mark.parametrize(("field", "value", "message"), [
    ("x", [[float("nan"), 0.0]], "finite"),
    ("x", [[10**400, 0.0]], "finite"),
    ("spacing", [0.0], "spacing"),
    ("spacing", [-1.0], "spacing"),
    ("origin", [0.0, float("inf")], "finite"),
    ("atol", -1.0, "atol"),
    ("rtol", True, "rtol"),
    ("rtol", float("inf"), "finite"),
])
def test_nonfinite_or_bad_tolerance_rejected(fixture_input, field, value, message):
    directory, header = fixture_input

    def change(doc):
        if field in {"atol", "rtol"}:
            doc["metadata"]["tolerances"][field] = value
        else:
            doc["arrays"][field]["data"] = value

    rewrite_fixture(directory, change)
    assert_rejected(directory, header, message)


@pytest.mark.parametrize(("file", "content", "message"), [
    ("manifest.json", '{"schema_version":2,"fixtures":{}}', "schema_version"),
    ("manifest.json", '{"schema_version":true,"fixtures":{}}', "schema_version"),
    ("manifest.json", '{"schema_version":1,"fixtures":{},"fixtures":{}}', "duplicate"),
    (CASE, '{"schema_version":2}', "schema_version"),
    (CASE, '{"schema_version":1,"name":"example","name":"other"}', "duplicate"),
])
def test_wrong_version_or_duplicate_key_rejected(fixture_input, file, content, message):
    directory, header = fixture_input
    (directory / file).write_text(content, encoding="utf-8")
    if file == CASE:
        update_digest(directory)
    assert_rejected(directory, header, message)


@pytest.mark.parametrize("name", ["../example.json", r"..\example.json", "C:example.json",
                                   "/example.json", "NUL.json", "manifest.json"])
def test_unsafe_manifest_paths_rejected(fixture_input, name):
    directory, header = fixture_input
    write_json(directory / "manifest.json", {"schema_version": 1, "fixtures": {name: "0" * 64}})
    assert_rejected(directory, header, "unsafe")


def test_invalid_hash_rejected(fixture_input):
    directory, header = fixture_input
    write_json(directory / "manifest.json", {"schema_version": 1, "fixtures": {CASE: "g" * 64}})
    assert_rejected(directory, header, "SHA256")


def test_mismatched_file_name_rejected(fixture_input):
    directory, header = fixture_input
    rewrite_fixture(directory, lambda doc: doc.update(name="other"))
    assert_rejected(directory, header, "name")


@pytest.mark.parametrize("extra", [False, True])
def test_manifest_file_set_mismatch_rejected(fixture_input, extra):
    directory, header = fixture_input
    if extra:
        (directory / "extra.json").write_bytes(b"{}")
    else:
        (directory / CASE).unlink()
    assert_rejected(directory, header, "file set")


def test_deterministic_generation(fixture_input):
    directory, header = fixture_input
    result = compile_fixtures(directory, header)
    assert result.returncode == 0, result.stderr
    original = header.read_bytes()
    assert b"0x1.999999999999ap-4" in original
    assert b"-0x0.0p+0" in original

    # Re-encode identical values in a different JSON key order at a new output path.
    path = directory / CASE
    value = json.loads(path.read_bytes())
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    update_digest(directory)
    second = header.with_name("second.h")
    result = compile_fixtures(directory, second)
    assert result.returncode == 0, result.stderr
    assert second.read_bytes() == original


def test_quote_backslash_and_space_paths(fixture_input):
    directory, header = fixture_input
    rewrite_fixture(
        directory, lambda doc: doc["metadata"].update(provenance='A "quote" and a \\ path.')
    )
    result = compile_fixtures(directory, header)
    assert result.returncode == 0, result.stderr
    assert b'A \\"quote\\" and a \\\\ path.' in header.read_bytes()
