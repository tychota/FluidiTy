"""Validate explicit wave inputs and author the qualified bounded fixture."""

import argparse
from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path

from preflight_reference_wave import CAPACITIES, CONTROL_FRAME_END, digest, run


BOUNDED = {
    "width_m": 1, "length_m": 5, "inlet_depth_m": 0.75, "cell_width_m": 0.25,
    "wave_height_m": 0.25, "wave_period_s": 2, "frame_end": CONTROL_FRAME_END, "fps": 24,
}
GRADED_INPUTS = (
    "camera_refinement_footprint", "canonical_leaf_roster", "resolved_scene", "initial_state",
    "pipeline_settings", "matched_error_targets",
)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest field {key}")
        result[key] = value
    return result


def positive(inputs, field):
    value = inputs.get(field)
    if type(value) not in (int, Decimal) or (isinstance(value, Decimal) and not value.is_finite()) or value <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return Fraction(str(value))


def floor_sum(count, modulus, multiplier, offset):
    """Sum floor((multiplier*i + offset)/modulus) in logarithmic steps."""
    total = 0
    while True:
        quotient, multiplier = divmod(multiplier, modulus)
        total += count * (count - 1) // 2 * quotient
        quotient, offset = divmod(offset, modulus)
        total += count * quotient
        upper = multiplier * count + offset
        if upper < modulus:
            return total
        count, offset = divmod(upper, modulus)
        modulus, multiplier = multiplier, modulus


def uniform_wet_count(width, length, depth, spacing):
    # Count columns whose centers are inside the taper and sum ceil(depth/h).
    columns = max(0, math.ceil(length / spacing - Fraction(1, 2)))
    if not columns:
        return 0
    first = depth / spacing * (1 - (columns - Fraction(1, 2)) * spacing / length)
    step = depth / length
    denominator = math.lcm(first.denominator, step.denominator)
    first_units = int(first * denominator)
    step_units = int(step * denominator)
    return math.ceil(width / spacing) * floor_sum(columns, denominator, step_units, first_units + denominator - 1)


def validate(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {"version", "case", "inputs", "provenance"}:
        raise ValueError("manifest requires version, case, inputs and provenance")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("unsupported wave manifest version")
    if not isinstance(manifest["provenance"], str) or not manifest["provenance"].strip():
        raise ValueError("provenance must describe the source of the authored values")
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object")
    case = manifest["case"]
    base_fields = set(BOUNDED)
    if case == "bounded_column":
        if set(inputs) != base_fields or any(positive(inputs, key) != Fraction(str(value)) for key, value in BOUNDED.items()):
            raise ValueError("bounded_column values must match the pinned native fixture")
        if any(type(inputs[key]) is not int for key in ("frame_end", "fps")):
            raise ValueError("frame_end and fps must be integers")
        return {"status": "fixture_admitted", "reference_gate_credit": False}
    if case not in ("uniform_wedge", "graded_reference"):
        raise ValueError("unknown wave case")
    fields = base_fields | {"height_m"}
    allowed = fields | (set(GRADED_INPUTS) if case == "graded_reference" else set())
    if not fields <= set(inputs) <= allowed:
        raise ValueError("invalid wedge fields")
    numeric = {field: positive(inputs, field) for field in fields}
    if any(type(inputs[key]) is not int for key in ("frame_end", "fps")):
        raise ValueError("frame_end and fps must be integers")
    if numeric["inlet_depth_m"] > numeric["height_m"]:
        raise ValueError("inlet depth exceeds domain height")
    if case == "graded_reference":
        for key in GRADED_INPUTS:
            if key in inputs and (not isinstance(inputs[key], str) or not inputs[key].strip()):
                raise ValueError(f"{key} must identify an authored input")
        missing = [key for key in GRADED_INPUTS if key not in inputs]
        return {
            "status": "missing_authored_inputs" if missing else "graded_admission_not_verified",
            "missing_inputs": missing,
            "reference_gate_credit": False,
        }
    width, length, height, depth, h = (
        numeric[key] for key in ("width_m", "length_m", "height_m", "inlet_depth_m", "cell_width_m")
    )
    wet = uniform_wet_count(width, length, depth, h)
    cap = CAPACITIES["public_uniform"]["cells"]
    return {
        "status": "capacity_refused" if wet > cap else "geometry_within_cell_cap_only",
        "uniform_lattice_dimensions": [math.ceil(extent / h) for extent in (width, length, height)],
        "initial_wet_cells_at_column_centers": wet,
        "uniform_cell_capacity": cap,
        "cell_capacity_excess": max(0, wet - cap),
        "reference_gate_credit": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if bool(args.solver) != bool(args.fixture):
        parser.error("--solver and --fixture must be supplied together")
    manifest_text = args.manifest.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text, parse_float=Decimal, object_pairs_hook=unique_object)
    receipt = validate(manifest)
    receipt["manifest_sha256"] = digest(args.manifest)
    receipt["case"] = manifest["case"]
    receipt["provenance"] = manifest["provenance"]
    root = args.output.resolve()
    root.mkdir(parents=True)
    try:
        if receipt["status"] == "fixture_admitted" and args.solver:
            solver, fixture = args.solver.resolve(), args.fixture.resolve()
            scene = root / "wave.flrs"
            volume = root / "wave.flvc"
            commands = receipt["commands"] = []
            run(root, "author", [fixture, "scene", scene, solver, args.backend, "bounded-wave-column-seed"], commands)
            run(root, "volumes", [fixture, "wave-volume-settings", volume], commands)
            files = (scene, scene.with_suffix(".flrs.flst"), volume)
            receipt["artifact_sha256"] = {path.name: digest(path) for path in files}
            receipt["solver_sha256"] = digest(solver)
            receipt["fixture_sha256"] = digest(fixture)
    finally:
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
