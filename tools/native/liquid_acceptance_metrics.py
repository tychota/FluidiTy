"""Physical measurements from verified, frame-ordered free-fall points."""

import math
import sys

FREE_FALL_CASES = {"free-fall-pic": 0.0, "free-fall-default": 0.6, "free-fall-retained": 1.0}
FREE_FALL_LAST_FRAME = 6
FRAMES_PER_SECOND = 24
GRAVITY_M_PER_S2 = 9.81
MAX_RELATIVE_ACCELERATION_ERROR = 0.001
MAX_TRANSVERSE_SPEED_M_PER_S = 1e-10


def finite_number(value):
    """JSON booleans are not numeric observations."""
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


def vector3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(map(finite_number, value)):
        raise ValueError("expected a finite three-component vector")
    return value


def uniform_history_velocities(energy, stored_velocity, sampled_old_velocity, sampled_new_velocity):
    """Evaluate the scalar §5.1 equation with maximum liquid separation.

    In uniform free fall, partition of unity gives sampled_old_velocity equal to
    stored_velocity. Then the residual is zero, so the affine-in-energy stored
    path has equal endpoint values. The advecting path also has zero correction
    for any separation weight, including the maximum beta = energy.
    """
    if not finite_number(sampled_old_velocity):
        raise ValueError("consistent old history requires a finite old sample")
    if not finite_number(energy) or not 0 <= energy <= 1:
        raise ValueError("energy must be in [0, 1]")
    if not all(map(finite_number, (stored_velocity, sampled_new_velocity))):
        raise ValueError("expected finite stored and new velocities")
    residual = stored_velocity - sampled_old_velocity
    retained_velocity = 0.99 * energy * residual
    return sampled_new_velocity + retained_velocity, sampled_new_velocity + energy * retained_velocity


def evaluate_free_fall(frame_samples):
    """Measure each stored-velocity interval and report position change separately.

    Samples must cover exactly frames 0 through 6, with the single stable ID 1
    at the frozen seed in frame zero. The input's accepted clocks are used for
    velocity differences, after checking them against the 24 fps schedule.
    Malformed observations raise ValueError. Valid observations outside the
    physical tolerances return their measurements with passed=False.
    """
    if len(frame_samples) != FREE_FALL_LAST_FRAME + 1:
        raise ValueError("free fall requires all seven committed frames")
    positions = []
    velocities = []
    times = []
    for ordinal, sample in enumerate(frame_samples):
        if not isinstance(sample, dict) or type(sample.get("frame")) is not int or sample["frame"] != ordinal:
            raise ValueError(f"unexpected frame ordinal at sample {ordinal}")
        time = sample.get("time_seconds")
        if not finite_number(time) or time != ordinal / FRAMES_PER_SECOND:
            raise ValueError(f"wrong accepted time at frame {ordinal}")
        points = sample.get("points")
        if not isinstance(points, list) or len(points) != 1 or not isinstance(points[0], dict):
            raise ValueError(f"expected exactly one point at frame {ordinal}")
        point = points[0]
        if type(point.get("id")) is not int or point["id"] != 1:
            raise ValueError(f"wrong particle ID at frame {ordinal}")
        times.append(time)
        positions.append(vector3(point.get("world_position")))
        velocities.append(vector3(point.get("world_velocity")))
    if tuple(positions[0]) != (2.5, 2.5, 2.5) or tuple(velocities[0]) != (0, 0, 0):
        raise ValueError("frame zero does not match the frozen free-fall seed")

    intervals = []
    for end_frame in range(1, len(times)):
        start_frame = end_frame - 1
        duration = times[end_frame] - times[start_frame]
        acceleration = (velocities[end_frame][1] - velocities[start_frame][1]) / duration
        relative_error = abs(acceleration + GRAVITY_M_PER_S2) / GRAVITY_M_PER_S2
        interval_displacement = [end - start for start, end in zip(positions[start_frame], positions[end_frame])]
        if not all(map(finite_number, (acceleration, relative_error, *interval_displacement))):
            raise ValueError(f"nonfinite measurement at frame {end_frame}")
        intervals.append({
            "end_frame": end_frame,
            "duration_s": duration,
            "stored_acceleration_y_m_per_s2": acceleration,
            "relative_acceleration_error": relative_error,
            "world_displacement_m": interval_displacement,
        })

    displacement = [end - start for start, end in zip(positions[0], positions[-1])]
    transverse_speed = max(math.hypot(velocity[0], velocity[2]) for velocity in velocities)
    if not all(map(finite_number, (transverse_speed, *displacement))):
        raise ValueError("nonfinite transverse or displacement measurement")
    maximum_acceleration_error = max(interval["relative_acceleration_error"] for interval in intervals)
    return {
        "passed": (
            maximum_acceleration_error <= MAX_RELATIVE_ACCELERATION_ERROR
            and transverse_speed <= MAX_TRANSVERSE_SPEED_M_PER_S
        ),
        "gravity_m_per_s2": GRAVITY_M_PER_S2,
        "maximum_relative_acceleration_error": maximum_acceleration_error,
        "maximum_transverse_speed_m_per_s": transverse_speed,
        "world_displacement_m": displacement,
        "intervals": intervals,
    }
