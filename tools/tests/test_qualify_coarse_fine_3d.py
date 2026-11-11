"""Regression checks for the standalone 3D qualification receipt."""

import runpy
from pathlib import Path


def test_qualification():
    script = Path(__file__).resolve().parents[1] / "adaptivity" / "qualify_coarse_fine_3d.py"
    report = runpy.run_path(str(script))["qualify"]()

    assert report["passed"]
    assert len(report["rows"]) == 6
    for row in report["rows"]:
        assert row["queries"] == 120
        assert row["mixed_queries"] > 0
        assert row["pressure"]["interface_faces"] == 4 * row["codimension"]
        assert row["pressure"]["coarse_outward_flux"] == [0.25] * row["codimension"]
        assert row["pressure"]["signed_balance_defect"] == 0
        assert row["pressure"]["corrupt_signed_balance_defect"] == 0.75
        assert row["mask"]["incomplete"]["rank"] == 3
        assert row["mask"]["coplanar"]["rank"] == 2
        assert row["mask"]["incomplete"]["raw_partition_defect"] > 0.01
        assert row["mask"]["coplanar"]["raw_partition_defect"] > 0.01



def test_half_space_marginal_matches_2d_oracle():
    import numpy as np
    from fluidity_reference.experiments.coarse_fine_basis import evaluate_coarse_fine_basis

    script = Path(__file__).resolve().parents[1] / "adaptivity" / "qualify_coarse_fine_3d.py"
    stencil = runpy.run_path(str(script))["stencil"]
    position = np.array([0.875, -0.375, 0.125])
    for normal in (0, 1):
        anchors, weights, gradients, levels = stencil(position, np.zeros(3), 1, np.ones(3), normal, (0,))
        expected = evaluate_coarse_fine_basis(
            position[:2], interface_origin=np.zeros(2), coarse_cell_width=1,
            face_normal_axis=normal, interface_axis=0, dtype=np.float64,
        )
        expected_keys = set(zip(expected.levels, map(tuple, expected.anchor_world_positions), strict=True))
        assert set(zip(levels, map(tuple, anchors[:, :2]), strict=True)) == expected_keys
        for index, anchor in enumerate(expected.anchor_world_positions):
            mask = (levels == expected.levels[index]) & np.all(anchors[:, :2] == anchor, axis=1)
            np.testing.assert_allclose(weights[mask].sum(), expected.weights[index], rtol=0, atol=1e-15)
            expected_gradient = np.append(expected.physical_gradients[index], 0)
            np.testing.assert_allclose(gradients[mask].sum(axis=0), expected_gradient, rtol=0, atol=1e-15)



def test_nonfinite_query_is_reported_as_a_failure(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "adaptivity" / "qualify_coarse_fine_3d.py"
    qualify = runpy.run_path(str(script))["qualify"]
    namespace = qualify.__globals__
    measure = namespace["defects"]
    calls = 0

    def inject_nonfinite(*args):
        nonlocal calls
        calls += 1
        result = measure(*args)
        if calls == 3:
            result["partition"] = float("nan")
        return result

    monkeypatch.setitem(namespace, "defects", inject_nonfinite)
    report = qualify()
    assert not report["passed"]
    assert report["rows"][0]["peak_defects"]["adaptive"]["partition"] is None
