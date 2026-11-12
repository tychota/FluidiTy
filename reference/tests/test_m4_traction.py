import pytest

from fluidity_reference.traction import measure_couette, measure_extension


@pytest.mark.parametrize('resolution', [16, 32])
@pytest.mark.parametrize('dt', [0.005, 0.0025])
def test_uniform_extension_fractional_traction(resolution, dt):
    report = measure_extension(resolution, dt, viscosity_ratio=1, coupled=False)
    assert report.normal_rms < 1e-10
    assert report.tangential_max < 1e-10


@pytest.mark.parametrize('resolution', [16, 32])
@pytest.mark.parametrize('dt', [0.005, 0.0025])
@pytest.mark.parametrize('ratio', [1, 10, 1000])
def test_coupled_extension_preserves_manufactured_traction(resolution, dt, ratio):
    report = measure_extension(resolution, dt, viscosity_ratio=ratio, coupled=True)
    assert report.normal_rms < 1e-8
    assert report.normal_max < 1e-8
    assert report.tangential_rms < 1e-8
    assert report.tangential_max < 1e-8


@pytest.mark.parametrize('resolution', [16, 32])
@pytest.mark.parametrize('dt', [0.005, 0.0025])
@pytest.mark.parametrize('ratio', [1, 10, 1000])
def test_couette_reference_preserves_wall_driven_tangential_traction(resolution, dt, ratio):
    report = measure_couette(resolution, dt, viscosity_ratio=ratio)
    assert report.normal_max == 0.0
    assert report.tangential_rms < 1e-8
    assert report.tangential_max < 1e-8

@pytest.mark.parametrize('resolution', [16, 32])
@pytest.mark.parametrize('dt', [0.005, 0.0025])
@pytest.mark.parametrize('ratio', [10, 1000])
def test_uncoupled_extension_exceeds_the_predeclared_normal_traction_gate(resolution, dt, ratio):
    report = measure_extension(resolution, dt, viscosity_ratio=ratio, coupled=False)
    assert report.normal_rms > 0.05
    assert report.normal_max > 0.10
