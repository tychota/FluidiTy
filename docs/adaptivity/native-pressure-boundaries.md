# Adaptive exterior pressure qualification

The explicit graph accepts exterior subfaces alongside interior coarse/fine links. Each exterior subface specifies its incident cell, positive-normal orientation, full area, aperture and signed integrated wall flux. Open exterior faces use Dirichlet pressure and sealed faces have zero aperture. Slots are unique across both spans and the solve accumulates them in canonical slot order. The topology caller supplies the geometry and complete face ownership.

The manufactured two-cell case connects volumes of 1 and 0.125 m³ through a 0.25 m² subface. The coarse sealed wall supplies 0.125 m³/s inward. A partially open fine boundary adds 0.0625 m³/s inward wall flux and receives 0.1875 m³/s outward fluid flux. The analytic pressures are 1.75 and 1.375 Pa. The internal normal velocity is +0.5 m/s. The exterior fluid normal velocity is −1.5 m/s when the fine cell is the head and +1.5 m/s when it is the tail.

The test requires FP64 physical defect and normal-velocity errors within 1e-12, bounded CPU pressures within 1e-6 Pa, and the 0.1875 m³/s inward wall-flux receipt within 1e-12 m³/s. It rejects duplicate slot ownership and nonzero wall flux on a fully open boundary, then checks the prior pressure publication. The existing sealed edge/corner and uniform open-boundary cases run as controls.

From a configured native CPU build, run the focused test with `ctest --test-dir <build-directory> --output-on-failure -R 'MacPressureExplicitGraph3D.ProjectsExteriorPressureAndSignedWallFlux'`. Use the same command with `-R '^MacPressure'` to include the controls. The fixture is independent of M2 surface or motion inputs. Production still needs to map accepted cut geometry and wall/source data onto these records.
