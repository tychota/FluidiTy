# Adaptive pressure graph inputs for M4

The bounded assembler in `native/src/mac_pressure_3d_system.h` takes an immutable `AdaptiveMacTopology3D` and step duration. It reconstructs the global slots from `BuildAdaptiveMacFaceOwnership3D`. M4 should use the same topology and ownership for strain, traction, pressure, and constituent tests.

| Input | Order and meaning |
| --- | --- |
| Leaf keys | Finest-lattice signed lower corner and dyadic width exponent. Canonical leaf order is signed `(z,y,x)`. |
| Pressure cells | One row per canonical leaf. Use the leaf lower corner as `cell_index`, positive liquid volume bounded by the geometric volume, source divergence in s⁻¹, bulk modulus in Pa, and previous accepted pressure in Pa. |
| Pressure subfaces | One physical sample per global slot in `(axis,z,y,x,width exponent)` order, including sealed faces. The graph derives incidence, full area, and two-cell center distance from geometry. |
| Open and blocked parts | Supply physical aperture in m², face mobility in m³/kg, predicted positive-normal velocity in m/s, and signed integrated wall flux in m³/s. The wall flux is allowed only on the blocked part. |
| Exterior faces | The geometry identifies the single incident leaf and its orientation. An open exterior needs finite exterior pressure and positive distance to that pressure. A sealed exterior has zero aperture and zero pressure distance. |

Each oriented subface contributes outward flux with sign +1 at its tail and −1 at its head. The matrix and velocity correction use the same aperture, mobility, and distance. Cell source flux is liquid volume times source divergence. M4 traction should use the geometric face identity, endpoint incidence, and physical aperture from this same frozen roster. For constituent conservation, start from extensive per-leaf inventories in canonical cell order and use one signed constituent flux per global slot. Sum sibling inventories and signed outer subfaces during restriction. An internal sibling flux cancels across the two incident cells. Those conventions provide test inputs for the M4 extension. The pressure fixture has no constituent field.

## Qualified two-leaf control

The shared test fixture uses finest width 0.5 m and leaves `((0,0,0),1)` and `((2,0,0),0)`. One two-sided and thirteen exterior subfaces fill the ownership roster. The two-sided face has full area 0.25 m², aperture 0.125 m², and +0.0625 m³/s prescribed wall flux. The coarse source adds 0.09375 m³/s. The fine outlet has aperture 0.25 m², pressure 1 Pa, distance 0.25 m, and mobility 1 m³/kg. All other exterior faces are sealed. With a 1 s step, the analytic coarse and fine pressures are 1.28125 Pa and 1.09375 Pa. The interior fluid velocity is +0.25 m/s and the exterior fluid velocity is +0.375 m/s. The CPU control also closes the interior aperture and verifies a gauge on the isolated coarse component with compatible signed wall and source flux.

## Runtime boundary

The scheduler still uses uniform block/face storage, uniform transfer and uniform checkpoint records. The checked adaptive transfer evaluator covers bounded aligned two-level patches. A production caller still needs a general graded-leaf transfer and carrier mapping, certified cut and source samples from the trial geometry, and accepted adaptive topology and inventory storage through device publication and checkpoint. The owning graph assembler allocates when it freezes the graph. Integration into the existing allocation-free attempt path needs reserved staging.

The graph admits at most 4,096 pressure cells and 24,576 total subfaces, including sealed entries. The slot limit can bind before the leaf limit. No adaptive interrupted-replay or rejected-scheduler-trial receipt exists for this slice. No wave-scale runtime or memory claim is made.
