# M3 native 3D coarse–fine transfer parity

The C++20 reference evaluates the aligned half-space, edge and corner fixtures from the
[3D qualification](README.md#m3-3d-coarsefine-qualification). It keeps coarse Greville
anchors and sums positive dyadic child contributions by the first axis that fails closed
containment. Each accepted fine row uses its own anchor. Rows report normalized offsets
and derivatives in world-axis order, with FP32 stencil arithmetic after normalizing the
world query. This reference covers one 2:1 transition within 64 coarse cell widths of the
patch origin on each axis.

A row key contains the level, MAC face component and signed world-axis lattice index
of the low-face owner. The oracle numbers coarse as level 0 and fine as level 1. To map
that key to a topology that numbers width by its exponent relative to the finest cell,
use exponent `1 - level`. The owner's lower corner in finest-grid indices is
`patch_origin_in_finest_indices + owner.lattice_index * (2 - level)`. The conversion
requires the common patch origin to be aligned to the topology's finest lattice. Keep
the face component and use the cell's low face at the resulting level.

The Release CPU tests exercise the oracle's 360 adaptive queries and 120 unique
same-degree uniform queries. They check all three face components, two widths,
negative world coordinates, reflected and rotated junctions, knots, and off-knot
positions. The half-space, edge and corner produced 48, 48 and 44 mixed-level queries,
respectively. Three measured high-defect queries check the owner, weight and gradient
of selected rows against the oracle, including a signed corner. The exact-knot case
checks the fine containment equality and the separate coarse and fine owners.

| FP32 metric | Largest defect | Gate |
| --- | ---: | ---: |
| Partition | 2.38419e-7 | 1e-6 |
| First moment | 4.47035e-8 | 1e-6 |
| Gradient sum | 2.38419e-7 | 2e-6 |
| Affine gradient moment | 2.38419e-7 | 2e-6 |
| Affine value | 8.94070e-8 | 1e-6 |
| Affine gradient | 8.94070e-8 | 2e-6 |

Removing the largest row left full affine rank and a partition defect of approximately
0.289256. Keeping the plane through that anchor reduced the rank to two and gave
defects of 0.230143, 0.233826 and 0.237393 for the three topologies. Exact rank uses
integer coordinates in quarter-coarse-cell units. Adding one off-plane carrier restored
rank three, and removing all carriers returned `kFailedPrecondition`. The two
nonempty masks report their respective PIC requirements without renormalizing weights.

On Windows, the Release MSVC 19.44.35229 build from base
`601d775e5f989bbad38b7c13d518fc0be9df89e1` passed all 506 tests in the native CPU
test executable, including the five `CoarseFineTransfer3D.*` tests. The executable
SHA-256 was `9051d4045a789e1fe0ee2bc36fa2656e5fd7d0f854daa36a1e6bf6fea6da4d34`.
The 67 Python 2D and 3D qualification controls and the clang-format 19.1.5 native
source check passed.

Build using the [Windows native procedure](../native/README.md#cpu-only-build-on-windows),
then substitute the path to the native test executable in the selected build tree.

```powershell
& ./out/build/windows-release-cpu/fluidity_native_tests.exe '--gtest_filter=CoarseFineTransfer3D.*' '--gtest_output=xml:out/adaptivity-m3/native-transfer.xml'
```

Native FP64 and CUDA parity, solver integration, multilevel
junctions and conservative remap remain open.

## Masked PIC fallback

The checked CPU fallback takes one caller-certified new-stage Eulerian normal velocity
for every retained owner key. It uses only retained weights and normalizes their sum
in canonical owner order. A low deposited mass can coexist with a certified usable
Eulerian velocity. The low-mass value from deposit normalization is a placeholder
under that separate API's contract.

On success the gather receives the PIC velocity for the affected component. Its grid
velocity increment, particle-minus-old-PIC residual, fitted velocity slope row and
velocity gradient row are zero. The fallback disables particle affine use and sets
`new_pic_fallback`. A low-mass donor sets `low_mass_fallback`. Existing low-mass and
old-PIC flags and unaffected component values carry through. Empty masks or missing
retained carriers return `kFailedPrecondition`. Duplicate, unexpected or masked
owner keys and nonfinite donor velocities return `kInvalidArgument`. Failures leave
the gather unchanged.

The same Release CPU procedure passed all 510 tests, including nine focused transfer
tests. The four PIC cases cover exact constant reproduction, a weighted non-affine
coplanar donor field, reversed order including signed zeros, a single certified
low-mass donor and transactional refusal of malformed support. The executable
SHA-256 was `f31e8d5bb3dd60e8bff05917153db0a8802fcb3014037be65d73837f2c797802`.
The clang-format 19.1.5 check passed. Production substep integration and CUDA
parity remain open.
