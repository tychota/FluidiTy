# Adaptive geometric face ownership

A graded leaf side is either uncovered, fully shared with neighboring leaves, or shared in individual quarters at a 2:1 transition. The ownership pass preserves each existing interior subface. It emits one exterior face for a wholly uncovered side and one exterior quarter for each gap on a partially covered coarse side. Each exterior record names the owning cell and its positive-normal incidence. Interior and exterior subfaces share one deterministic slot order by axis, signed lattice coordinate and width exponent.

The focused fixture places one fine neighbor at a signed coarse face, then reverses the orientation. Its interior area is 0.25 m², and the three uncovered exterior quarters add 0.75 m². The total area on each leaf side exactly matches the square of its width. The graph has one interior and thirteen exterior slots, with every slot used once.

Run `ctest --test-dir <build-directory> --output-on-failure -R '^AdaptiveMacTopology3D'` after configuring a native CPU build. Physical apertures and boundary conditions come from the accepted geometry and motion at the production call site.
