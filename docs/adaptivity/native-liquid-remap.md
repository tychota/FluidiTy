# M3 native conservative liquid remap

The host API restricts eight equal-volume Cartesian child inventories in
lexicographic xyz order, with z changing fastest. Prolongation uses the parent and
its three same-size low/high neighbor pairs. It reconstructs each extensive
channel with minmod slopes at offsets ±¼ parent width, limits the reconstruction
to the stencil extrema, and converts the parent-scale reconstruction to a child
integral by dividing by eight. Mass, reference volume and current volume must
be nonnegative. Momentum and enthalpy are signed. Current volume is the sole
constituent volume for this single-liquid model.

Restriction uses the existing deterministic compensated sum. Prolongation checks
conservation by restricting the stored binary64 children. The roundoff bound is
64 machine epsilons times the larger of the absolute parent and largest absolute
child in each channel. A nonzero subnormal parent or an all-subnormal scale
requires exact conservation. If the representation loses inventory, the call
returns an error and the input stays unchanged.

Four equal-area subfaces use unique owner slots and explicit signs relative to
the coarse normal. Restriction converts each integrated flux to the coarse
orientation and sums in ascending slot order. Prolongation divides the coarse
flux into exact quarters and returns each in its supplied slot's local
orientation. Duplicate slots, invalid signs, nonfinite inputs and unrepresentable
arithmetic fail before publishing a result.

The five focused tests cover exact affine prolongation and restriction, extrema
and positivity, rejection of subnormal loss and overflow, signed cancellation,
duplicate slots, orientation and input preservation on refusal. The flux
cancellation fixture returns 0.5 m³/s with either input order, and the mixed-sign
prolongation round trip returns −10 m³/s. The Python reference matched every affine child, the restricted parent and the signed cancellation result.

```powershell
& ./out/build/windows-debug-cpu/fluidity_native_tests.exe '--gtest_filter=LiquidRemap3D.*'
ctest --test-dir out/build/windows-debug-cpu --output-on-failure
```

All five focused tests and all 564 CTest cases passed on the qualified Windows Debug CPU configuration with MSVC
19.44.35229, the cached Debug Conan toolchain and the CPU qualification receipt.
See the [native build procedure](../native/README.md#cpu-only-build-on-windows).

The API covers one aligned 2:1 transfer and assumes the caller has already
identified the children, faces and complete neighbors. Cut-cell remap, mixture
EOS recovery, adaptive topology and hysteresis, pressure warm starts, and
accepted-step replay still require integration. See the
[M3 plan](../superpowers/plans/2026-11-11-m3-adaptivity-and-band.md#task-2--native-parity-masking-and-conservative-remap).
