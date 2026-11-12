# M7 Low-Mach Gas Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Qualify low-Mach gas from a fixed-grid heating oracle through the Blender bake and replay workflow.

**Architecture:** Reference pressure projection checks the thermodynamic source independently. Production transport shares the staggered face topology with the pressure system and publishes cell gas state with the accepted solver state.

**Tech Stack:** Python, NumPy, SciPy, C++20, CUDA, Blender and OpenVDB.

**Spec:** docs/superpowers/specs/2026-11-07-fluidity-design.md sections 8.4 and Artistic acceptance scenes.

## Global Constraints

- Use fixed-composition air with gamma 1.4 and one absolute p0 per connected component.
- Use an ambient-hydrostatic gauge for the projection and residual buoyancy.
- Use APIC as the baseline and gate IPIC and maps on measurements.
- Keep physical density independent of marker weights.

## Review Focus

- Isolated closed cells and independently heated components.
- Positivity under scalar outflow and conservative signed inflow/outflow budgets.
- Rejection before publication when pressure projection or transport fails.
- Resume with component p0, all scalar state, and APIC history.
- Render-only edits preserve the baked cache.

### Task 1 — Fixed-grid reference

- [x] Add reference/tests/test_gas.py and observe missing fluidity_reference.gas.
- [x] Implement project_heating and transport_densities in reference/src/fluidity_reference/gas.py.
- [x] Run the reference suite. Check open boundary displacement against the 25% oracle, sealed pressure response within 1%, isolated pressure, and signed mass and enthalpy budgets.
- [x] Commit the fixture and its evidence.

### Task 2 — Production gas state and solver

- [x] Exercise conservative fuel/soot/enthalpy, bounded reaction, EOS density and residual buoyancy through the ordinary solver entry point.
- [x] Reuse MAC prescribed divergence, pressure boundaries and APIC where their interfaces satisfy gas requirements.
- [x] Verify native CPU and real CUDA, including rejected substep rollback.

### Task 3 — Committed state and replay

- [x] Extend committed-state and checkpoint contracts for cell state, APIC history and component p0.
- [x] Verify uninterrupted and resumed accepted output on CPU and CUDA.

### Task 4 — Fire in a draft

- [x] Wire authored animated fuel/heat and a supported Blender field into the bake.
- [x] Publish density, temperature, fuel and flame grids and check each in the native Volume.
- [x] Verify render edits preserve cache identity and run final acceptance checks.
