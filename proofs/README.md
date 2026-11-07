# Exact pressure and integer pilots

This project contains two pilots from [F5 and F8](../docs/references/formal-contracts.md), checked by Lean in exact arithmetic. It does not discharge either whole contract family or the other eight families.

The toolchain is official Lean **4.34.1** (`leanprover/lean4:v4.34.1`, release commit `5045d0056413266e57c625dcd7c365b10e377c52`). The mathlib dependency is pinned to `d13f23b723b8a846827a245b89c10fc7d3f11612`; `lake-manifest.json` also locks its inherited dependencies. No third-party source is vendored.

With `elan` and network access to the pinned dependencies, run from this directory:

```sh
lake exe cache get
lake build
lake env lean Fluidity/Pressure.lean
lake env lean Fluidity/FixedPoint.lean
```

The default build includes the consumer examples in `Fluidity.lean`. Build outputs and locally retained evidence under `.lake/` and `artifacts/` are ignored. The four targets have also been exported with `olean-export` **0.2.0** and checked against the matching Init prelude by experimental `nl-fast` from nano-lean commit `54ce971a5ac2513a1bac0ac9bca73f672196c0f2`, built with Rust **1.97.1**. This independent checker is experimental, and its build receipt is a trusted external assertion, not a reproducible-build proof. A successful Lean build alone does not establish independent acceptance.

Build `olean-export` 0.2.0 and nano-lean at the pinned revision from their official sources with Rust 1.97.1 and `cargo build --release --locked` (add `--bin nl-fast` for the checker). Supply a JSON receipt with `schema_version: 1`, `nano_lean_commit`, `rust_version`, and `checker_sha256` matching the executable; include source URL, archive checksum and build command to document provenance. Run from the repository root:

```sh
python tools/verify_proofs.py --lean-bin /path/to/lean/bin \
  --exporter /path/to/olean-export --checker /path/to/nl-fast \
  --checker-provenance /path/to/receipt.json --output proofs/artifacts/verification
```

The verifier checks pins and provenance, builds, audits all four axiom closures, exports Init and the four theorem closures under the same Lake environment, and runs the checker with an explicit axiom allowlist and no cache. It writes evidence only on success. `--lake-packages FILE` permits a local package override; its contents and hash are recorded in evidence and are an additional trust input. Keep verification evidence private if paths disclose local information.

The public theorem targets are:

| Target | Exact conclusion and hypotheses |
| --- | --- |
| `Fluidity.pressure_quadratic_form_eq` | Expands `xᵀ A x` for the independently defined component action `A x = R diag(k) Rᵀ x + diag(C) x` into `Σf kf (Σi Rif xi)² + Σi Ci xi²`. No sign assumptions are needed for the equality. |
| `Fluidity.pressure_psd` | The actual operator quadratic form is nonnegative when every `k f` and `C i` is nonnegative. |
| `Fluidity.fixed_point_order_independent` | Two lists of mathematical integers with a supplied permutation proof have equal sums. |
| `Fluidity.fixed_point_partial_bound` | If `Σ |qᵢ| ≤ B` in `ℤ`, then `|sum (q.take n)| ≤ B` for every natural prefix length `n`. |

The pressure model has `n` cells, `m` faces, and arbitrary real incidence entries `R : Fin n → Fin m → ℝ`; it supports `n = 0`, `m = 0`, and zero coefficients. `pressureAction` explicitly performs the two finite component sums and diagonal action. `pressureQuadraticForm` contracts that action with `x`; it is not defined as the sum of squares. The proof rearranges the finite sums and polynomial factors. The diagonal `C` may combine nonnegative anchor and compliance coefficients.

No incidence topology, connected-component kernel characterization, strict positive definiteness, solvability, or gauge theorem is proved. Those remain F5 obligations. The more general nonnegative-conductance pilot allows zero conductances without establishing a graph interpretation.

The integer prefix theorem also handles empty lists, `n = 0`, and `n` exceeding the list length. Together with the permutation theorem it applies to prefixes of any permutation: the total absolute-contribution budget is preserved by permuting and mapping the list. A supplied budget `B = 2^62` gives an inclusive bound, including equality. No quantizer, rounding-error bound, reduction-tree model, signed-word conversion, floating-point agreement, or CPU/CUDA implementation is modeled here. Those remain F8 or implementation obligations.

Each theorem source prints the public targets' axiom closure. The accepted closure is a subset of `{propext, Classical.choice, Quot.sound}`: the pressure targets use all three; integer permutation uses `propext`; the prefix bound uses `propext` and `Quot.sound`. No additional axioms or proof admissions are allowed.
