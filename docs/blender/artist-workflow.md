# FluidiTy 0.3 artist workflow

Use **Properties > Physics > FluidiTy**. The extension no longer installs a 3D View sidebar.

## Domain, fluid and collision authoring

Start with **Create Liquid Setup**, or open the packaged **Water Tank.blend** example. This creates a domain with a real floor and four side walls, an initial liquid box, and the Water preset. The top is open. Existing scene objects are preserved. **Assign Selected** and the plus buttons let you use existing meshes or create missing role objects.

Choose **Boundaries** explicitly: Open Tank, Free Space, Plane Collider or Slab Collider. Open Tank sends the domain bounds to the native solver. Free Space adds no solid boundary. Plane uses the top Z face of a closed box as an infinite floor; Slab uses the X thickness of a box as an infinite slab. Collider objects may be static or use two linear location keys covering the bake. The engine accepts one collision model, so tank walls and an independent arbitrary mesh collider cannot be combined. Inactive collider assignments are retained when switching models, without participating in the new model.

The Fluid panel offers Water (1000 kg/m³, 0.072 N/m), Low Surface Tension (0.025 N/m), High Surface Tension (0.15 N/m), No Surface Tension, and Custom. These are editable starting points. Presets change only density and surface tension, leaving geometry, velocity, cache paths and solver settings intact. Editing either value switches the preset to Custom. The engine has no viscosity model; these presets do not claim honey/oil behavior.

Initial velocity is specified on the world X/Y/Z axes in m/s. Wall friction is dimensionless. Simulation controls expose maximum step duration (seconds), particle velocity retention, and the pressure iteration budget. These quantities are encoded into the actual resolved scene/initial state and participate in Resume compatibility. The native solver executable is unchanged; the authoring bridge now supports a checked version-2 packet.

The persistent UI uses compact groups and concise status. The question-mark buttons contain geometry/unit/collision details instead of full-height paragraph rows. Add-on Preferences supplies new-setup fluid, cell-width and backend defaults, preview length, and debug visibility.

The original **First Bake.blend** paddle example remains available for compatibility. Existing scenes default to **Legacy Paddle**, retaining the original version-1 packet bytes and fixed material defaults; new material/solver fields are disabled in that mode. Select a modern boundary model to use the new authoring fields. All geometric inputs remain closed, world-axis-aligned boxes; 24 fps, meter scale, at most 1024 seeded particles and frames 0–240 remain the bounded M1 contract. Arbitrary/procedural meshes, multiple independent colliders, volume outputs and viscosity require further solver work.

## Legacy paddle quick start

1. Save the project. Assign **Domain**, **Initial Liquid**, and **Paddle** in Inputs; the arrow beside each input selects it for editing. These are scene-wide simulation inputs, independent of which object is selected.
2. **Set Timing & Units** sets 24 fps, one Blender unit per meter, frame start zero, and the preferred backend. It preserves geometry and animation. Choose the end frame and cell width, then **Check Scene**.
3. **Preview** runs a shortened simulation in its own cache. **Attach Preview** displays it. Check the motion before starting the full bake.
4. **Bake** requires a fresh cache directory. **Cancel** stops at a safe publication boundary; **Resume** continues compatible data. Changed inputs require a fresh cache. The progress bar counts committed samples, including frame zero.
5. **Attach Points** creates native playback. **Relink Cache...** validates a relocated cache before replacing the saved binding. Save the project after attaching.

For the legacy path, the paddle spans Y/Z and translates on X. The qualified RTX backend remains hardware-pinned; CPU is available separately. This release does not close M1 solver acceptance gates.

## Defaults and diagnostics

**Edit > Preferences > Add-ons > FluidiTy** stores the default backend, preview end frame (default 3, giving samples 0–3), and debug visibility. Existing scenes retain their backend until explicitly initialized. Preview uses the preference without changing the full bake end frame.

Diagnostics offers a local **Copy Support Report** action with Blender/build, backend, current/committed frame, status and raw solver diagnostic. Nothing is uploaded. Debug visibility reveals committed-point inspection: stable ID, position in meters and velocity in meters/second. The report includes diagnostic text, which can contain paths reported by the solver.

## Geometry Nodes and fields

The version 2 Points asset exposes:

| Socket | Meaning |
| --- | --- |
| Geometry | Sphere instances for immediate display/rendering |
| Points | Uninstanced native committed point geometry for another node graph |
| Velocity | Point-domain vector in Blender world coordinates, meters/second |
| Speed | Point-domain magnitude of Velocity, meters/second |
| Radius | Unscaled committed display radius, meters |
| Size input | Display-only radius multiplier; accepts a field in a parent graph |
| Material input | Display material for sphere instances |

Select the playback object, open the Geometry Nodes workspace, and use the Points group inside a parent graph. Connect **Points** and its fields to your own nodes, for example Store Named Attribute or Instance on Points. Velocity/Speed/Radius must be evaluated on that point geometry. Stable 64-bit IDs remain authoritative in the committed cache and Inspect; Blender indices are not a persistent-ID contract.

The Physics Display panel also edits the existing modifier's size/material. Reattachment preserves artist-edited modifiers and materials. Existing v1 groups stay intact; v2 groups are used for new attachments. Saved playback/rendering uses native Blender data and needs neither the extension nor a frame-change Python handler.

## Interchange

| Action | Contract |
| --- | --- |
| Export Alembic... | Checked bridge exports every contiguous committed point sample, including empty samples, positions, widths, velocity and IDs |
| Export USD... | Blender exports verified raw point animation, without display materials/world textures; requires 24 fps, meter units and nonempty samples |
| Import Cache... | Native Blender Alembic, USD or OpenVDB reader; external playback does not become a resumable FluidiTy simulation |
| OpenVDB export | Unavailable: M1 does not produce volume grids |

Exports refuse to overwrite existing files. USD preserves the current frame/subframe, timeline range, selection and active object, including errors. It is an interchange derivative; Blender may omit stable IDs. In Blender 5.2.3 an empty-to-populated Alembic sequence exports as separate USD prims, losing the intended single-stream timeline. FluidiTy detects empty samples and rejects that USD export before publishing; use Alembic instead.

## Translation and verification

Edit `blender/fluidity/locale/fr_FR/LC_MESSAGES/fluidity.po`; the POT lists translatable messages. Compile with `python tools/blender/compile_translations.py`. The offline packager recompiles catalogs and includes PO/POT/MO. Runtime loads MO into Blender's translation registry, including operator contexts. The dependency-free compiler supports singular messages, contexts and multiline strings; fuzzy entries are excluded and unsupported plurals fail.

Measured in Blender **5.2.3 LTS e13b728c6880**:

- Physics-only panel discovery, RNA descriptions, preference definitions, catalog readback and repeated registration passed.
- Packaged offline install, actual RTX Preview/Bake/cancel/resume, changed-input rejection, inspect, attachment preservation, relocation/relink and two distinct extension-disabled renders passed.
- Alembic and USD exported/imported all 13 real-bake frames with point position/radius agreement within 1e-5; reverse/repeat visits passed. Alembic's empty-start, changing-count fixture passed; USD's empty-sample rejection passed. Altered USD timing/units and existing destinations were rejected.
- A fresh reviewer evaluated external Velocity/Speed/Radius field connections against all eight native points at frame 3; all matched.
- Native OpenVDB import loaded the reference's `level_set.vdb` surface FLOAT grid. This is one local import check, not qualification of every VDB schema or sequence.
- Actual English/French Physics panels were captured at the default roughly 350-pixel Properties width and visually reviewed. No Python draw errors were logged.

Additional 0.3 verification:

- Actual short Tank, Free Space, Plane and Slab bakes passed on CPU and the pinned RTX backend. Frame-zero point velocities matched the authored world-space vector. Every exposed material/step/velocity control changed canonical output identity.
- The default Water/Open Tank setup completed all 48 frames with 64 particles; final particle centers stayed above the domain floor.
- A packed, installed **Water Tank.blend** passed the authoring tests, including material-change Resume rejection, retained inactive collider transitions and preset preservation of unrelated inputs.
- Native packet tests covered all four collision kinds, invalid material/step parameters, and byte-for-byte version-1 compatibility against the previously packaged bridge. A version-2 packet with equivalent legacy defaults produced the same resolved/state bytes.
- Fresh review found the hidden-collider mode-switch issue; inactive role pointers are now ignored by the selected model while retained for switching back. Regression checks cover Tank/Free/Slab transitions.
- The installed English/French 0.3 panels were captured and reviewed at the measured 424-pixel Physics region width. Paragraph rows have been removed from the persistent workflow. Source formatting passed Ruff and clang-format 19.1.5 over tracked native sources.
- Local receipts: `out/domain-package-check`, `out/domain-installed-rtx`, `out/domain-authoring-cpu`, `out/domain-default-48`, and `out/domain-ui`. Offline ZIP: `out/domain-release/fluidity-0.3.0-windows-x64.zip`, SHA-256 `f783ff942fa5a5039a6f53142d4f515b13320226ceb71f1cb122f25982162359`.

Run the focused native adapter verification with:

```powershell
python tools/blender/verify_authoring_packet.py BRIDGE SOLVER PREVIOUS_BRIDGE out/domain-packet-check
```

Run `verify_domain_authoring.py` inside Blender with `-- OUTPUT_DIRECTORY BIN_DIRECTORY`; add `--cpu` for CPU or `--installed` when using an enabled offline package. `verify_m1_package.py` continues to exercise the original paddle install/recovery/render contract independently.

Research used the official developer-docs checkout at `1fed2eb75041215b07e9c812de6421bd02133adc`: `features/interface/operators.md`, `internationalization.md`, and the Human Interface Guidelines for layouts, writing and design paradigms. The API parameters were also inspected from the pinned Blender RNA. Local receipts and screenshots are under ignored `out/m1-artist-*`, `out/artist-ui`, and `out/blender-api-docs`.
