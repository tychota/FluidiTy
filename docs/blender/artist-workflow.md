# FluidiTy 0.2 artist workflow

Use **Properties > Physics > FluidiTy**. The extension no longer installs a 3D View sidebar.

1. Save the project. Assign **Domain**, **Initial Liquid**, and **Paddle** in Inputs; the arrow beside each input selects it for editing. These are scene-wide simulation inputs, independent of which object is selected.
2. **Set Timing & Units** sets 24 fps, one Blender unit per meter, frame start zero, and the preferred backend. It preserves geometry and animation. Choose the end frame and cell width, then **Check Scene**.
3. **Preview** runs a shortened simulation in its own cache. **Attach Preview** displays it. Check the motion before starting the full bake.
4. **Bake** requires a fresh cache directory. **Cancel** stops at a safe publication boundary; **Resume** continues compatible data. Changed inputs require a fresh cache. The progress bar counts committed samples, including frame zero.
5. **Attach Points** creates native playback. **Relink Cache...** validates a relocated cache before replacing the saved binding. Save the project after attaching.

M1 remains a bounded first-bake adapter: static, closed, world-axis-aligned domain/liquid boxes, one translating X-axis paddle spanning Y/Z, 0–240 frames, and at most 1024 seeded particles. Arbitrary meshes, procedural source geometry, additional obstacles and volume simulation parameters are not exposed as supported physics. The qualified RTX backend remains hardware-pinned; CPU is available separately. This release does not close M1 solver acceptance gates.

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

Research used the official developer-docs checkout at `1fed2eb75041215b07e9c812de6421bd02133adc`: `features/interface/operators.md`, `internationalization.md`, and the Human Interface Guidelines for layouts, writing and design paradigms. The API parameters were also inspected from the pinned Blender RNA. Local receipts and screenshots are under ignored `out/m1-artist-*`, `out/artist-ui`, and `out/blender-api-docs`.
