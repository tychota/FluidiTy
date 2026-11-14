# FluidiTy

FluidiTy will be a simulator for liquids, snow, gas (fire and smoke) and moving solids, driven from Blender. When it
is finished, it will move particles through a grid of cells of mixed sizes, run on the CPU or on an NVIDIA GPU, and
write its results in open formats (Alembic, USD, OpenVDB) that Blender reads natively. Most of it does not exist
yet (see "Status").

## Status

This branch, `rebuild`, is a rebuild from the ground up, in progress. It shares no history with the earlier proof of
concept. Modules arrive one at a time, each in its final form, so most of the solver does not exist here yet.
GPU code is compiled but not run until the owner's hardware run is recorded in `docs/gpu-runs.md`.

## Where things are

- `AGENTS.md`: the rules every file and every commit follow. Read it before writing code.
- `docs/glossary.md`: the words the code uses, what they mean, and the words it never uses.
- `docs/design/programme.md`: the design of the whole rebuild, its decisions and the module order.
