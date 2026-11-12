# Adaptive Simulate runtime integration

- Bind physical pressure inputs to canonical graded topology. Check cell volume, global subface order, two-sided cut links, exterior pressure, and positive-normal wall flux before publishing a frozen operator. Compare the original operator, bounded CPU projection, and real CUDA projection.
- Extend accepted state and checkpoint records with adaptive topology, extensive inventories, canonical face carriers and warm starts. Validate them before restoring a scheduler. Keep candidate state inside the existing accepted-state transaction.
- Wire adaptive activation, transfer, pressure and gather together so every physical stage uses the trial topology. Run rejection and fresh-process interrupted replay on both backends.
- Document the checked M4 input and receipt, including bounded capacities and remaining wave qualification.
