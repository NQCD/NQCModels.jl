# Full dimensional model library

## ASE interface

The easiest way to obtain potentials and forces from established codes is to
use the interfaces implemented in [ASE](https://wiki.fysik.dtu.dk/ase/).

We provide the `ClassicalASEModel` (available through `NQCDInterfASE.jl`) which wraps an ASE atoms object and its
associated calculator to implement the required `potential` and
`derivative` functions.

## AtomsCalculators.jl interface

[AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl) is a framework developed by
[JuliaMolSim](https://github.com/JuliaMolSim) that provides a common interface for calculators compatible
with [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl) systems.
NQCModels.jl can interoperate with AtomsCalculators in both directions:

- **AtomsCalculators → NQCModels**: Wrap any AtomsCalculators-compatible calculator as an
  NQCModels `ClassicalModel` using [`AtomsCalculatorsModel`](@ref).
- **NQCModels → AtomsCalculators**: All `ClassicalModel` types in NQCModels automatically implement
  the AtomsCalculators interface, so they can be used wherever an AtomsCalculators calculator is expected.

This bidirectional interoperability makes it straightforward to connect NQCModels to the broader
JuliaMolSim ecosystem, including packages such as
[DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl),
[ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl), and others.

!!! note

    All unit conversions between NQCModels' internal atomic units and the units used by
    the AtomsCalculators calculator are handled automatically.

### Using an AtomsCalculators calculator as an NQCModels model

Any calculator that implements the AtomsCalculators interface can be wrapped in an
[`AtomsCalculatorsModel`](@ref) to use it within NQCDynamics.jl.
The constructor requires the calculator object and an initial structure to determine
atomic species and cell information:

```julia
using NQCModels
using NQCBase
using AtomsCalculators

# Define your structure
atoms = Atoms([:H, :H])
cell = InfiniteCell()
R = rand(3, 2)
structure = NQCBase.Structure(atoms, R, cell)

# Wrap an AtomsCalculators-compatible calculator
# (here we use NQCModels' own Harmonic model as a demonstration)
calc = Harmonic()
model = AtomsCalculatorsModel(calc, structure)

# Now use it as you would any NQCModels model
potential(model, R)
derivative(model, R)
```

For a more complete worked example, including forward and back integration with
AtomsCalculators, see the [AtomsCalculators interoperability](@ref) page.
