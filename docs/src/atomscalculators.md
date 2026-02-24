# [AtomsCalculators interoperability](@id atomscalculators-interop)

[AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl) is a community standard
for defining calculators that operate on [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl)
systems. NQCModels.jl supports bidirectional interoperability with this interface.

## Using AtomsCalculators calculators in NQCDynamics.jl

Any calculator implementing the AtomsCalculators interface can be used in NQCDynamics.jl by
wrapping it in an [`AtomsCalculatorsModel`](@ref).

### Example: wrapping a custom AtomsCalculators calculator

```julia
using NQCModels
using NQCBase
using AtomsCalculators
using AtomsBase
using Unitful, UnitfulAtomic

# Define a minimal AtomsCalculators-compatible calculator
struct MyCalculator end

AtomsCalculators.energy_unit(::MyCalculator) = u"eV"
AtomsCalculators.length_unit(::MyCalculator) = u"Å"

function AtomsCalculators.potential_energy(sys::AbstractSystem, ::MyCalculator; kwargs...)
    # Example: harmonic potential based on first atom's position
    pos = position(sys, 1)
    return 0.5 * sum(ustrip.(u"Å", pos) .^ 2) * u"eV"
end

function AtomsCalculators.forces(sys::AbstractSystem, ::MyCalculator; kwargs...)
    n = length(sys)
    return [-ustrip.(u"Å", position(sys, i)) .* u"eV/Å" for i in 1:n]
end

# Create an NQCBase structure to define atoms and cell
atoms = Atoms([:H, :H])
cell = InfiniteCell()
R = rand(3, 2)
structure = NQCBase.Structure(atoms, R, cell)

# Wrap the AtomsCalculators calculator as an NQCModels model
model = AtomsCalculatorsModel(MyCalculator(), structure)

# Use the model just like any other NQCModels ClassicalModel
V = potential(model, R)
D = derivative(model, R)
```

The [`AtomsCalculatorsModel`](@ref) constructor can also accept an `AtomsBase.AbstractSystem`
directly as the second argument, which is useful when you already have an AtomsBase system:

```julia
using AtomsBase

# Construct an AtomsBase system
sys = isolated_system([
    Atom(:H, [0.0, 0.0, 0.0]u"Å"),
    Atom(:H, [0.0, 0.0, 0.74]u"Å"),
])

model = AtomsCalculatorsModel(MyCalculator(), sys)
```

## Using NQCModels models as AtomsCalculators calculators

All `ClassicalModel` types in NQCModels.jl automatically implement the AtomsCalculators
interface. This means you can pass any `ClassicalModel` wherever an AtomsCalculators
calculator is expected.

The following AtomsCalculators functions are supported for all `ClassicalModel`s:

- `AtomsCalculators.potential_energy(sys, model)` — returns the potential energy in hartree
- `AtomsCalculators.forces(sys, model)` — returns forces as a vector of `SVector{3}`, one per atom, in hartree/bohr
- `AtomsCalculators.virial(sys, model)` — returns the virial tensor (zero for non-periodic systems)

### Example: using an NQCModels model with AtomsCalculators

```julia
using NQCModels
using AtomsBase
using AtomsCalculators

# Create an NQCModels model
model = Harmonic()

# Create an AtomsBase system
sys = isolated_system([
    Atom(:H, [0.1, 0.0, 0.0]u"a0_au"),
])

# Use via the AtomsCalculators interface
E = AtomsCalculators.potential_energy(sys, model)
F = AtomsCalculators.forces(sys, model)
W = AtomsCalculators.virial(sys, model)
```

The returned energy has units of `u"hartree"` and forces have units of `u"hartree/a0_au"`,
consistent with NQCModels' internal convention of working in atomic units.

## Round-trip example

The following example demonstrates using an NQCModels `ClassicalModel` through the
AtomsCalculators interface and then wrapping it back in an `AtomsCalculatorsModel`,
confirming that the round-trip preserves the physical values:

```julia
using NQCModels
using NQCBase
using AtomsCalculators

# Original NQCModels model
original_model = Harmonic()

# Set up structure
atoms = Atoms([:H])
cell = InfiniteCell()
R = rand(3, 1)
structure = NQCBase.Structure(atoms, R, cell)

# Wrap the NQCModels model using AtomsCalculators, then re-wrap in AtomsCalculatorsModel
roundtrip_model = AtomsCalculatorsModel(original_model, structure)

# Both models should produce the same potential and forces
V_original = potential(original_model, R)
V_roundtrip = potential(roundtrip_model, R)

@assert isapprox(V_original, V_roundtrip, rtol=1e-10)

D_original = derivative(original_model, R)
D_roundtrip = derivative(roundtrip_model, R)

@assert isapprox(D_original, D_roundtrip, rtol=1e-10)
```

!!! note

    Unit conversions are handled automatically in both directions. NQCModels uses
    atomic units internally (hartree for energy, bohr for length), and all conversions
    to and from AtomsCalculators units are performed transparently.
