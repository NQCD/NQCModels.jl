
"""
    AdiabaticModels

All models defined within this module have only a single electronic state
and return potentials as scalars and derivatives as simple arrays.

The central abstract type is the [`AdiabaticModel`](@ref) which all models should subtype.
"""
module AdiabaticModels

using ..NQCModels: NQCModels
using Requires: Requires
using Zygote: Zygote
using Parameters: Parameters

using Unitful: @u_str, ustrip
using UnitfulAtomic: austrip, auconvert

"""
    AdiabaticModel <: Model

`AdiabaticModel`s represent the familiar potentials from classical molecular dynamics
where the potential is a simple function of position.

# Implementation

`AdiabaticModel`s should implement:
* `potential(model, R)`
* `derivative!(model, D, R)`
* `ndofs(model)`

# Example

This example creates a 2 dimensional adiabatic model `MyModel`.
We implement the 3 compulsory functions then evaluate the potential.
Here the argument `R` is an `AbstractMatrix` since this is a 2D model
that can accept multiple atoms.

```jldoctest
struct MyModel{P} <: NQCModels.AdiabaticModels.AdiabaticModel
    param::P
end

NQCModels.ndofs(::MyModel) = 2

NQCModels.potential(model::MyModel, R::AbstractMatrix) = model.param*sum(R.^2)
NQCModels.derivative!(model::MyModel, D, R::AbstractMatrix) = D .= model.param*2R

model = MyModel(10)

NQCModels.potential(model, [1 2; 3 4])

# output

300

```
"""
abstract type AdiabaticModel <: NQCModels.Model end

NQCModels.nstates(::AdiabaticModel) = 1

NQCModels.zero_derivative(::AdiabaticModel, R) = zero(R)

include("free.jl")
export Free
include("harmonic.jl")
export Harmonic
include("diatomic_harmonic.jl")
export DiatomicHarmonic
include("darling_holloway_elbow.jl")
export DarlingHollowayElbow
include("ase_interface.jl")
export AdiabaticASEModel

function __init__()
    Requires.@require JuLIP="945c410c-986d-556a-acb1-167a618e0462" @eval include("julip.jl")
end

end # module
