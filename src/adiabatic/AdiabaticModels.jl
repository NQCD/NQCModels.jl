
module AdiabaticModels

using ..NonadiabaticModels: NonadiabaticModels
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
struct MyModel{P} <: NonadiabaticModels.AdiabaticModels.AdiabaticModel
    param::P
end

NonadiabaticModels.ndofs(::MyModel) = 2

NonadiabaticModels.potential(model::MyModel, R::AbstractMatrix) = model.param*sum(R.^2)
NonadiabaticModels.derivative!(model::MyModel, D, R::AbstractMatrix) = D .= model.param*2R

model = MyModel(10)

NonadiabaticModels.potential(model, [1 2; 3 4])

# output

300

```
"""
abstract type AdiabaticModel <: NonadiabaticModels.Model end

NonadiabaticModels.nstates(::AdiabaticModel) = 1

NonadiabaticModels.zero_derivative(::AdiabaticModel, R) = zero(R)

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
