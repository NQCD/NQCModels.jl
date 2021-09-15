
module AdiabaticModels

using ..NonadiabaticModels: NonadiabaticModels
using Requires: Requires
using Zygote: Zygote
using Parameters: Parameters

using Unitful: @u_str, ustrip
using UnitfulAtomic: austrip, auconvert

"""
    AdiabaticModel <: Model

`AdiabaticModel`s should implement both `potential` and `derivative!`.

`potential(model, R)` must return the value of the potential evaluated at `R`

`derivative!(model, D::AbstractMatrix, R)` must fill `D` with `size = (ndofs, natoms)`.

# Example

```jldoctest
struct MyModel{P} <: NonadiabaticModels.AdiabaticModel
    param::P
end

ndofs(::MyModel) = 2

NonadiabaticModels.potential(model::MyModel, R) = model.param*sum(R.^2)
NonadiabaticModels.derivative!(model::MyModel, D, R) = D .= model.param*2R

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
