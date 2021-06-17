"""
NonadiabaticModels define the potentials and derivatives that
govern the dynamics of the particles.
These can exist as analytic models or as interfaces to other codes. 
"""
module NonadiabaticModels

using Unitful, UnitfulAtomic
using LinearAlgebra
using Parameters
using ReverseDiff
using Requires
using SparseArrays
using NonadiabaticDynamicsBase

export Model
export AdiabaticModel
export DiabaticModel
export DiabaticFrictionModel
export AdiabaticFrictionModel

export potential!, potential
export derivative!, derivative
export friction!, friction

export energy, forces

"""
Top-level type for models.

# Implementation
When adding new models, this should not be directly subtyped. Instead, depending on
the intended functionality of the model, one of the child abstract types should be
subtyped.
If an appropriate type is not already available, a new abstract subtype should be created.
"""
abstract type Model end

Base.broadcastable(model::Model) = Ref(model)

"""
    AdiabaticModel <: Model

`AdiabaticModel`s should implement both `potential!` and `derivative!`.

`potential!` must fill an `AbstractVector` with `length = 1`.

`derivative!` must fill an `AbstractMatrix` with `size = (DoFs, atoms)`.

# Example

```jldoctest
struct MyModel{P} <: NonadiabaticModels.AdiabaticModel
    param::P
end

NonadiabaticModels.potential!(model::MyModel, V, R) = V .= model.param*sum(R.^2)
NonadiabaticModels.derivative!(model::MyModel, D, R) = D .= model.param*2R

model = MyModel(10)

NonadiabaticModels.potential(model, [1 2; 3 4])

# output

1-element Vector{Int64}:
 300

```
"""
abstract type AdiabaticModel <: Model end

"""
    DiabaticModel <: Model

`DiabaticModel`s should implement both `potential!` and `derivative!`.
Further, each model must have the field `n_states`, which determines the size of the matrix.

`potential!` must fill a `Hermitian` with `size = (n_states, n_states)`.

`derivative!` must fill an `AbstractMatrix{<:Hermitian}` with `size = (DoFs, atoms)`,
and each entry must have `size = (n_states, n_states)`.

# Example

```jldoctest
struct MyModel <: NonadiabaticModels.DiabaticModel
    n_states::Int # Mandatory `n_states` field.
    MyModel() = new(2)
end

function NonadiabaticModels.potential!(::MyModel, V, R) 
    V[1,1] = sum(R)
    V[2,2] = -sum(R)
    V.data[1,2] = 1 # Must use `.data` to set off-diagonal elements of `Hermitian`.
    return V
end

function NonadiabaticModels.derivative!(::MyModel, D, R)
    for d in eachindex(D)
        D[d][1,1] = 1
        D[d][2,2] = -1
    end
    return D
end

model = MyModel()
NonadiabaticModels.potential(model, [1 2; 3 4])

# output

2×2 LinearAlgebra.Hermitian{Int64, Matrix{Int64}}:
 10    1
  1  -10
```
"""
abstract type DiabaticModel <: Model end

"""
    SparseDiabaticModel <: DiabaticModel

Same as diabatic model but uses sparse arrays to store potentials and derivatives.
"""
abstract type SparseDiabaticModel <: DiabaticModel end

"""
    DiabaticFrictionModel <: DiabaticModel

`DiabaticFrictionModel`s are defined identically to the `DiabaticModel`.

However, they additionally allow for the calculation of electronic friction
internally from the diabatic potential after diagonalisation
and calculation of nonadiabatic couplings.

Use of this type leads to the allocation of extra arrays inside the `Calculator`
for the friction calculation.
"""
abstract type DiabaticFrictionModel <: DiabaticModel end

"""
    AdiabaticFrictionModel <: AdiabaticModel

`AdiabaticFrictionModel`s must implement `potential!`, `derivative!`, and `friction!`

`potential!` and `friction!` should be the same as for the `AdiabaticModel`.

`friction!` must fill an `AbstractMatrix` with `size = (DoFs*atoms, DoFs*atoms)`.
"""
abstract type AdiabaticFrictionModel <: AdiabaticModel end

"""
    potential!(model::Model, V, R::AbstractMatrix)

Fill `V` with the electronic potential of the system as a function of the positions `R`.

This must be implemented for all models.
"""
function potential! end

"""
    derivative!(model::Model, D, R::AbstractMatrix)

Fill `D` with the derivative of the electronic potential as a function of the positions `R`.

This must be implemented for all models.
"""
function derivative! end

"ReverseDiff fallback derivative for adiabatic models."
function derivative!(model::AdiabaticModel, D, R)
    f(R) = potential(model, R)
    D .= ReverseDiff.gradient(f, R)
end

"""
    friction!(model::AdiabaticFrictionModel, F, R:AbstractMatrix)

Fill `F` with the electronic friction as a function of the positions `R`.

This need only be implemented for `AdiabaticFrictionModel`s.
"""
function friction! end

zero_potential(::AdiabaticModel, R) = zeros(eltype(R), 1)
zero_potential(model::DiabaticModel, R) = Hermitian(zeros(eltype(R), model.n_states, model.n_states))
zero_potential(model::SparseDiabaticModel, R) = Hermitian(spzeros(eltype(R), model.n_states, model.n_states))

zero_derivative(::AdiabaticModel, R) = zero(R)
zero_derivative(model::DiabaticModel, R) = [zero_potential(model, R) for _ in CartesianIndices(R)]

zero_friction(::AdiabaticFrictionModel, R) = zeros(eltype(R), length(R), length(R))

"""
    potential(model::Model, R)

Obtain the potential for the current position `R`.

This is an allocating version of `potential!`.
"""
function potential(model::Model, R)
    V = zero_potential(model, R)
    potential!(model, V, R)
    return V
end
energy(model::AdiabaticModel, R) = potential(model, R)[1]

"""
    derivative(model::Model, R)

Obtain the derivative of the potential for the position `R`.

This is an allocating version of `derivative!`.
"""
function derivative(model::Model, R)
    D = zero_derivative(model, R)
    derivative!(model, D, R)
    return D
end
forces(model::AdiabaticModel, R) = -derivative(model, R)

"""
    friction(model::Model, R)

Obtain the friction for the current position `R`.

This is an allocating version of `friction!`.
"""
function friction(model::Model, R)
    F = zero_friction(model, R)
    friction!(model, F, R)
    return F
end

include("plot.jl")

include("adiabatic/free.jl")
include("adiabatic/harmonic.jl")
include("adiabatic/diatomic_harmonic.jl")
include("adiabatic/darling_holloway_elbow.jl")

include("friction/constant_friction.jl")
include("friction/random_friction.jl")

include("diabatic/double_well.jl")
include("diabatic/tully_models.jl")
include("diabatic/three_state_morse.jl")
include("diabatic/spin_boson.jl")
include("diabatic/1D_scattering.jl")
include("diabatic/ouyang_models.jl")
include("diabatic/gates_holloway_elbow.jl")
include("diabatic/subotnik.jl")

function __init__()
    @require JuLIP="945c410c-986d-556a-acb1-167a618e0462" @eval include("adiabatic/julip.jl")
end

end # module
