"""
NonadiabaticModels define the potentials and derivatives that
govern the dynamics of the particles.
These can exist as analytic models or as interfaces to other codes. 
"""
module NonadiabaticModels

using Unitful, UnitfulAtomic
using LinearAlgebra
using Parameters
using Requires
using StaticArrays
using NonadiabaticDynamicsBase
using Zygote

export Model
export AdiabaticModel
export DiabaticModel
export LargeDiabaticModel
export DiabaticFrictionModel
export AdiabaticFrictionModel

export potential!, potential
export derivative!, derivative
export friction!, friction

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

`AdiabaticModel`s should implement both `potential` and `derivative!`.

`potential(model, R)` must return the value of the potential evaluated at `R`

`derivative!(model, D::AbstractMatrix, R)` must fill `D` with `size = (DoFs, atoms)`.

# Example

```jldoctest
struct MyModel{P} <: NonadiabaticModels.AdiabaticModel
    param::P
end

NonadiabaticModels.potential(model::MyModel, R) = model.param*sum(R.^2)
NonadiabaticModels.derivative!(model::MyModel, D, R) = D .= model.param*2R

model = MyModel(10)

NonadiabaticModels.potential(model, [1 2; 3 4])

# output

300

```
"""
abstract type AdiabaticModel <: Model end

"""
    DiabaticModel <: Model

`DiabaticModel`s should implement both `potential` and `derivative!`.
Further, each model must have the field `n_states`, which determines the size of the matrix.

`potential` must return a `Hermitian{SMatrix}` with `size = (n_states, n_states)`.

`derivative!` must fill an `AbstractMatrix{<:Hermitian}` with `size = (DoFs, atoms)`,
and each entry must have `size = (n_states, n_states)`.

# Example


```jldoctest
using StaticArrays
using LinearAlgebra

struct MyModel <: NonadiabaticModels.DiabaticModel
    n_states::Int # Mandatory `n_states` field.
    MyModel() = new(2)
end

function NonadiabaticModels.potential(::MyModel, R) 
    V11 = sum(R)
    V22 = -sum(R)
    V12 = 1
    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NonadiabaticModels.derivative!(::MyModel, D, R)
    for d in eachindex(D)
        D[d] = Hermitian(SMatrix{2,2}(1, 0, 0, 1))
    end
    return D
end

model = MyModel()
NonadiabaticModels.potential(model, [1 2; 3 4])

# output

2×2 Hermitian{Int64, SMatrix{2, 2, Int64, 4}}:
 10    1
  1  -10
```
"""
abstract type DiabaticModel <: Model end

"""
    LargeDiabaticModel <: DiabaticModel

Diabatic model too large for static arrays, instead uses normal julia arrays and must
implement the inplace `potential!`
"""
abstract type LargeDiabaticModel <: DiabaticModel end

"""
    DiabaticFrictionModel <: DiabaticModel

`DiabaticFrictionModel`s are defined identically to the `DiabaticModel`.

However, they additionally allow for the calculation of electronic friction
internally from the diabatic potential after diagonalisation
and calculation of nonadiabatic couplings.

Use of this type leads to the allocation of extra arrays inside the `Calculator`
for the friction calculation.
"""
abstract type DiabaticFrictionModel <: LargeDiabaticModel end

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

"""
    friction!(model::AdiabaticFrictionModel, F, R:AbstractMatrix)

Fill `F` with the electronic friction as a function of the positions `R`.

This need only be implemented for `AdiabaticFrictionModel`s.
"""
function friction! end


function matrix_template(model::DiabaticModel, eltype)
    n = convert(Int, model.n_states)
    return SMatrix{n,n}(zeros(eltype, n, n))
end
matrix_template(model::LargeDiabaticModel, eltype) = zeros(eltype, model.n_states, model.n_states)

function vector_template(model::DiabaticModel, eltype)
    n = convert(Int, model.n_states)
    return SVector{n}(zeros(eltype, n))
end
vector_template(model::LargeDiabaticModel, eltype) = zeros(eltype, model.n_states)

zero_derivative(::AdiabaticModel, R) = zero(R)
zero_derivative(model::DiabaticModel, R) = [Hermitian(matrix_template(model, eltype(R))) for _ in CartesianIndices(R)]

zero_friction(::AdiabaticFrictionModel, R) = zeros(eltype(R), length(R), length(R))

"""
    potential(model::Model, R)

Obtain the potential for the current position `R`.

This is an allocating version of `potential!`.
"""
function potential(model::LargeDiabaticModel, R)
    V = Hermitian(matrix_template(model, eltype(R)))
    potential!(model, V, R)
    return V
end

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
include("adiabatic/ase_interface.jl")

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
