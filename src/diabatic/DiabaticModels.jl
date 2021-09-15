
module DiabaticModels

using ..NonadiabaticModels: NonadiabaticModels
using Unitful: @u_str
using UnitfulAtomic: austrip

using Parameters: Parameters
using LinearAlgebra: Hermitian
using StaticArrays: SMatrix

"""
    DiabaticModel <: Model

`DiabaticModel`s should implement both `potential` and `derivative!`.
Further, each model must implement `nstates` which determines the size of the matrix.

`potential` must return a `Hermitian{SMatrix}` with `size = (nstates, nstates)`.

`derivative!` must fill an `AbstractMatrix{<:Hermitian}` with `size = (ndofs, natoms)`,
and each entry must have `size = (nstates, nstates)`.

# Example


```jldoctest
using StaticArrays
using LinearAlgebra

struct MyModel <: NonadiabaticModels.DiabaticModel end

nstates(::MyModel) = 2
ndofs(::MyModel) = 1

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
abstract type DiabaticModel <: NonadiabaticModels.Model end

"""
    LargeDiabaticModel <: DiabaticModel

Diabatic model too large for static arrays, instead uses normal julia arrays and must
implement the inplace `potential!`
"""
abstract type LargeDiabaticModel <: DiabaticModel end

"""
    DiabaticFrictionModel <: LargeDiabaticModel

`DiabaticFrictionModel`s are defined identically to the `DiabaticModel`.

However, they additionally allow for the calculation of electronic friction
internally from the diabatic potential after diagonalisation
and calculation of nonadiabatic couplings.

Use of this type leads to the allocation of extra arrays inside the `Calculator`
for the friction calculation.
"""
abstract type DiabaticFrictionModel <: LargeDiabaticModel end

function matrix_template(model::DiabaticModel, eltype)
    n = NonadiabaticModels.nstates(model)
    return SMatrix{n,n}(zeros(eltype, n, n))
end
function matrix_template(model::LargeDiabaticModel, eltype)
    zeros(eltype, NonadiabaticModels.nstates(model), NonadiabaticModels.nstates(model))
end

function vector_template(model::DiabaticModel, eltype)
    n = NonadiabaticModels.nstates(model)
    return SVector{n}(zeros(eltype, n))
end
function vector_template(model::LargeDiabaticModel, eltype)
    zeros(eltype, NonadiabaticModels.nstates(model))
end

function NonadiabaticModels.zero_derivative(model::DiabaticModel, R)
    [Hermitian(matrix_template(model, eltype(R))) for _ in CartesianIndices(R)]
end

"""
    potential(model::Model, R)

Obtain the potential for the current position `R`.

This is an allocating version of `potential!`.
"""
function NonadiabaticModels.potential(model::LargeDiabaticModel, R)
    V = Hermitian(matrix_template(model, eltype(R)))
    NonadiabaticModels.potential!(model, V, R)
    return V
end

include("double_well.jl")
export DoubleWell
include("tully_models.jl")
export TullyModelOne
export TullyModelTwo
export TullyModelThree
include("three_state_morse.jl")
export ThreeStateMorse
include("spin_boson.jl")
export OhmicSpectralDensity
export DebyeSpectralDensity
export SpinBoson
export BosonBath
include("1D_scattering.jl")
export Scattering1D
include("ouyang_models.jl")
export OuyangModelOne
include("gates_holloway_elbow.jl")
export GatesHollowayElbow
include("subotnik.jl")
export MiaoSubotnik

end # module
