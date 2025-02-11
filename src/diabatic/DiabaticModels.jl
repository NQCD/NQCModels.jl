
"""
    DiabaticModels

Models defined within this module subtype the [`DiabaticModel`](@ref) and provide
potentials as Hermitian matrices and derivatives as arrays of Hermitian matrices.
"""
module DiabaticModels

using ..NQCModels: NQCModels
using Unitful: @u_str
using UnitfulAtomic: austrip

using Parameters: Parameters
using LinearAlgebra: LinearAlgebra, Hermitian
using StaticArrays: SMatrix, SVector

"""
    DiabaticModel <: Model

`DiabaticModel`s are used when a system has multiple electronic states that are
presented in the diabatic representation. This is the case for the majority
of model systems.

# Implementation

`DiabaticModel`s should implement:
* `potential(model, R)`
* `derivative!(model, D, R)`
* `nstates(model)`
* `ndofs(model)`

# Example

In this example we create a simple 2 state, 1 dimensional diabatic model `MyModel`.
As noted above, we implement the 4 relevant functions then evaluate the potential.
Since this is a 1D model the argument `R` accepts a `Real` value.

```jldoctest
using StaticArrays: SMatrix
using LinearAlgebra: Hermitian

struct MyModel <: NQCModels.DiabaticModels.DiabaticModel end

NQCModels.nstates(::MyModel) = 2
NQCModels.ndofs(::MyModel) = 1

function NQCModels.potential(::MyModel, R::Real) 
    V11 = R
    V22 = -R
    V12 = 1
    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NQCModels.derivative!(::MyModel, D, R::Real)
    return Hermitian(SMatrix{2,2}(1, 0, 0, 1))
end

model = MyModel()
NQCModels.potential(model, 10)

# output

2×2 Hermitian{Int64, SMatrix{2, 2, Int64, 4}}:
 10    1
  1  -10
```
"""
abstract type DiabaticModel <: NQCModels.Model end

"""
    LargeDiabaticModel <: DiabaticModel

Same as the `DiabaticModels` but uses normal Julia arrays instead of StaticArrays and must
implement the inplace `potential!` rather than `potential`.
This is useful when `nstates` is very large and StaticArrays are no longer efficient.
"""
abstract type LargeDiabaticModel <: DiabaticModel end

function NQCModels.derivative!(model::LargeDiabaticModel, D, R::AbstractMatrix)
    if NQCModels.ndofs(model) == 1
        if size(R, 2) == 1
            NQCModels.derivative!(model, D[1], R[1])
            return D
        else
            NQCModels.derivative!(model, view(D, 1, :), view(R, 1, :))
            return D
        end
    elseif size(R, 2) == 1
        NQCModels.derivative!(model, view(D, :, 1), view(R, :, 1))
        return D
    else
        throw(MethodError(NQCModels.derivative!, (model, D, R)))
    end
end

"""
    DiabaticFrictionModel <: LargeDiabaticModel

These models are defined identically to the `LargeDiabaticModel` but
allocate extra temporary arrays when used with `NQCDynamics.jl`.

This allows for the calculation of electronic friction
internally from the diabatic potential after diagonalisation
and calculation of nonadiabatic couplings.
"""
abstract type DiabaticFrictionModel <: LargeDiabaticModel end

function matrix_template(model::DiabaticModel, eltype)
    n = NQCModels.nstates(model)
    return SMatrix{n,n}(zeros(eltype, n, n))
end
function matrix_template(model::LargeDiabaticModel, eltype)
    zeros(eltype, NQCModels.nstates(model), NQCModels.nstates(model))
end

function vector_template(model::DiabaticModel, eltype)
    n = NQCModels.nstates(model)
    return SVector{n}(zeros(eltype, n))
end
function vector_template(model::LargeDiabaticModel, eltype)
    zeros(eltype, NQCModels.nstates(model))
end

function NQCModels.zero_derivative(model::DiabaticModel, R::AbstractMatrix)
    [Hermitian(matrix_template(model, eltype(R))) for _ in CartesianIndices(R)]
end

function NQCModels.potential(model::LargeDiabaticModel, R::AbstractMatrix)
    V = Hermitian(matrix_template(model, eltype(R)))
    NQCModels.potential!(model, V, R)
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
include("ananth_models.jl")
export AnanthModelOne
export AnanthModelTwo
include("erpenbeck_thoss.jl")
export ErpenbeckThoss
include("widebandbath.jl")
export WideBandBath

include("wide_band_bath_discretisation.jl")
export TrapezoidalRule
export WindowedTrapezoidalRule
export WindowedTrapezoidalRule2
export WindowedTrapezoidalRule3
export WindowedTrapezoidalRule4
export WindowedTrapezoidalRule5
export ShenviGaussLegendre
export ReferenceGaussLegendre
export FullGaussLegendre
export GapGaussLegendre
export GapTrapezoidalRule

include("anderson_holstein.jl")
export AndersonHolstein

include("adiabatic_state_selector.jl")
export AdiabaticStateSelector

end # module
