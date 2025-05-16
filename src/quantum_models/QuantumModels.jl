
"""
    QuantumModels

Models defined within this module subtype the [`QuantumModel`](@ref) and provide
potentials as Hermitian matrices and derivatives as arrays of Hermitian matrices.
"""
module QuantumModels

using ..NQCModels: NQCModels
using Unitful: @u_str
using UnitfulAtomic: austrip

using Parameters: Parameters
using LinearAlgebra: LinearAlgebra, Hermitian
using StaticArrays: SMatrix, SVector

"""
    QuantumModel <: Model

`QuantumModel`s are used when a system has multiple electronic states and the dynamics
of the system are propagated by a Hamiltonian in the diabatic representation. This is 
the case for the majority of model systems.

# Implementation

`QuantumModel`s should implement:
* `potential!(model, R)`
* `derivative!(model, D, R)`
* `nstates(model)`
* `ndofs(model)`

# Example

In this example we create a simple 2 state, 1 dimensional quantum model `MyModel`.
As noted above, we implement the 4 relevant functions then evaluate the potential.
Potential and Derivative functions take in positions as abstract matrices, since 
this is a 1D model the argument `R` should be a `Real` wrapped in a 1x1 matrix. It is
recommended that you use the hcat() function to do this.

```jldoctest
using StaticArrays: SMatrix
using LinearAlgebra: Hermitian

struct MyModel <: NQCModels.QuantumModels.QuantumModel end

NQCModels.nstates(::MyModel) = 2
NQCModels.ndofs(::MyModel) = 1

function NQCModels.potential!(::MyModel, V::Hermitian, R::AbstractMatrix) 
    V11 = R[1]
    V22 = -R[1]
    V12 = 1
    V = Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NQCModels.derivative!(::MyModel, D::Hermitian, R::AbstractMatrix)
    D = Hermitian(SMatrix{2,2}(1, 0, 0, 1))
end

model = MyModel()
V = Hermitian(zeros(2,2))
NQCModels.potential!(model, V, hcat(10))

# output

2×2 Hermitian{Int64, SMatrix{2, 2, Int64, 4}}:
 10    1
  1  -10
```
"""
abstract type QuantumModel <: NQCModels.Model end

function NQCModels.derivative!(model::QuantumModel, D, R::AbstractMatrix)
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
    QuantumFrictionModel <: QuantumModel

These models are defined identically to a typical `QuantumModel` but
allocate extra temporary arrays when used with `NQCDynamics.jl`.

This allows for the calculation of electronic friction
internally from the diabatic potential after diagonalisation
and calculation of nonadiabatic couplings.

When a molecular dynamics with electronic friction simulation is 
set up in NQCDynamics, the `QuantumFrictionModel` is paired with a
`FrictionEvaluationMethod` in order to calculate the electronic 
friction from the potential and derivative matrices.
"""
abstract type QuantumFrictionModel <: QuantumModel end

function matrix_template(model::QuantumModel, eltype)
    n = NQCModels.nstates(model)
    return zeros(eltype, n, n)
end

function vector_template(model::QuantumModel, eltype)
    n = NQCModels.nstates(model)
    return zeros(eltype, n)
end

function NQCModels.zero_derivative(model::QuantumModel, R::AbstractMatrix)
    return [Hermitian(matrix_template(model, eltype(R))) for _ in CartesianIndices(R)]
end

function NQCModels.potential(model::QuantumModel, R::AbstractMatrix)
    V = Hermitian(matrix_template(model, eltype(R)))
    NQCModels.potential!(model, V, R)
    return V
end


include("avoided_crossing_models/ananth_models.jl")
export AnanthModelOne
export AnanthModelTwo
include("avoided_crossing_models/tully_models.jl")
export TullyModelOne
export TullyModelTwo
export TullyModelThree

include("bosonic_models/double_well.jl")
export DoubleWell
include("bosonic_models/spin_boson.jl")
export OhmicSpectralDensity
export DebyeSpectralDensity
export SpinBoson
export BosonBath

include("explicit_bath_models/anderson_holstein.jl")
export AndersonHolstein

include("molecular_state_models/erpenbeck_thoss.jl")
export ErpenbeckThoss
include("molecular_state_models/gates_holloway_elbow.jl")
export GatesHollowayElbow
include("molecular_state_models/subotnik.jl")
export MiaoSubotnik
include("molecular_state_models/three_state_morse.jl")
export ThreeStateMorse

include("quantum_friction_models/1D_scattering.jl")
export Scattering1D
include("quantum_friction_models/ouyang_models.jl")
export OuyangModelOne
include("quantum_friction_models/widebandbath.jl")
export WideBandBath

include("adiabatic_state_selector.jl")
export AdiabaticStateSelector

end # module
