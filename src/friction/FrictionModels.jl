
module FrictionModels

using ..NQCModels: NQCModels
using ..AdiabaticModels: AdiabaticModel

export friction, friction!

"""
    AdiabaticFrictionModel <: AdiabaticModel

`AdiabaticFrictionModel`s must implement `potential!`, `derivative!`, and `friction!`

`potential!` and `friction!` should be the same as for the `AdiabaticModel`.

`friction!` must fill an `AbstractMatrix` with `size = (ndofs*natoms, ndofs*natoms)`.
"""
abstract type AdiabaticFrictionModel <: AdiabaticModel end

"""
    ElectronicFrictionProvider

Abstract type for defining models that provide electronic friction only.

Subtypes of this must implement `friction!` and `ndofs`.
Subtypes of this should implement `get_friction_matrix` for functionality with `Subsystem`s and `CompositeModel`s.
friction! must act on a square Matrix{<:Number} of size ndofs * length(atoms.masses). 

Units of friction are mass-weighted, and the atomic unit of friction is: [E_h / ħ / m_e]
Convert common friction units such as ps^-1 or meV ps Å^-2 using `UnitfulAtomic.austrip`. 
If atomic masses are required to calculate friction in your ElectronicFrictionProvider (e.g. for Isotope support), the atomic masses to use should be included as a type field. 

"""
abstract type ElectronicFrictionProvider end


"""
    DiagonalFriction
    
Abstract model type which yields diagonal friction matrices. This allows some integrators to fall back to simpler routines and save time. 

Subtypes of this must implement `friction!` and `ndofs`.
Subtypes of this should implement `get_friction_matrix` for functionality with `Subsystem`s and `CompositeModel`s.
friction! must act on a square Matrix{<:Number} of size ndofs * length(atoms.masses). 

Units of friction are mass-weighted, and the atomic unit of friction is: [E_h / ħ / m_e]
Convert common friction units such as ps^-1 or meV ps Å^-2 using `UnitfulAtomic.austrip`. 
If atomic masses are required to calculate friction in your ElectronicFrictionProvider (e.g. for Isotope support), the atomic masses to use should be included as a type field. 
"""
abstract type DiagonalFriction <: ElectronicFrictionProvider end

"""
    TensorialFriction
    
Abstract model type which yields full-rank friction matrices. 

If atomic masses are required to calculate friction in your ElectronicFrictionProvider (e.g. for Isotope support), the atomic masses to use should be included as a type field. 
A `friction_atoms` field should be included - This indicates that the friction matrix is only partially returned. 

Subtypes of this must implement `get_friction_matrix(model::ACEdsODF, R::AbstractMatrix, friction_atoms::AbstractVector) --> ::AbstractMatrix{eltype(R)}`.
Subtypes of this must implement `ndofs(m::TensorialFriction) --> ::Int`.

Units of friction are mass-weighted, and the atomic unit of friction is: [E_h / ħ / m_e]
Convert common friction units such as ps^-1 or meV ps Å^-2 using `UnitfulAtomic.austrip`. 

A system-size friction matrix is obtained with `friction(model, positions)`, or `friction!(model, friction_matrix, positions)`
"""
abstract type TensorialFriction <: ElectronicFrictionProvider end

function get_friction_matrix(::TensorialFriction, ::AbstractMatrix)
    @error "Implement this function for your method"
end
NQCModels.ndofs(model::ElectronicFrictionProvider) = model.ndofs
NQCModels.dofs(model::ElectronicFrictionProvider) = 1:model.ndofs

"""
    ConstantFriction

Friction model which returns a constant value for all positions. Use with a single value, a vector of diagonal values or a full-size Matrix

"""
struct ConstantFriction{T} <: ElectronicFrictionProvider
    ndofs::Int
    γ::T
end

"""
    RandomFriction <: ElectronicFrictionProvider

Provide a random positive semi-definite matrix of friction values.
Used mostly for testing and examples.
"""
struct RandomFriction <: ElectronicFrictionProvider
    ndofs::Int
end

function friction!(::RandomFriction, F::AbstractMatrix, ::AbstractMatrix)
    randn!(F)
    F .= F'F
    F .= (F + F')/2
end

"""
    friction!(model::AdiabaticFrictionModel, F, R:AbstractMatrix)

Fill `F` with the electronic friction as a function of the positions `R`.

`F` must be a square Matrix{<:Number} of size ndofs * n_atoms. 

Units of friction are mass-weighted, and the atomic unit of friction is: [E_h / ħ / m_e]

Convert common friction units such as ps^-1 or meV ps Å^-2 using `UnitfulAtomic.austrip`. 

This need only be implemented for `AdiabaticFrictionModel`s.
"""
function friction! end

"""
    friction(model::Model, R)

Obtain the friction for the current position `R`.

Yarr, Friction be a square Matrix{<:Number} of size ndofs * n_atoms. 

Units of friction are mass-weighted, and the atomic unit of friction is: [E_h / ħ / m_e]

This is an allocating version of `friction!`.

"""
function friction(model::AdiabaticFrictionModel, R)
    F = zero_friction(model, R)
    friction!(model, F, R)
    return F
end

zero_friction(::AdiabaticFrictionModel, R) = zeros(eltype(R), length(R), length(R))

include("diagonal_friction.jl")
export DiagonalFriction, LDFAFriction, ElectronDensityProvider
include("tensorial_friction.jl")
export get_friction_matrix, TensorialFriction

include("composite_friction_model.jl")
export CompositeFrictionModel
include("ase_friction_interface.jl")
export ASEFrictionProvider

# include("constant_friction.jl")
export ConstantFriction
# include("random_friction.jl")
export RandomFriction

end # module
