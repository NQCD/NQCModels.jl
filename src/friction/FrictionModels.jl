
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
Subtypes of this should implement `friction!` and `ndofs`.
"""
abstract type ElectronicFrictionProvider end

"""
    friction!(model::AdiabaticFrictionModel, F, R:AbstractMatrix)

Fill `F` with the electronic friction as a function of the positions `R`.

This need only be implemented for `AdiabaticFrictionModel`s.
"""
function friction! end

"""
    friction(model::Model, R)

Obtain the friction for the current position `R`.

This is an allocating version of `friction!`.
"""
function friction(model::AdiabaticFrictionModel, R)
    F = zero_friction(model, R)
    friction!(model, F, R)
    return F
end

zero_friction(::AdiabaticFrictionModel, R) = zeros(eltype(R), length(R), length(R))

include("composite_friction_model.jl")
export CompositeFrictionModel
include("ase_friction_interface.jl")
export ASEFrictionProvider

include("constant_friction.jl")
export ConstantFriction
include("random_friction.jl")
export RandomFriction

end # module
