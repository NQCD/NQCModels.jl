
module FrictionModels

using ..NQCModels: NQCModels
using ..ClassicalModels: ClassicalModel

export friction, friction!

"""
    ClassicalFrictionModel <: ClassicalModel

`ClassicalFrictionModel`s must implement `potential!`, `derivative!`, and `friction!`

`potential!` and `friction!` should be the same as for the `ClassicalModel`.

`friction!` must fill an `AbstractMatrix` with `size = (ndofs*natoms, ndofs*natoms)`.
"""
abstract type ClassicalFrictionModel <: ClassicalModel end

"""
    ElectronicFrictionProvider

Abstract type for defining models that provide electronic friction only.
Subtypes of this should implement `friction!` and `ndofs`.
"""
abstract type ElectronicFrictionProvider end

"""
    friction!(model::ClassicalFrictionModel, F, R:AbstractMatrix)

Fill `F` with the electronic friction as a function of the positions `R`.

This need only be implemented for `ClassicalFrictionModel`s.
"""
function friction! end

"""
    friction(model::Model, R)

Obtain the friction for the current position `R`.

This is an allocating version of `friction!`.

"""
function friction(model::ClassicalFrictionModel, R)
    F = zero_friction(model, R)
    friction!(model, F, R)
    return F
end

zero_friction(::ClassicalFrictionModel, R) = zeros(eltype(R), length(R), length(R))

NQCModels.dofs(model::ElectronicFrictionProvider) = 1:model.ndofs
NQCModels.ndofs(model::ElectronicFrictionProvider) = model.ndofs

include("composite_friction_model.jl")
export CompositeFrictionModel
include("ase_friction_interface.jl")
export ASEFrictionProvider

include("constant_friction.jl")
export ConstantFriction
include("random_friction.jl")
export RandomFriction

end # module
