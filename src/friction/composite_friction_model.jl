using NQCModels: ndofs

"""
    CompositeFrictionModel{M,F} <: AdiabaticFrictionModel

Combine an `AdiabaticModel` with an `ElectronicFrictionProvider`.

This allows for arbitrary composition of potentials and friction providers,
such that any adiabatic model can be augmented with any form of electronic friction.
"""
struct CompositeFrictionModel{M<:AdiabaticModel,F} <: AdiabaticFrictionModel
    pes_model::M
    friction_model::F
end

function NQCModels.ndofs(model::CompositeFrictionModel)
    model_dofs = ndofs(model.pes_model)
    friction_dofs = ndofs(model.friction_model)

    if model_dofs == friction_dofs
        return model_dofs
    else
        throw(ArgumentError("Dimensionality of PES and friction models does not match.")) 
    end
end

function NQCModels.potential(model::CompositeFrictionModel, R::AbstractMatrix)
    NQCModels.potential(model.pes_model, R)
end

function NQCModels.derivative!(model::CompositeFrictionModel, D, R::AbstractMatrix)
    NQCModels.derivative!(model.pes_model, D, R)
end

function friction!(model::CompositeFrictionModel, F, R)
    friction!(model.friction_model, F, R)
end
