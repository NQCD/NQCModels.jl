using LinearAlgebra: diagind

struct ConstantFriction{M,T} <: AdiabaticFrictionModel
    model::M
    γ::T
end

function NonadiabaticModels.potential(model::ConstantFriction, R::AbstractMatrix)
    NonadiabaticModels.potential(model.model, R)
end

function NonadiabaticModels.derivative!(model::ConstantFriction, D::AbstractMatrix, R::AbstractMatrix)
    NonadiabaticModels.derivative!(model.model, D, R)
end

function friction!(model::ConstantFriction, F::AbstractMatrix, ::AbstractMatrix)
    F[diagind(F)] .= model.γ
end

NonadiabaticModels.ndofs(model::ConstantFriction) = NonadiabaticModels.ndofs(model.model)
