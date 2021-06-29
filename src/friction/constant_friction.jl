using LinearAlgebra: diagind

export ConstantFriction

struct ConstantFriction{M,T} <: AdiabaticFrictionModel
    model::M
    γ::T
end

potential(model::ConstantFriction, R::AbstractMatrix) = potential(model.model, R)

function derivative!(model::ConstantFriction, D::AbstractMatrix, R::AbstractMatrix)
    derivative!(model.model, D, R)
end

function friction!(model::ConstantFriction, F::AbstractMatrix, ::AbstractMatrix)
    F[diagind(F)] .= model.γ
end
