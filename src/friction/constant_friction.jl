using LinearAlgebra: diagind

export ConstantFriction

struct ConstantFriction{M,T} <: AdiabaticFrictionModel
    model::M
    γ::T
end

function potential!(model::ConstantFriction, V::AbstractVector, R::AbstractMatrix)
    potential!(model.model, V, R)
end

function derivative!(model::ConstantFriction, D::AbstractMatrix, R::AbstractMatrix)
    derivative!(model.model, D, R)
end

function friction!(model::ConstantFriction, F::AbstractMatrix, ::AbstractMatrix)
    F[diagind(F)] .= model.γ
end
