using LinearAlgebra: diagind

struct ConstantFriction{T} <: ElectronicFrictionProvider
    ndofs::Int
    γ::T
end

function friction!(model::ConstantFriction, F::AbstractMatrix, ::AbstractMatrix)
    F[diagind(F)] .= model.γ
end

NonadiabaticModels.ndofs(model::ConstantFriction) = model.ndofs
