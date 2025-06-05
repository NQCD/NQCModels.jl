using LinearAlgebra: diagind

"""
    ConstantFriction

Friction model which returns a constant value for all positions. Use with a single value, a vector of diagonal values or a full-size Matrix

"""
struct ConstantFriction{T} <: ElectronicFrictionProvider
    ndofs::Int
    γ::T
end

function friction!(model::ConstantFriction, F::AbstractMatrix, ::AbstractMatrix)
    F[diagind(F)] .= model.γ
end

NQCModels.ndofs(model::ConstantFriction) = model.ndofs