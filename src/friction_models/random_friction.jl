using Random: randn!

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

NQCModels.ndofs(model::RandomFriction) = model.ndofs