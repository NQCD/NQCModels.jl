using Random

export RandomFriction

struct RandomFriction{M} <: AdiabaticFrictionModel
    model::M
end

function potential!(model::RandomFriction, V::AbstractVector, R::AbstractMatrix)
    potential!(model.model, V, R)
end

function derivative!(model::RandomFriction, D::AbstractMatrix, R::AbstractMatrix)
    derivative!(model.model, D, R)
end

function friction!(::RandomFriction, F::AbstractMatrix, ::AbstractMatrix)
    randn!(F)
    F .= F'F
    F .= (F + F')/2
end
