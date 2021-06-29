using Random

export RandomFriction

struct RandomFriction{M} <: AdiabaticFrictionModel
    model::M
end

potential(model::RandomFriction, R::AbstractMatrix) = potential(model.model, R)

function derivative!(model::RandomFriction, D::AbstractMatrix, R::AbstractMatrix)
    derivative!(model.model, D, R)
end

function friction!(::RandomFriction, F::AbstractMatrix, ::AbstractMatrix)
    randn!(F)
    F .= F'F
    F .= (F + F')/2
end
