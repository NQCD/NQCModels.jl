using Random: randn!

struct RandomFriction{M} <: AdiabaticFrictionModel
    model::M
end

function NonadiabaticModels.potential(model::RandomFriction, R::AbstractMatrix)
    NonadiabaticModels.potential(model.model, R)
end

function NonadiabaticModels.derivative!(model::RandomFriction, D::AbstractMatrix, R::AbstractMatrix)
    NonadiabaticModels.derivative!(model.model, D, R)
end

function friction!(::RandomFriction, F::AbstractMatrix, ::AbstractMatrix)
    randn!(F)
    F .= F'F
    F .= (F + F')/2
end

NonadiabaticModels.ndofs(model::RandomFriction) = NonadiabaticModels.ndofs(model.model)
