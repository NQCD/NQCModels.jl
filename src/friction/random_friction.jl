using Random: randn!

struct RandomFriction <: ElectronicFrictionProvider
    ndofs::Int
end

function friction!(::RandomFriction, F::AbstractMatrix, ::AbstractMatrix)
    randn!(F)
    F .= F'F
    F .= (F + F')/2
end

NonadiabaticModels.ndofs(model::RandomFriction) = model.ndofs
