
struct AdiabaticStateSelector{M} <: NQCModels.AdiabaticModels.AdiabaticModel
    model::M
    state::Int
    function AdiabaticStateSelector(model, state)
        state < 1 && throw(DomainError(state, "selected state must be greater than 0"))
        state > NQCModels.nstates(model) && throw(DomainError(state, "selected state must be less than the total number of states of the diabatic model"))
        return new{typeof(model)}(model, state)
    end
end

NQCModels.ndofs(model::AdiabaticStateSelector) = NQCModels.ndofs(model.model)

function NQCModels.potential(model::AdiabaticStateSelector, r::AbstractMatrix)
    V = NQCModels.potential(model.model, r)
    eigenvalues = LinearAlgebra.eigvals(V)
    return eigenvalues[model.state]
end

function NQCModels.derivative!(model::AdiabaticStateSelector, output::AbstractMatrix, r::AbstractMatrix)
    V = NQCModels.potential(model.model, r)
    U = LinearAlgebra.eigvecs(V)
    D = NQCModels.derivative(model.model, r)
    for I in eachindex(output, D)
        output[I] = (U' * D[I] * U)[model.state, model.state]
    end
    return output
end
