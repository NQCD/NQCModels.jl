using LinearAlgebra: diagind

struct WideBandBath{V<:AbstractVector,M<:DiabaticModel} <: DiabaticFrictionModel
    model::M
    bathstates::V
end

WideBandBath(model; nbathstates, bandmin, bandmax) = WideBandBath(model, range(bandmin, bandmax; length=nbathstates))
WideBandBath(model::DiabaticModel; bathstates) = WideBandBath(model, austrip.(bathstates))

NQCModels.nstates(model::WideBandBath) = NQCModels.nstates(model.model) + length(model.bathstates) - 1
NQCModels.ndofs(model::WideBandBath) = NQCModels.ndofs(model.model)

function NQCModels.potential!(model::WideBandBath, V::Hermitian, r::Real)

    Vsystem = NQCModels.potential(model.model, r)
    n = NQCModels.nstates(model.model)
    ϵ0 = Vsystem[1,1]

    # System states
    V[diagind(V)[begin:n-1]] .= Vsystem[diagind(Vsystem)[begin+1:end]] .- ϵ0

    # Bath states
    V[diagind(V)[n:end]] .= model.bathstates

    # Coupling
    for i=1:n-1
        V.data[i,n:end] .= Vsystem[i+1,1]
    end

    return V
end

function NQCModels.derivative!(model::WideBandBath, D::Hermitian, r::Real)

    Dsystem = NQCModels.derivative(model.model, r)
    n = NQCModels.nstates(model.model)
    ∂ϵ0 = Dsystem[1,1]

    # System states
    D[diagind(D)[begin:n-1]] .= Dsystem[diagind(Dsystem)[begin+1:end]] .- ∂ϵ0

    # Coupling
    for i=1:n-1
        D.data[i,n:end] .= Dsystem[i+1,1]
    end

    return D
end

function NQCModels.state_independent_potential(model::WideBandBath, r)
    Vsystem = NQCModels.potential(model.model, r)
    return Vsystem[1,1]
end

function NQCModels.state_independent_derivative!(model::WideBandBath, ∂V::AbstractMatrix, r)
    Dsystem = NQCModels.derivative(model.model, r)
    for I in eachindex(∂V, Dsystem)
        ∂V[I] = Dsystem[I][1,1]
    end

    return ∂V
end
