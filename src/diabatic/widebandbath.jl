using LinearAlgebra: diagind

struct WideBandBath{T<:AbstractFloat,M<:DiabaticModel} <: DiabaticFrictionModel
    model::M
    nbathstates::Int
    bandmin::T
    bandmax::T
end

WideBandBath(model; nbathstates, bandmin, bandmax) = WideBandBath(model, nbathstates, bandmin, bandmax)

NQCModels.nstates(model::WideBandBath) = NQCModels.nstates(model.model) + model.nbathstates - 1
NQCModels.ndofs(model::WideBandBath) = NQCModels.ndofs(model.model)

function NQCModels.potential!(model::WideBandBath, V::Hermitian, r::Real)

    (;bandmin, bandmax, nbathstates) = model

    Vsystem = NQCModels.potential(model.model, r)
    n = NQCModels.nstates(model.model)
    ϵ0 = Vsystem[1,1]

    # System states
    V[diagind(V)[begin:n-1]] .= Vsystem[diagind(Vsystem)[begin+1:end]] .- ϵ0

    # Bath states
    ϵ = range(bandmin, bandmax, length=nbathstates)
    V[diagind(V)[n:end]] .= ϵ

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
    ∂V[1,1] = Dsystem[1,1]
    return ∂V
end
