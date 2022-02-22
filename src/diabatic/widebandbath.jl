using LinearAlgebra: diagind

struct WideBandBath{M<:DiabaticModel,V<:AbstractVector,T} <: DiabaticFrictionModel
    model::M
    bathstates::V
    fermilevel::T
    function WideBandBath(model, bathstates, fermilevel)
        bathstates = austrip.(bathstates)
        fermilevel = austrip(fermilevel)
        new{typeof(model),typeof(bathstates),typeof(fermilevel)}(model, bathstates, fermilevel)
    end
end

function WideBandBath(model::DiabaticModel; step, bandmin, bandmax, fermilevel=0.0)
    WideBandBath(model, range(bandmin, bandmax; step=step), fermilevel)
end

NQCModels.nstates(model::WideBandBath) = NQCModels.nstates(model.model) + length(model.bathstates) - 1
NQCModels.ndofs(model::WideBandBath) = NQCModels.ndofs(model.model)
NQCModels.fermilevel(model::WideBandBath) = model.fermilevel

function NQCModels.potential!(model::WideBandBath, V::Hermitian, r::Real)

    Vsystem = NQCModels.potential(model.model, r)
    n = NQCModels.nstates(model.model)
    ϵ0 = Vsystem[1,1]

    # System states
    @views V[diagind(V)[begin:n-1]] .= Vsystem[diagind(Vsystem)[begin+1:end]] .- ϵ0

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
    @views D[diagind(D)[begin:n-1]] .= Dsystem[diagind(Dsystem)[begin+1:end]] .- ∂ϵ0

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
