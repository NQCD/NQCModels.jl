using LinearAlgebra: diagind

struct WideBandBath{M<:DiabaticModel,V<:AbstractVector,T} <: DiabaticFrictionModel
    model::M
    bathstates::V
    ρ::T
    function WideBandBath(model, bathstates)
        bathstates = austrip.(bathstates)
        ρ = length(bathstates) / (bathstates[end] - bathstates[begin])
        new{typeof(model),typeof(bathstates),typeof(ρ)}(model, bathstates, ρ)
    end
end

function WideBandBath(model::DiabaticModel; step, bandmin, bandmax)
    WideBandBath(model, range(bandmin, bandmax; step=step))
end

NQCModels.nstates(model::WideBandBath) = NQCModels.nstates(model.model) + length(model.bathstates) - 1
NQCModels.ndofs(model::WideBandBath) = NQCModels.ndofs(model.model)
NQCModels.nelectrons(model::WideBandBath) = fld(NQCModels.nstates(model), 2)
NQCModels.fermilevel(::WideBandBath) = 0.0

function NQCModels.potential!(model::WideBandBath, V::Hermitian, r::AbstractMatrix)

    Vsystem = NQCModels.potential(model.model, r)
    n = NQCModels.nstates(model.model)
    ϵ0 = Vsystem[1,1]

    # System states
    @views V[diagind(V)[begin:n-1]] .= Vsystem[diagind(Vsystem)[begin+1:end]] .- ϵ0

    # Bath states
    V[diagind(V)[n:end]] .= model.bathstates

    # Coupling
    for i=1:n-1
        V.data[i,n:end] .= Vsystem[i+1,1] / sqrt(model.ρ)
    end

    return V
end

function NQCModels.derivative!(model::WideBandBath, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)

    Dsystem = NQCModels.derivative(model.model, r)
    n = NQCModels.nstates(model.model)
    
    for I in eachindex(Dsystem, D)
        ∂ϵ0 = Dsystem[I][1,1]

        # System states
        D_system_output = @view D[I][diagind(D[I])[begin:n-1]]
        D_system_input = @view Dsystem[I][diagind(Dsystem[I])[begin+1:end]]
        D_system_output .= D_system_input .- ∂ϵ0

        # Coupling
        for i=1:n-1
            D[I].data[i,n:end] .= Dsystem[I][i+1,1] / sqrt(model.ρ)
        end
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
