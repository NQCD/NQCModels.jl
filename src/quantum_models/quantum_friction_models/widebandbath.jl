using LinearAlgebra: diagind

struct WideBandBath{M<:QuantumModel,V<:AbstractVector,T,D} <: QuantumFrictionModel
    model::M
    bathstates::V
    ρ::T
    tmp_derivative::Base.RefValue{D}
#=     function WideBandBath(model, bathstates, tmp_derivative)
        bathstates = austrip.(bathstates)
        ρ = length(bathstates) / (bathstates[end] - bathstates[begin])
        new{typeof(model), typeof(bathstates), typeof(ρ)}(model, bathstates, ρ, tmp_derivative)
    end =#
end

function WideBandBath(model::QuantumModel; step, bandmin, bandmax)
    bathstates = austrip.(collect(range(bandmin, bandmax; step=step)))
    ρ = length(bathstates) / (bathstates[end] - bathstates[begin])

    tmp_derivative = Ref(NQCModels.zero_derivative(model, zeros(1,1)))

    return WideBandBath(model, bathstates, ρ, tmp_derivative)
end

NQCModels.nstates(model::WideBandBath) = NQCModels.nstates(model.model) + length(model.bathstates) - 1
NQCModels.ndofs(model::WideBandBath) = NQCModels.ndofs(model.model)
NQCModels.nelectrons(model::WideBandBath) = fld(NQCModels.nstates(model), 2)
NQCModels.fermilevel(::WideBandBath) = 0.0

function NQCModels.potential(model::WideBandBath, r::AbstractMatrix)

    Vsystem = NQCModels.potential(model.model, r)
    n = NQCModels.nstates(model.model)
    ϵ0 = Vsystem[1,1]
    V = zeros((NQCModels.nstates(model), NQCModels.nstates(model)))

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
end

function NQCModels.derivative(model::WideBandBath, r::AbstractMatrix)

    Dsystem = NQCModels.get_subsystem_derivative(model.model, r)
    n = NQCModels.nstates(model.model)
    
    D = zeros((NQCModels.nstates(model), NQCModels.nstates(model)))

    for I in eachindex(Dsystem, D)
        # Coupling
        for i=1:n-1
            D[I][i,n:end] .= Dsystem[I][i+1,1] / sqrt(model.ρ)
        end
    end

    return D
end

function NQCModels.derivative!(model::WideBandBath, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)

    Dsystem = NQCModels.get_subsystem_derivative(model.model, r)
    n = NQCModels.nstates(model.model)
    
    for I in eachindex(Dsystem, D)
        ∂ϵ0 = Dsystem[I][1,1]

        # System states
        D_system_output = @view D[I][diagind(D[I])[begin:n-1]]
        D_system_input = @view Dsystem[I][diagind(Dsystem[I])[begin+1:end]]
        @. D_system_output = D_system_input - ∂ϵ0

        # Coupling
        for i=1:n-1
            D[I].data[i,n:end] .= Dsystem[I][i+1,1] / sqrt(model.ρ)
        end
    end
end

function NQCModels.state_independent_potential(model::WideBandBath, r::AbstractMatrix)
    Vsystem = NQCModels.potential(model.model, r)
    return Vsystem[1,1]
end

function NQCModels.state_independent_potential!(model::WideBandBath, Vsystem::AbstractMatrix, r::AbstractMatrix)
    Vsystem .= NQCModels.potential(model.model, r)
end

function NQCModels.state_independent_derivative(model::WideBandBath, r::AbstractMatrix)
    Dsystem = NQCModels.get_subsystem_derivative(model.model, r)
    ∂V = zeros(size(r))
    for I in eachindex(∂V, Dsystem)
        ∂V[I].data .= Dsystem[I][1,1]
    end

    return ∂V
end

function NQCModels.state_independent_derivative!(model::WideBandBath, ∂V::AbstractMatrix, r::AbstractMatrix)
    Dsystem = NQCModels.get_subsystem_derivative(model.model, r)
    for I in eachindex(∂V, Dsystem)
        ∂V[I].data .= Dsystem[I][1,1]
    end
end

function get_subsystem_derivative(model::WideBandBath, r::AbstractMatrix)
    if size(r) != size(model.tmp_derivative[])
        model.tmp_derivative[] = NQCModels.zero_derivative(model.model, r)
    end
    NQCModels.derivative!(model.model, model.tmp_derivative[], r)
    return model.tmp_derivative[]
end
