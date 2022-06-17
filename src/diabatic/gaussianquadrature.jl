using LinearAlgebra: diagind
using FastGaussQuadrature: gausslegendre

struct GaussQuadrature{M<:DiabaticModel,V<:AbstractVector,T} <: DiabaticFrictionModel
    model::M
    bathstates::V
    ρ::T
    x_gauss::Vector{Float64}
    w_gauss::Vector{Float64}
    function GaussQuadrature(model, bathstates)
        bathstates = austrip.(bathstates)
        nelectrons = floor(Int, length(bathstates)/2)
        ρ = length(bathstates) / (bathstates[end] - bathstates[begin])
        x_gauss, w_gauss = gausslegendre(nelectrons)
        new{typeof(model),typeof(bathstates),typeof(ρ)}(model, bathstates, ρ, x_gauss, w_gauss)
    end
end

function GaussQuadrature(model::DiabaticModel; nstates, bandmin, bandmax)
    #GaussQuadrature(model, range(bandmin, bandmax; step=step))
    GaussQuadrature(model, range(bandmin, bandmax; length=nstates))
end

NQCModels.nstates(model::GaussQuadrature) = NQCModels.nstates(model.model) + length(model.bathstates) - 1
NQCModels.ndofs(model::GaussQuadrature) = NQCModels.ndofs(model.model)
NQCModels.nelectrons(model::GaussQuadrature) = fld(NQCModels.nstates(model), 2)
NQCModels.fermilevel(::GaussQuadrature) = 0.0

function NQCModels.potential!(model::GaussQuadrature, V::Hermitian, r::AbstractMatrix)

    Vsystem = NQCModels.potential(model.model, r)
    n = NQCModels.nstates(model.model)
    ϵ0 = Vsystem[1,1]

    # System states
    # Sets the first state of the Hamiltonian V[1,1]
    @views V[diagind(V)[begin:n-1]] .= Vsystem[diagind(Vsystem)[begin+1:end]] .- ϵ0

    # adds Bath states to the diagonal
    #V[diagind(V)[n:end]] .= model.bathstates
    DeltaE = model.bathstates[end] - model.bathstates[begin]
    bath = @view V[diagind(V)[2:end]]
    set_bath_energies!(bath, model.x_gauss, DeltaE)

    # Set coupling values
    #for i=1:n-1
    #    V.data[i,n:end] .= Vsystem[i+1,1] / sqrt(model.ρ)
    #end
    couplings = @view V.data[2:end,1]
    E_coup = Vsystem[2,1]
    set_coupling_elements!(couplings, model.w_gauss, DeltaE, E_coup)
    couplings = @view V.data[1, 2:end]
    set_coupling_elements!(couplings, model.w_gauss, DeltaE, E_coup)

    return V
end

function NQCModels.derivative!(model::GaussQuadrature, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)

    Dsystem = NQCModels.derivative(model.model, r)
    n = NQCModels.nstates(model.model)
    DeltaE = model.bathstates[end] - model.bathstates[begin]
    
    for I in eachindex(Dsystem, D)
        ∂ϵ0 = Dsystem[I][1,1]

        # System states
        D_system_output = @view D[I][diagind(D[I])[begin:n-1]]
        D_system_input = @view Dsystem[I][diagind(Dsystem[I])[begin+1:end]]
        D_system_output .= D_system_input .- ∂ϵ0

        # Coupling
        for i=1:n-1
            #D[I].data[i,n:end] .= Dsystem[I][i+1,1] / sqrt(model.ρ)
            coupling = @view D[I].data[i,n:end]
            set_coupling_elements!(coupling, model.w_gauss, DeltaE, Dsystem[I][i+1,1])
            coupling = @view D[I].data[n:end,i]
            set_coupling_elements!(coupling, model.w_gauss, DeltaE, Dsystem[I][i+1,1])
        end
    end

    return D
end

function NQCModels.state_independent_potential(model::GaussQuadrature, r)
    Vsystem = NQCModels.potential(model.model, r)
    return Vsystem[1,1]
end

function NQCModels.state_independent_derivative!(model::GaussQuadrature, ∂V::AbstractMatrix, r)
    Dsystem = NQCModels.derivative(model.model, r)
    for I in eachindex(∂V, Dsystem)
        ∂V[I] = Dsystem[I][1,1]
    end

    return ∂V
end

function set_bath_energies!(bath, x_gauss, DeltaE)
    @inbounds for i in eachindex(x_gauss)
        bath[i] = DeltaE * (-1 + x_gauss[i]) / 4
    end
    n = length(x_gauss)
    @inbounds for i in eachindex(x_gauss)
        bath[i+n] = DeltaE * (1 + x_gauss[i]) / 4
    end
end

function set_coupling_elements!(coupling, w_gauss, DeltaE, E_coup)
    nelectrons = length(w_gauss)
    @inbounds for i in eachindex(w_gauss)
        coupling[i] = sqrt(DeltaE * w_gauss[i]) / 2 * E_coup / sqrt(DeltaE)
    end
    @inbounds for i in eachindex(w_gauss)
        coupling[i+nelectrons] = coupling[i]
    end
end