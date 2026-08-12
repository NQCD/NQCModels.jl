struct StateSelector{M<:QuantumModels.QuantumModel,B<:State} <: NQCModels.ClassicalModels.ClassicalModel
    quantum_model::M
    state::Int

    function StateSelector(quantum_model::QuantumModels.QuantumModel, state::Int, ::Type{B}) where {B<:State}
        state < 1 && throw(DomainError(state, "selected state must be greater than 0"))
        state > NQCModels.nstates(quantum_model) && throw(
            DomainError(state, "selected state must be less than or equal to the total number of states of the model"),
        )
        return new{typeof(quantum_model),B}(quantum_model, state)
    end
end

NQCModels.ndofs(model::StateSelector) = NQCModels.ndofs(model.quantum_model)

# --- Diabatic model, want the adiabatic-basis energy of `state` ---
function NQCModels.potential(model::StateSelector{M,Adiabatic}, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    return eigen(V).values[model.state]
end

function NQCModels.derivative!(model::StateSelector{M,Adiabatic}, output::AbstractMatrix, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    U = eigen(V).vectors
    D = NQCModels.derivative(model.quantum_model, r)
    for I in eachindex(output, D)
        output[I] = (U' * D[I] * U)[model.state, model.state]
    end
    return output
end

# --- Diabatic model, want the raw diagonal diabatic element ---
function NQCModels.potential(model::StateSelector{M,Diabatic}, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    return V[model.state, model.state]
end

function NQCModels.derivative!(model::StateSelector{M,Diabatic}, output::AbstractMatrix, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    D = NQCModels.derivative(model.quantum_model, r)
    for I in eachindex(output, D)
        output[I] = D[I][model.state, model.state]
    end
    return output
end

# --- Adiabatic model, want the (already adiabatic) energy of `state` ---
function NQCModels.potential(model::StateSelector{M,Adiabatic}, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Adiabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    return V[model.state]
end

function NQCModels.derivative!(model::StateSelector{M,Adiabatic}, output::AbstractMatrix, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Adiabatic}}
    D = NQCModels.derivative(model.quantum_model, r)
    for I in eachindex(output, D)
        output[I] = D[I][model.state]
    end
    return output
end

struct ReducedQuantumModel{M<:QuantumModels.QuantumModel,B<:State} <: QuantumModels.QuantumModel{B}
    quantum_model::M
    states::Vector{Int}

    function ReducedQuantumModel(quantum_model::QuantumModels.QuantumModel, states::AbstractVector{Int}, ::Type{B}) where {B<:State}
        n = NQCModels.nstates(quantum_model)
        isempty(states) && throw(ArgumentError("`states` must not be empty"))
        any(s -> s < 1, states) && throw(DomainError(states, "selected states must be greater than 0"))
        any(s -> s > n, states) && throw(DomainError(states, "selected states must not exceed the total number of states ($n) of the underlying model"))
        allunique(states) || throw(ArgumentError("`states` must not contain duplicates"))
        return new{typeof(quantum_model),B}(quantum_model, collect(states))
    end
end

NQCModels.ndofs(model::ReducedQuantumModel) = NQCModels.ndofs(model.quantum_model)
NQCModels.nstates(model::ReducedQuantumModel) = length(model.states)

# --- Diabatic underlying model, stay in diabatic representation ---
# Just take the submatrix — no diagonalization needed.
function NQCModels.potential(model::ReducedQuantumModel{M,Diabatic}, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    return V[model.states, model.states]
end

function NQCModels.derivative!(model::ReducedQuantumModel{M,Diabatic}, output::AbstractMatrix, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    D = NQCModels.derivative(model.quantum_model, r)
    for I in eachindex(output, D)
        output[I] = D[I][model.states, model.states]
    end
    return output
end

# --- Diabatic underlying model, truncate to a subset of adiabatic eigenstates ---
# Diagonalize the FULL Hamiltonian (truncating states before diagonalizing would
# corrupt the eigenbasis), then slice the requested rows/cols out of the result.
function NQCModels.potential(model::ReducedQuantumModel{M,Adiabatic}, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    eigenvalues = eigen(V).values
    return Diagonal(eigenvalues[model.states])
end

function NQCModels.derivative!(model::ReducedQuantumModel{M,Adiabatic}, output::AbstractMatrix, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Diabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    U = eigen(V).vectors
    D = NQCModels.derivative(model.quantum_model, r)
    for I in eachindex(output, D)
        output[I] = (U' * D[I] * U)[model.states, model.states]
    end
    return output
end

# --- Adiabatic underlying model, stay in adiabatic representation ---
# potential returns a vector of eigenvalues; derivative returns full matrices
# (diagonal Hellmann-Feynman terms + any off-diagonal derivative-coupling terms
# the underlying model already encodes) — both get sliced to `states`.
function NQCModels.potential(model::ReducedQuantumModel{M,Adiabatic}, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Adiabatic}}
    V = NQCModels.potential(model.quantum_model, r)
    return V[model.states]
end

function NQCModels.derivative!(model::ReducedQuantumModel{M,Adiabatic}, output::AbstractMatrix, r::AbstractMatrix) where {M<:QuantumModels.QuantumModel{Adiabatic}}
    D = NQCModels.derivative(model.quantum_model, r)
    for I in eachindex(output, D)
        output[I] = D[I][model.states, model.states]
    end
    return output
end