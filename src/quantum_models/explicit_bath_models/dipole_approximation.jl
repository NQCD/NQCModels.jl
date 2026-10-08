
"""
    DipoleApproximation{M<:QuantumModel,T,E} <: QuantumModel

Wraps any `QuantumModel` (e.g. [`AndersonHolstein`](@ref)) and adds the dipole-approximation
interaction `-μ⋅E(t)` to its potential, where `μ` is a fixed dipole matrix and `E` is a
(possibly time-dependent) external field.

Everything other than the potential (number of states, degrees of freedom, derivatives,
state-independent terms, etc.) is inherited unchanged from the wrapped `system` model, since
`μ` and `E(t)` do not depend on the nuclear positions `r`.

`field` may either be a constant `Number` or a callable `field(t)` returning the field
strength at time `t`. The model keeps its own internal clock, updated with [`NQCModels.set_time!`](@ref)
and read with [`NQCModels.get_time`](@ref).
"""
struct DipoleApproximation{M<:QuantumModel,T,E} <: QuantumModel
    system::M
    μ::Matrix{T}
    field::E
    t::Base.RefValue{Float64}
end

function DipoleApproximation(system::QuantumModel, μ::Matrix, field; t0=0.0)
    n = NQCModels.nstates(system)
    size(μ) == (n, n) || throw(DimensionMismatch(
        "μ must be an $n×$n matrix matching nstates(system), got size $(size(μ))"
    ))
    return DipoleApproximation(system, μ, field, Ref(Float64(t0)))
end

evaluate_field(field, t) = field(t)
evaluate_field(field::Number, t) = field

NQCModels.nstates(model::DipoleApproximation) = NQCModels.nstates(model.system)
NQCModels.ndofs(model::DipoleApproximation) = NQCModels.ndofs(model.system)
NQCModels.nelectrons(model::DipoleApproximation) = NQCModels.nelectrons(model.system)
NQCModels.fermilevel(model::DipoleApproximation) = NQCModels.fermilevel(model.system)

NQCModels.set_time!(model::DipoleApproximation, t::Real) = (model.t[] = t; nothing)
NQCModels.get_time(model::DipoleApproximation) = model.t[]

function NQCModels.potential!(model::DipoleApproximation, V::Hermitian, r::AbstractMatrix)
    NQCModels.potential!(model.system, V, r)
    V.data .-= model.μ .* evaluate_field(model.field, model.t[])
    return nothing
end

function NQCModels.derivative!(model::DipoleApproximation, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)
    NQCModels.derivative!(model.system, D, r)
end

function NQCModels.derivative!(model::DipoleApproximation, D::Hermitian, r::AbstractMatrix)
    NQCModels.derivative!(model.system, D, r)
end

NQCModels.state_independent_potential(model::DipoleApproximation, r::AbstractMatrix) =
    NQCModels.state_independent_potential(model.system, r)

function NQCModels.state_independent_potential!(model::DipoleApproximation, Vsystem::AbstractMatrix, r::AbstractMatrix)
    NQCModels.state_independent_potential!(model.system, Vsystem, r)
end

NQCModels.state_independent_derivative(model::DipoleApproximation, r::AbstractMatrix) =
    NQCModels.state_independent_derivative(model.system, r)

function NQCModels.state_independent_derivative!(model::DipoleApproximation, ∂V::AbstractMatrix, r::AbstractMatrix)
    NQCModels.state_independent_derivative!(model.system, ∂V, r)
end

NQCModels.get_subsystem_derivative(model::DipoleApproximation, r::AbstractMatrix) =
    NQCModels.get_subsystem_derivative(model.system, r)


"""
    Compute the dipole matrix for a given Hamiltonian in a simple way.
    The dipole matrix is computed using the model parameters A, Γ, and W.
    The Hamiltonian H is expected to be a Hermitian matrix.
    Γ is the broadening parameter that accounts for the finite lifetime of the states. This is estimated
    from  max(ħ/τ, ħv_f/L) where τ is the lifetime of the states, v_f is the Fermi velocity, and L is the size of the system.
    A is the dipole strenth given by (ħ^2/ml) where m is mass and l is the length scale of the system.
    l is the min(L, v_Fτ, surface-decay)
    Finally, W is the half-bandwidth of the electronic states. This is used to ensure that the dipole 
    matrix elements go to zero at the band edges.
    If type is set to :NAH, the dipole matrix has the first row and column set to zero, which is appropriate for the Newns-Anderson-Holstein model.
"""
function model_dipole_matrix(H; A, Γ, W=Inf, type=:NAH)
    E = diag(H)
    s  = @. (1 - (E/W)^2)^(1/4)      # band-edge envelope (requires |E| < W)
    ΔE = E .- E'
    μ = @. im * A * s * s' * ΔE / (ΔE^2 + Γ^2)
    if type == :NAH
        μ[1, :] .= 0.0
        μ[:, 1] .= 0.0
    end
    return μ
end