
"""
    GatesHollowayElbow()

Simple two state elbow potential from Gates and Holloway:
[Journal of Electron Spectroscopy and Related Phenomena, 64/65 (1993) 633-639](https://doi.org/10.1016/0368-2048(93)80131-5)

Has two diabatic states each comprised of the sum of a Morse and a repulsive potential.
The coupling between them is an exponential function of `z` (distance from the surface).
"""
Parameters.@with_kw struct GatesHollowayElbow <: QuantumModel
    λ₁::Float64 = 3.5
    λ₂::Float64 = 3.5
    z₀::Float64 = 1.4
    x₀::Float64 = 0.6
    α::Float64 = 1.028
    d::Float64 = austrip(4.67u"eV")
    z12::Float64 = 0.5
    c::Float64 = austrip(0.5u"eV")
    γ::Float64 = 1.0
end

NQCModels.nstates(::GatesHollowayElbow) = 2
NQCModels.ndofs(::GatesHollowayElbow) = 1

function NQCModels.potential(model::GatesHollowayElbow, R::AbstractMatrix)
    Parameters.@unpack λ₁, λ₂, z₀, x₀, α, d, z12, c, γ = model

    repel(x, λ, d) = exp(-λ*(x+d))
    morse(x, d, α) = d*(1-exp(-α*x))^2

    x = R[1]
    z = R[2]

    V11 = morse(x, d, α) + repel(z, λ₁, z₀)
    V22 = morse(z, d, α) + repel(x, λ₂, x₀)
    V12 = c * repel(z, γ,-z12)

    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NQCModels.potential!(model::GatesHollowayElbow, V::Hermitian, R::AbstractMatrix)
    Parameters.@unpack λ₁, λ₂, z₀, x₀, α, d, z12, c, γ = model

    repel(x, λ, d) = exp(-λ*(x+d))
    morse(x, d, α) = d*(1-exp(-α*x))^2

    x = R[1]
    z = R[2]

    V11 = morse(x, d, α) + repel(z, λ₁, z₀)
    V22 = morse(z, d, α) + repel(x, λ₂, x₀)
    V12 = c * repel(z, γ,-z12)

    V = Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NQCModels.derivative!(model::GatesHollowayElbow, D::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    Parameters.@unpack λ₁, λ₂, z₀, x₀, α, d, z12, c, γ = model

    drepel(x, λ, d) = -λ*exp(-λ*(x+d))
    dmorse(x, d, α) = 2α*d*(1-exp(-α*x))*exp(-α*x)

    x = R[1]
    z = R[2]

    # x derivative
    D11 = dmorse(x, d, α)
    D22 = drepel(x, λ₂, x₀)
    D[1] = Hermitian(SMatrix{2,2}(D11, 0, 0, D22))

    # z derivative
    D11 = drepel(z, λ₁, z₀)
    D22 = dmorse(z, d, α)
    D12 = c * drepel(z, γ, -z12)
    D[2] = Hermitian(SMatrix{2,2}(D11, D12, D12, D22))

    return D
end
