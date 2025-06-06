
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

    x = R[1]
    z = R[2]

    V11 = d*(1-exp(-α*x))^2 + exp(-λ₁*(x+d))
    V22 = d*(1-exp(-α*z))^2 + exp(-λ₂*(x+x₀))
    V12 = c * exp(-γ*(z-z12))

    return Hermitian([V11 V12; V12 V22])
end

function NQCModels.potential!(model::GatesHollowayElbow, V::Hermitian, R::AbstractMatrix)
    Parameters.@unpack λ₁, λ₂, z₀, x₀, α, d, z12, c, γ = model

    x = R[1]
    z = R[2]

    V11 = d*(1-exp(-α*x))^2 + exp(-λ₁*(x+d))
    V22 = d*(1-exp(-α*z))^2 + exp(-λ₂*(x+x₀))
    V12 = c * exp(-γ*(z-z12))

    V.data .= Hermitian([V11 V12; V12 V22])
end

function NQCModels.derivative(model::GatesHollowayElbow, R::AbstractMatrix)
    D = [Hermitian(zero(matrix_template(model, eltype(R)))) for _=1:size(R, 1), _=1:size(R, 2)]
    NQCModels.derivative!(model, D, R)
    return D
end

function NQCModels.derivative!(model::GatesHollowayElbow, D::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    Parameters.@unpack λ₁, λ₂, z₀, x₀, α, d, z12, c, γ = model

    drepel(x, λ, d) = -λ*exp(-λ*(x+d))
    dmorse(x, d, α) = 2α*d*(1-exp(-α*x))*exp(-α*x)

    x = R[1]
    z = R[2]

    # x derivative
    D11 = 2α*d*(1-exp(-α*x))*exp(-α*x)
    D22 = -λ₂*exp(-λ₂*(x+x₀))
    D[1].data .= Hermitian([D11 0; 0 D22])

    # z derivative
    D11 = -λ₁*exp(-λ₁*(z+z₀))
    D22 = 2α*d*(1-exp(-α*z))*exp(-α*z)
    D12 = c * -γ*exp(-γ*(x+d))
    D[2].data .= Hermitian([D11 D12; D12 D22])

    return D
end
