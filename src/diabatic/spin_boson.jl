export SpinBoson
export BosonBath
export DebyeSpectralDensity
export OhmicSpectralDensity

"""
    SpinBoson(density::SpectralDensity, N::Integer, ϵ, Δ)

Spin boson model with `N` bosons with given spectral density.

# References
[Xin He, Jian Liu, J. Chem. Phys. 151, 024105 (2019)](http://aip.scitation.org/doi/10.1063/1.5108736)
"""
struct SpinBoson{T} <: DiabaticModel
    n_states::Int
    ϵ::T
    Δ::T
    ωⱼ::Vector{T}
    cⱼ::Vector{T}
end

nstates(::SpinBoson) = 2

function SpinBoson(density::SpectralDensity, N::Integer, ϵ, Δ)
    ωⱼ, cⱼ = discretize(density, N)
    SpinBoson(2, ϵ, Δ, ωⱼ, cⱼ)
end

function potential(model::SpinBoson, R::AbstractMatrix)
    r = @view R[1,:]

    @unpack ωⱼ, cⱼ, ϵ, Δ = model

    v0 = 0.0
    for i in eachindex(ωⱼ)
        v0 += ωⱼ[i]^2 * r[i]^2 / 2
    end
    V11 = v0 + ϵ
    V22 = v0 - ϵ

    for i in eachindex(model.cⱼ)
        V11 += cⱼ[i] * r[i]
        V22 -= cⱼ[i] * r[i]
    end

    V12 = Δ

    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function derivative!(model::SpinBoson, D::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    r = @view R[1,:]

    @unpack ωⱼ, cⱼ = model

    for i in eachindex(r)
        d0 = ωⱼ[i]^2 * r[i]
        D[1,i] = Hermitian(SMatrix{2,2}(d0 + cⱼ[i], 0, 0, d0 - cⱼ[i]))
    end

    return D
end

"""
    BosonBath(density::SpectralDensity, N::Integer)

Bosonic bath with given spectral density.

Useful for sampling the bath uncoupled from the spin for spin-boson dynamics.
"""
struct BosonBath{T} <: AdiabaticModel
    ωⱼ::Vector{T}
end

function BosonBath(density::SpectralDensity, N::Integer)
    ωⱼ, _ = discretize(density, N)
    BosonBath(ωⱼ)
end

function potential(model::BosonBath, R::AbstractMatrix)
    r = @view R[1,:]
    return sum(model.ωⱼ .^2 .* r .^2 ./ 2)
end

function derivative!(model::BosonBath, D::AbstractMatrix, R::AbstractMatrix)
    r = @view R[1,:]

    for i in eachindex(r)
        D[1,i] = model.ωⱼ[i]^2 * r[i]
    end
end

abstract type SpectralDensity end

function discretize(J::SpectralDensity, N::Integer)
    ωⱼ = ω.(Ref(J), N, 1:N)
    cⱼ = c.(Ref(J), N, ωⱼ)
    return ωⱼ, cⱼ
end

struct OhmicSpectralDensity{T} <: SpectralDensity
    ωᶜ::T
    α::T
end

(J::OhmicSpectralDensity)(ω) = π/2 * J.α * ω * exp(-ω / J.ωᶜ)
ω(J::OhmicSpectralDensity, N, j) = -J.ωᶜ * log(1 - j/(N+1))
c(J::OhmicSpectralDensity, N, ωⱼ) = sqrt(J.α * J.ωᶜ/(N+1)) * ωⱼ

struct DebyeSpectralDensity{T} <: SpectralDensity
    ωᶜ::T
    λ::T
end

(J::DebyeSpectralDensity)(ω) = 2λ * ωᶜ * ω / (ωᶜ^2 + ω^2)
ω(J::DebyeSpectralDensity, N, j) = J.ωᶜ * tan(π/2 * (1 - j/(N+1)))
c(J::DebyeSpectralDensity, N, ωⱼ) = sqrt(2J.λ/(N+1)) * ωⱼ

struct AltDebyeSpectralDensity{T} <: SpectralDensity
    ωᶜ::T
    λ::T
    ωᵐ::T
end

(J::AltDebyeSpectralDensity)(ω) = 2λ * ωᶜ * ω / (ωᶜ^2 + ω^2)
ω(J::AltDebyeSpectralDensity, N, j) = tan(j * atan(J.ωᵐ / J.ωᶜ) / N) * J.ωᶜ
c(J::AltDebyeSpectralDensity, N, ωⱼ) = sqrt(4J.λ * atan(J.ωᵐ / J.ωᶜ) / (π * N)) * ωⱼ
