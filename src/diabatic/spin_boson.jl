export DebyeSpinBoson
export DebyeBosonBath

"""
    DebyeSpinBoson(Nᵇ; ϵ=0, Δ=1, η=0.09, ωᶜ=2.5)

Spin boson model with `Nᵇ` bosons with Debye spectral density:
`J(ω) = 2λ * ω * ωᶜ / (ω^2 + ωᶜ^2)`

Default parameters are chosen to match model 13 from the reference.

# References
[Xin He, Jian Liu, J. Chem. Phys. 151, 024105 (2019)](http://aip.scitation.org/doi/10.1063/1.5108736)
"""
struct DebyeSpinBoson{E,D,N,W} <: DiabaticModel
    n_states::Int
    ϵ::E
    Δ::D
    λ::N
    ωᶜ::W
    ωⱼ::Vector{W}
    cⱼ::Vector{W}
end

function DebyeSpinBoson(Nᵇ; ϵ=0, Δ=1, λ=0.25, ωᶜ=0.25)

    ω(j) = ωᶜ * tan(π/2 * (1 - j/(Nᵇ+1)))
    c(ωⱼ) = sqrt(2λ/(Nᵇ+1)) * ωⱼ

    ωⱼ = ω.(1:Nᵇ)
    cⱼ = c.(ωⱼ)

    DebyeSpinBoson(2, ϵ, Δ, λ, ωᶜ, ωⱼ, cⱼ)
end

function potential(model::DebyeSpinBoson, R::AbstractMatrix)
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

function derivative!(model::DebyeSpinBoson, D::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    r = @view R[1,:]

    @unpack ωⱼ, cⱼ = model

    for i in eachindex(r)
        d0 = ωⱼ[i]^2 * r[i]
        D[1,i] = Hermitian(SMatrix{2,2}(d0 + cⱼ[i], 0, 0, d0 - cⱼ[i]))
    end

    return D
end

"""
    DebyeBosonBath(Nᵇ; ωᶜ=2.5)

Bosonic bath with Debye spectral density.
Useful for sampling the bath uncoupled from the spin for spin-boson dynamics.
"""
struct DebyeBosonBath{WC,WJ} <: AdiabaticModel
    n_states::Int
    ωᶜ::WC
    ωⱼ::Vector{WJ}
end

function DebyeBosonBath(Nᵇ; ωᶜ=0.25)

    ω(j) = ωᶜ * tan(π/2 * (1 - j/(Nᵇ+1)))
    ωⱼ = ω.(1:Nᵇ)

    DebyeBosonBath(2, ωᶜ, ωⱼ)
end

function potential(model::DebyeBosonBath, R::AbstractMatrix)
    r = @view R[1,:]
    return sum(model.ωⱼ .^2 .* r .^2 ./ 2)
end

function derivative!(model::DebyeBosonBath, D::AbstractMatrix, R::AbstractMatrix)
    r = @view R[1,:]

    for i in eachindex(r)
        D[1,i] = model.ωⱼ[i]^2 * r[i]
    end
end
