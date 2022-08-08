"""
    MiaoSubotnik{T<:AbstractFloat} <: DiabaticModel

Double well model with parameters matching those of Miao and Subotnik in the reference.
This model should be paired with the `AndersonHolstein` model to couple to the bath of metallic states.

# References

- J. Chem. Phys. 150, 041711 (2019)
"""
struct MiaoSubotnik{T<:AbstractFloat} <: DiabaticModel
    Γ::T
    m::T
    ω::T
    g::T
    ΔG::T
    coupling::T
end

function MiaoSubotnik(;
    Γ,
    m = 2e3,
    ω = 2e-4,
    g = 20.6097,
    ΔG = -3.8e-3,
    coupling = sqrt(Γ/2π)
    )
    return MiaoSubotnik(Γ, m, ω, g, ΔG, coupling)
end

NQCModels.ndofs(::MiaoSubotnik) = 1
NQCModels.nstates(::MiaoSubotnik) = 2

function NQCModels.potential(model::MiaoSubotnik, x::Real)
    (;m, ω, g, ΔG, coupling) = model

    V11 = (m*ω^2*x^2)/2
    V22 = (m*ω^2*(x-g)^2)/2 + ΔG
    V12 = coupling

    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NQCModels.derivative(model::MiaoSubotnik, x::Real)
    (;m, ω, g) = model

    D11 = m*ω^2*x
    D22 = m*ω^2*(x-g)
    
    return Hermitian(SMatrix{2,2}(D11, 0, 0, D22))
end
