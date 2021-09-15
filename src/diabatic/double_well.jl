
"""
    DoubleWell(mass=1, ω=1, γ=1, Δ=1)

Two state double well, also called the one-dimensional spin-boson model.
See: [J. Chem. Phys. 150, 244102 (2019)](https://doi.org/10.1063/1.5096276)
"""
Parameters.@with_kw struct DoubleWell{M,W,Y,D} <: DiabaticModel
    mass::M = 1
    ω::W = 1
    γ::Y = 1
    Δ::D = 1
end

NonadiabaticModels.ndofs(::DoubleWell) = 1
NonadiabaticModels.nstates(::DoubleWell) = 2

function NonadiabaticModels.potential(model::DoubleWell, R::Real)

    V0(R) = 0.5 * model.mass * model.ω^2 * R^2

    V₀ = V0(R)
    v = sqrt(2)*model.γ*R
    V11 = V₀ + v
    V22 = V₀ - v
    V12 = model.Δ/2

    return Hermitian(SMatrix{2,2,}(V11, V12, V12, V22))
end

function NonadiabaticModels.derivative(model::DoubleWell, R::Real)

    D0(R) = model.mass * model.ω^2 * R

    D₀ = D0(R[1])
    v = sqrt(2)*model.γ
    D11 = D₀ + v
    D22 = D₀ - v
    return Hermitian(SMatrix{2,2}(D11, 0, 0, D22))
end