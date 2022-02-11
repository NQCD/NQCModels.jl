
"""
    struct ErpenbeckThoss{T<:AbstractFloat} <: DiabaticFrictionModel

1D two-state diabatic system capable of modelling a molecule adsorbed on a metal surface
or a single-molecule junction.

In the two references, all of the parameters are identical except for the particle mass `m`
and the vertical shift `c` applied to the ϵ₀ state.
Both references modify the shift to ensure the quantum ground-state has an energy of 0 eV.
Note that the mass `m` is specified in atomic mass units (amu) **not** atomic units.
We calculate the offset automatically in the constructor from the Morse potential
zero-point energy.

# References

- PHYSICAL REVIEW B 97, 235452 (2018)
- J. Chem. Phys. 151, 191101 (2019)
"""
struct ErpenbeckThoss{T<:AbstractFloat} <: DiabaticFrictionModel
    Γ::T
    morse::AdiabaticModels.Morse{T}
    D₁::T
    D₂::T
    x₀′::T
    a′::T
    c::T
    V∞::T
    q::T
    ã::T
    x̃::T
    V̄ₖ::T
end

function ErpenbeckThoss(;
    Γ,

    m   = austrip(1.0u"u"),
    Dₑ  = austrip(3.52u"eV"),
    x₀  = austrip(1.78u"Å"),
    a   = austrip(1.7361u"Å^-1"),
    D₁  = austrip(4.52u"eV"),
    D₂  = austrip(0.79u"eV"),
    x₀′ = x₀,
    a′  = austrip(1.379u"Å^-1"),
    V∞  = austrip(-1.5u"eV"),

    q   = 0.05,
    ã   = austrip(0.5u"Å"),
    x̃   = austrip(3.5u"Å"),
    V̄ₖ  = sqrt(Γ/2π)
) 
    morse = AdiabaticModels.Morse(;Dₑ, x₀, a, m)
    c = -NQCModels.AdiabaticModels.eigenenergy(morse, 0) # Set c to offset zero-point energy

    return ErpenbeckThoss(Γ, morse, D₁, D₂, x₀′, a′, c, V∞, q, ã, x̃, V̄ₖ)
end

function NQCModels.potential!(model::ErpenbeckThoss, V::Hermitian, r::Real)
    (;morse, c, D₁, D₂, x₀′, a′, V∞) = model
    ϵ₀(x) = NQCModels.potential(morse, x) + c
    ϵ₁(x) = D₁*exp(-2a′*(x-x₀′)) - D₂*exp(-a′*(x-x₀′)) + V∞

    (;q, ã, x̃, V̄ₖ) = model
    Vₖ(x) = V̄ₖ * ((1-q)/2*(1 - tanh((x-x̃)/ã)) + q)

    V[1,1] = ϵ₀(r)
    V[2,2] = ϵ₁(r)
    V.data[1,2] = Vₖ(r)
    return V
end

function NQCModels.derivative!(model::ErpenbeckThoss, D::Hermitian, r::Real)
    (;morse, D₁, D₂, x₀′, a′) = model
    ∂ϵ₀(x) = NQCModels.derivative(morse, x)
    ∂ϵ₁(x) = -2a′*D₁*exp(-2a′*(x-x₀′)) + a′*D₂*exp(-a′*(x-x₀′))

    (;q, ã, x̃, V̄ₖ) = model
    ∂Vₖ(x) = -V̄ₖ * (1-q)/2 * sech((x-x̃)/ã)^2 / ã

    D[1,1] = ∂ϵ₀(r)
    D[2,2] = ∂ϵ₁(r)
    D.data[1,2] = ∂Vₖ(r)
    return D
end

NQCModels.nstates(::ErpenbeckThoss) = 2
NQCModels.ndofs(::ErpenbeckThoss) = 1
