
"""
    Harmonic(m=1.0, ω=1.0, r₀=0.0)

Adiabatic harmonic potential. ``V(x) = mω^2(x-r₀)^2 / 2``

```jldoctest
julia> using Symbolics;

julia> @variables x, m, ω, r₀;

julia> model = Harmonic(m=m, ω=ω, r₀=r₀);

julia> potential(model, hcat(x))
0.5m*(ω^2)*((x - r₀)^2)

julia> derivative(model, hcat(x))
1×1 Matrix{Num}:
 m*(x - r₀)*(ω^2)
```
"""
Parameters.@with_kw struct Harmonic{M,W,R} <: AdiabaticModel
    m::M = 1.0
    ω::W = 1.0
    r₀::R = 0.0
    dofs::Int = 1
end

NQCModels.ndofs(harmonic::Harmonic) = harmonic.dofs

function NQCModels.potential(model::Harmonic, R::AbstractMatrix)
    return sum(0.5 .* model.m .* model.ω.^2 .* (R .- model.r₀) .^2)
end

function NQCModels.derivative!(model::Harmonic, D::AbstractMatrix, R::AbstractMatrix) 
    D .= model.m.* model.ω.^2 .* (R .- model.r₀)
end
