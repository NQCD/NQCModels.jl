
struct Scattering1D <: DiabaticFrictionModel
    n_states::UInt
    N::UInt
    a::Float64
    D::Float64
    α::Float64
    β::Float64
    B::Float64
    σ::Float64
end

NQCModels.ndofs(::Scattering1D) = 1
NQCModels.nstates(model::Scattering1D) = model.n_states

function Scattering1D(;N=10, a=1, D=1, α=0, β=1, B=1, σ=0.6u"eV")
    Scattering1D(N+1, N, a, D, α, β, B, austrip(σ))
end

function NQCModels.potential!(model::Scattering1D, V::Hermitian, R::AbstractMatrix)
    r = R[1]
    V0(r) = model.D*(exp(-2model.a*r) - 2*exp(-model.a*r))
    γ(r) = model.B*exp(-model.a*r^2)

    V₀ = V0(r)
    γᵣ = γ(r)

    V[1,1] = V₀ # Occupied molecule state
    V.data[1,2] = γᵣ # molecule metal coupling
    for i=2:model.N
        V[i,i] = model.α + V₀
        V.data[i,i+1] = model.β
    end
    V.data[2,end] = model.β
    V[end,end] = model.α + V₀
end

function NQCModels.derivative!(model::Scattering1D, D::Hermitian, R::AbstractMatrix)
    r = R[1]
    D0(r) = 2*model.D*model.a*(exp(-model.a*r)-exp(-2*model.a*r))
    dγ(r) = -2model.a*r*model.B*exp(-model.a*r^2)

    D₀ = D0(r)
    dγᵣ = dγ(r)

    D.data[1,2] = dγᵣ
    for i=1:model.N+1
        D[i,i] = D₀
    end
end
