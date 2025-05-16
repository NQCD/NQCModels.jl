
struct Scattering1D <: QuantumFrictionModel
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

function NQCModels.potential(model::Scattering1D, R::AbstractMatrix)
    r = R[1]
    V0(x) = model.D*(exp(-2model.a*x) - 2*exp(-model.a*x))
    γ(x) = model.B*exp(-model.a*x^2)

    V₀ = V0(r)
    γᵣ = γ(r)

    V = zeros((model.n_states, model.n_states))

    V[diagind(V,0)] .= model.α + V₀ #metal state energies
    V[diagind(V,1)] .= model.β #metal-metal couplings between neighbouring states
    
    V[end-1, end] = 0.0 #periodic coupling between first and last metal state
    V[2,end] = model.β 

    V[1,1] = V₀ # Occupied molecule state
    V[1,2] = γᵣ # molecule metal coupling

    return V
end

function NQCModels.potential!(model::Scattering1D, V::Hermitian, R::AbstractMatrix)
    r = R[1]
    V0(x) = model.D*(exp(-2model.a*x) - 2*exp(-model.a*x))
    γ(x) = model.B*exp(-model.a*x^2)

    V₀ = V0(r)
    γᵣ = γ(r)

    V.data[diagind(V,0)] .= model.α + V₀
    V.data[diagind(V,1)] .= model.β

    V.data[end-1, end] = 0.0
    V.data[2,end] = model.β

    V.data[1,1] = V₀ # Occupied molecule state
    V.data[1,2] = γᵣ # molecule metal coupling
end

function NQCModels.derivative!(model::Scattering1D, D::Hermitian, R::AbstractMatrix)
    r = R[1]
    D0(x) = 2*model.D*model.a*(exp(-model.a*x)-exp(-2*model.a*x))
    dγ(x) = -2model.a*x*model.B*exp(-model.a*x^2)

    D₀ = D0(r)
    dγᵣ = dγ(r)

    D.data[1,2] = dγᵣ
    D.data[diagind(D,0)] .= D₀
end
