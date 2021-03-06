using LinearAlgebra: diagind

"""
    OuyangModelOne(A=0.01, B=1.6, Γ=1e-4, N=10, ΔE=1.6e-2, D=1.0)

Model #1 from [Ouyang and Subotnik](http://aip.scitation.org/doi/10.1063/1.4908032).
See also [Ouyang's thesis](https://repository.upenn.edu/edissertations/2508/).
"""
Parameters.@with_kw struct OuyangModelOne <: DiabaticFrictionModel
    A::Float64 = 0.01
    B::Float64 = 1.6
    Γ::Float64 = 1e-4
    N::Int = 10
    ΔE::Float64 = 1.6e-2
    ρ::Float64 = N / ΔE
    C::Float64 = sqrt(Γ / 2π*ρ) / N
    D::Float64 = 1.0

    n_states::Int = N + 1
end

NQCModels.nstates(model::OuyangModelOne) = model.n_states
NQCModels.ndofs(::OuyangModelOne) = 1

function NQCModels.potential!(model::OuyangModelOne, V::Hermitian, r::Real)
    Parameters.@unpack A, B, C, D, ΔE, N = model

    Vsys(x) = A*tanh(B*x)
    Vbath(x, Δ) = -Vsys(x) + Δ
    Vsb(x) = C*exp(-D*x^2)

    V[1,1] = Vsys(r)
    Δs = range(-ΔE/2, ΔE/2, length=N)
    V.data[1,2:N+1] .= Vsb(r)
    V[diagind(N+1,N+1)[2:end]] .= Vbath.(r, Δs)

    return V
end

function NQCModels.derivative!(model::OuyangModelOne, derivative::Hermitian, r::Real)
    Parameters.@unpack A, B, C, D, N = model

    Dsys(x) = B*A*sech(B*x)^2
    Dbath(x) = -Dsys(x)
    Dsb(x) = -2D*x*C*exp(-D*x^2)

    derivative[1,1] = Dsys(r)
    derivative.data[1,2:N+1] .= Dsb(r)
    derivative[diagind(N+1,N+1)[2:end]] .= Dbath(r)

    return derivative
end
