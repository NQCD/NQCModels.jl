

# export Subotnik_A

# @doc raw"""
#     Model A according to Coffmann & Subotnik, Phys.Chem.Chem.Phys., 20, 9847 (2018).
#     V_0 =  0.5*m*ω^2 r^2
#     V_1 =  0.5*m*ω^2 (r-g)^2 + ΔG
#     Molecule-metal coupling:
#     Γ(x) = Γ_1(Γ_0 (K(r-d)^2)/(1+K(r-d)^2))
#     d is the minimum of the PES, d = g/2 +ΔG/(mω^2g) 
#     In the original model, K = 0.1 and Γ_0=0.01; Γ_1 is used as an energy unit
#     The numbers given as default correspond to the one listed in the paper.
# """
# @with_kw struct Subotnik_A <: DiabaticModel
#     n_states::UInt8 = 2
#     m::Float64 = 2000
#     om::Float64 = 2e-4
#     g::Float64 = 20.6097
#     DG::Float64 = -0.0038
#     G1::Float64 = 1e-4
#     G0::Float64 = 1e-2
#     K:: Float64 = 0.1
#     d::Float64 = 8
# end

# """
# Diabatic potenial matrix
# """
# function potential!(model::Subotnik_A, V::Hermitian, R::AbstractMatrix)
#     # Update d
#     d = model.g/2 + model.DG/(model.m*model.om^2*model.g)


#     # Higher potential well
#     V_0(r) = (model.m*model.om^2*r^2)/2.0
#     # Loweer energy potential well
#     V_1(r) = (model.m*model.om^2*(r-model.g)^2)/2.0 + model.DG
#     #println(R[1]," " , V_0(R[1]), " ", V_1(R[1]))
#     # weighted metal-molecule coupling between the diabates
#     G(r)=model.G1*(model.G0 + (model.K*(r - model.d)^2)/(1.0 + model.K*(r -model.d)^2))
    
    
#     #V = Hermitian(zeros(2,2))
#     # Set up the Hamiltonian
#     V[1,1] = V_0(R[1])
#     V[2,2] = V_1(R[1])
#     V.data[1,2] = G(R[1])
#     V.data[2,1] = G(R[1])
# end

# """
# Elementwise position derivative of the above potential
# """
# function derivative!(model::Subotnik_A, derivative::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    
#     q = R[1]
#     derivative[1][1,1] = model.m*model.om^2*q
#     derivative[1][2,2] =model.m*model.om^2*(q-model.g)
#     derivative[1].data[1,2] = model.G1*((2*model.K*(q-model.d)*(1+model.K*(q-model.d)^2)-2*model.K^2*(q-model.d)^2)/
#     (1+model.K*(q-model.d)^2)^2)

#     derivative[1].data[2,1] = model.G1*((2*model.K*(q-model.d)*(1+model.K*(q-model.d)^2)-2*model.K^2*(q-model.d)^2)/
#     (1+model.K*(q-model.d)^2)^2)
    
# end



@doc raw"""
    Model MiaoSubotnik according to Miao & Subotnik, J. Chem Phys., 150, 041711 (2019).
    V_0 =  0.5*m*ω^2 x^2
    V_1 =  0.5*m*ω^2 (x-g)^2 + ΔG
    Molecule-metal coupling:
    V = sqrt(\Gamma*2*W/(2*\pi*M))
    d is the minimum of the PES, d = g/2 +ΔG/(mω^2g) 
    2W is the bandwidth
    In the original model, m = 2000, ω = 0.0002, g = 20.6097, ΔG = -0.0038 and 
    kBT=0.00095. In the paper, Γ, 2W and M were varied and tested.
    The numbers given as default correspond to the one listed in the paper.
    The model is defined such that h=(V_0-V_1)d^t d + V_1, however, the diabatic surfaces
    that they show in the paper in Fig. 1 correspond to the picture implemented below
"""
Parameters.@with_kw struct MiaoSubotnik <: LargeDiabaticModel
    m::Float64 = 2000
    ω::Float64 = 2e-4
    g::Float64 = 20.6097
    ΔG::Float64 = -3.8e-3
    Γ::Float64 = 6.4e-3
    W::Float64 = 5Γ
    M::Int = 40
    ρ::Float64 = M/2W
    increment::Float64 = 1/ρ
    n_states::Int = M+1
end

NQCModels.ndofs(::MiaoSubotnik) = 1
NQCModels.nstates(model::MiaoSubotnik) = model.n_states

function NQCModels.potential!(model::MiaoSubotnik, V::Hermitian, R::Real)
    Parameters.@unpack m, ω, g, ΔG, Γ, ρ, increment, n_states = model

    U0(x) = (m*ω^2*x^2)/2
    U1(x) = (m*ω^2*(x-g)^2)/2 + ΔG
    
    V[1,1] = U1(R) - U0(R)

    Vₙ = sqrt(Γ/(2π*ρ))
    V.data[1,2:end] .= Vₙ

    V[2,2] = -model.W
    for i=3:n_states
        V[i,i] = V[i-1,i-1] + increment
    end

    return V
end

function NQCModels.derivative!(model::MiaoSubotnik, derivative::Hermitian, R::Real)
    Parameters.@unpack m, ω, g, n_states = model

    dU0(x) = m*ω^2*x
    dU1(x) = m*ω^2*(x-g)
    
    derivative[1,1] = dU1(R) - dU0(R)

    return derivative
end

function NQCModels.state_independent_potential(model::MiaoSubotnik, r::AbstractMatrix)
    (;m, ω) = model
    return 1/2 * m * ω^2 * r[1]^2
end

function NQCModels.state_independent_derivative!(model::MiaoSubotnik, derivative::AbstractMatrix, r::AbstractMatrix)
    (;m, ω) = model
    derivative[1] = m * ω^2 * r[1]
    return derivative
end
