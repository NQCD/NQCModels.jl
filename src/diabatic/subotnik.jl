

export Subotnik_A
export MiaoSubotnik

@doc raw"""
    Model A according to Coffmann & Subotnik, Phys.Chem.Chem.Phys., 20, 9847 (2018).
    V_0 =  0.5*m*ω^2 r^2
    V_1 =  0.5*m*ω^2 (r-g)^2 + ΔG
    Molecule-metal coupling:
    Γ(x) = Γ_1(Γ_0 (K(r-d)^2)/(1+K(r-d)^2))
    d is the minimum of the PES, d = g/2 +ΔG/(mω^2g) 
    In the original model, K = 0.1 and Γ_0=0.01; Γ_1 is used as an energy unit
    The numbers given as default correspond to the one listed in the paper.
"""
@with_kw struct Subotnik_A <: DiabaticModel
    n_states::UInt8 = 2
    m::Float64 = 2000
    om::Float64 = 2e-4
    g::Float64 = 20.6097
    DG::Float64 = -0.0038
    G1::Float64 = 1e-4
    G0::Float64 = 1e-2
    K:: Float64 = 0.1
    d::Float64 = 8
end

"""
Diabatic potenial matrix
"""
function potential!(model::Subotnik_A, V::Hermitian, R::AbstractMatrix)
    # Update d
    d = model.g/2 + model.DG/(model.m*model.om^2*model.g)


    # Higher potential well
    V_0(r) = (model.m*model.om^2*r^2)/2.0
    # Loweer energy potential well
    V_1(r) = (model.m*model.om^2*(r-model.g)^2)/2.0 + model.DG
    #println(R[1]," " , V_0(R[1]), " ", V_1(R[1]))
    # weighted metal-molecule coupling between the diabates
    G(r)=model.G1*(model.G0 + (model.K*(r - model.d)^2)/(1.0 + model.K*(r -model.d)^2))
    
    
    #V = Hermitian(zeros(2,2))
    # Set up the Hamiltonian
    V[1,1] = V_0(R[1])
    V[2,2] = V_1(R[1])
    V.data[1,2] = G(R[1])
    V.data[2,1] = G(R[1])
end

"""
Elementwise position derivative of the above potential
"""
function derivative!(model::Subotnik_A, derivative::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    
    q = R[1]
    derivative[1][1,1] = model.m*model.om^2*q
    derivative[1][2,2] =model.m*model.om^2*(q-model.g)
    derivative[1].data[1,2] = model.G1*((2*model.K*(q-model.d)*(1+model.K*(q-model.d)^2)-2*model.K^2*(q-model.d)^2)/
    (1+model.K*(q-model.d)^2)^2)

    derivative[1].data[2,1] = model.G1*((2*model.K*(q-model.d)*(1+model.K*(q-model.d)^2)-2*model.K^2*(q-model.d)^2)/
    (1+model.K*(q-model.d)^2)^2)
    
end



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
@with_kw struct MiaoSubotnik <: SparseDiabaticModel
    m::Float64 = 2000
    om::Float64 = 2e-4
    g::Float64 = 20.6097
    DG::Float64 = -3.8e-3
    Gamma::Float64 = 6.4e-3
    W::Float64 = 10*Gamma
    n_states::UInt64 = 40
end

function potential!(model::MiaoSubotnik, V::Hermitian, R::AbstractMatrix)
    # Note that for the purpose of this model, n_states includes V1 and V0
    #pi=3.14159265359
    n_states=model.n_states
    # Higher potential well
    V_0(r) = (model.m*model.om^2*r^2)/2.0
    # Loweer energy potential well
    V_1(r) = (model.m*model.om^2*(r-model.g)^2)/2.0 + model.DG
    #println(R[1]," " , V_0(R[1]), " ", V_1(R[1]))
    # weighted metal-molecule coupling between the diabates
    V_couple=sqrt(model.Gamma*2*model.W/(2*pi*n_states))
    
    # Set up the Hamiltonian
    V[1,1] = V_1(R[1])
    spacing = 2*model.W/(n_states-2)
    # The Fermi Level is assumed to be zero
    # Since half of the state will be filled with electrons, this puts the 
    # onset of the states at half the bandwidth.
    V[2,2] = -model.W + V_0(R[1])
    V.data[1,2] = V_couple
    for i=3:n_states
        V[i,i] = V[i-1,i-1] + spacing
        V.data[1,i] = V_couple
    end
    return V
end

function derivative!(model::MiaoSubotnik, derivative::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    
    q = R[1]
    derivative[1][1,1] = model.m*model.om^2*(q-model.g)
    derivative[1][2,2] = model.m*model.om^2*q
    for i=3:model.n_states
        derivative[1][i,i] = derivative[1][2,2]
    end

    return derivative
end
