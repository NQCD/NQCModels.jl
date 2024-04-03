"""

This module implements the Anderson-Holstein model, which is a diabatic model with bandwidth.

Referencnes:
http://fy.chalmers.se/~hellsing/Many_Body-Physics/Newns-Anderson_080210.pdf 

https://link.springer.com/book/10.1007/978-1-4757-5714-9 section 4.2 (Anderson-Holstein model Hamiltonian)


"""



struct AndersonHolstein{M<:DiabaticModel,B,D,T} <: LargeDiabaticModel
    model::M
    bath::B
    tmp_derivative::Base.RefValue{D} # temporary array for storing the derivative of the subsystem model 
    fermi_level::T
    nelectrons::Int
end

function AndersonHolstein(model, bath; fermi_level=0.0)
    tmp_derivative = Ref(NQCModels.zero_derivative(model, zeros(1,1))) # size of tmp_derivative is 1x1 through Ref() reference can be accessed through []
    fermi_level = austrip(fermi_level)
    # x -> x <= fermi_level :  anonymous function that checks whether its argument x is less than or equal to fermi_level
    # It counts how many states are below or equal to the fermi level
    nelectrons = count(x -> x <= fermi_level, bath.bathstates)
    return AndersonHolstein(model, bath, tmp_derivative, fermi_level, nelectrons)
end

NQCModels.nstates(model::AndersonHolstein) = NQCModels.nstates(model.bath) + 1
NQCModels.ndofs(model::AndersonHolstein) = NQCModels.ndofs(model.model)
NQCModels.nelectrons(model::AndersonHolstein) = model.nelectrons
NQCModels.fermilevel(model::AndersonHolstein) = model.fermi_level

function NQCModels.potential!(model::AndersonHolstein, V::Hermitian, r::AbstractMatrix)
    Vsystem = NQCModels.potential(model.model, r) # potential matrix evaluated at r i.e. ErpenbeckThoss is symmetic 2x2 matrix
    V[1,1] = Vsystem[2,2] - Vsystem[1,1] # h(r) = U_1(r) - U_0(r)
    fillbathstates!(V, model.bath) # fill the diagonal elements [2,2]...[N,N] with bath energies (ϵ_k)
    fillbathcoupling!(V, Vsystem[2,1], model.bath) # Vsystem[2,1] Hybridization function component V_k(r)  
    #Reference for V https://louhokseson.github.io/SVG/NAH_Matrix.svg
    return V
end

function NQCModels.derivative!(model::AndersonHolstein, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)
    Dsystem = get_subsystem_derivative(model, r)
    
    for I in eachindex(Dsystem, D)
        D[I][1,1] = Dsystem[I][2,2] - Dsystem[I][1,1]
        fillbathcoupling!(D[I], Dsystem[I][2,1], model.bath)
    end

    return D
end

function NQCModels.state_independent_potential(model::AndersonHolstein, r::AbstractMatrix)
    Vsystem = NQCModels.potential(model.model, r)
    return Vsystem[1,1]
end

function NQCModels.state_independent_derivative!(model::AndersonHolstein, ∂V::AbstractMatrix, r::AbstractMatrix)
    Dsystem = get_subsystem_derivative(model, r)
    for I in eachindex(∂V, Dsystem)
        ∂V[I] = Dsystem[I][1,1]
    end
    return ∂V
end

function get_subsystem_derivative(model::AndersonHolstein, r::AbstractMatrix)
    # This function is to calculate the subsystem model's derivative i.e. ErpenbeckThoss model
    # i.e. model.model: ErpenbeckThoss model
    if size(r) != size(model.tmp_derivative[]) # model.tmp_derivative[] is a reference to the temporary derivative array
        model.tmp_derivative[] = NQCModels.zero_derivative(model.model, r)
    end
    NQCModels.derivative!(model.model, model.tmp_derivative[], r) # calculate the derivative of the subsystem model i.e. ErpenbeckThoss model
    return model.tmp_derivative[]
end
