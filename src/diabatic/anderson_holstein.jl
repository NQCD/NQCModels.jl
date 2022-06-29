
struct AndersonHolstein{M<:DiabaticModel,B} <: LargeDiabaticModel
    model::M
    bath::B
end

NQCModels.nstates(model::AndersonHolstein) = NQCModels.nstates(model.bath) + 1
NQCModels.ndofs(model::AndersonHolstein) = NQCModels.ndofs(model.model)
NQCModels.nelectrons(model::AndersonHolstein) = fld(NQCModels.nstates(model.bath), 2)
NQCModels.fermilevel(::AndersonHolstein) = 0

function NQCModels.potential!(model::AndersonHolstein, V::Hermitian, r::AbstractMatrix)
    Vsystem = NQCModels.potential(model.model, r)
    V[1,1] = Vsystem[2,2] - Vsystem[1,1]
    fillbathstates!(V, model.bath)
    fillbathcoupling!(V, Vsystem[2,1], model.bath)
    return V
end

function NQCModels.derivative!(model::AndersonHolstein, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)

    Dsystem = NQCModels.derivative(model.model, r)
    
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
    Dsystem = NQCModels.derivative(model.model, r)
    for I in eachindex(∂V, Dsystem)
        ∂V[I] = Dsystem[I][1,1]
    end
    return ∂V
end
