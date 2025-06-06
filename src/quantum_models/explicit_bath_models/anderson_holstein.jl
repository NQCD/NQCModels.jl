
struct AndersonHolstein{M<:QuantumModel,B,T} <: QuantumModel
    impurity_model::M
    bath_model::B
    fermi_level::T
    nelectrons::Int
end

function AndersonHolstein(impurity_model, bath; fermi_level=0.0)
    fermi_level = austrip(fermi_level)
    nelectrons = count(bath.bathstates .≤ fermi_level)
    return AndersonHolstein(impurity_model, bath, fermi_level, nelectrons)
end

NQCModels.nstates(model::AndersonHolstein) = NQCModels.nstates(model.bath_model) + 1
NQCModels.ndofs(model::AndersonHolstein) = NQCModels.ndofs(model.impurity_model)
NQCModels.nelectrons(model::AndersonHolstein) = model.nelectrons
NQCModels.fermilevel(model::AndersonHolstein) = model.fermi_level

function NQCModels.potential!(model::AndersonHolstein, V::Hermitian, r::AbstractMatrix)
    Vsystem = NQCModels.potential(model.impurity_model, r)
    V[1,1] = Vsystem[2,2] - Vsystem[1,1]
    fillbathstates!(V, model.bath_model)
    fillbathcoupling!(V, Vsystem[2,1], model.bath_model)
    return V
end

function NQCModels.derivative(model::AndersonHolstein, R::AbstractMatrix)
    D = [Hermitian(zero(matrix_template(model, eltype(R)))) for _=1:size(R, 1), _=1:size(R, 2)]
    NQCModels.derivative!(model, D, R)
    return D
end

function NQCModels.derivative!(model::AndersonHolstein, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)
    # Get system model derivative
    D_impurity_model = NQCModels.derivative(model.impurity_model, r)
    # Write and apply bath coupling
    for I in axes(r, 2) # All particles
        D[I][1,1] = D_impurity_model[2,2] - D_impurity_model[1,1]
        fillbathcoupling!(D[I], D_impurity_model[2,1], model.bath_model)
    end
    
    return nothing
end

function NQCModels.state_independent_potential(model::AndersonHolstein, r::AbstractMatrix)
    Vsystem = NQCModels.potential(model.impurity_model, r)
    return Vsystem[1,1]
end

function NQCModels.state_independent_derivative!(model::AndersonHolstein, ∂V::AbstractMatrix, r::AbstractMatrix)
    #Dsystem = get_subsystem_derivative(model, r)
    D_impurity_model = NQCModels.zero_derivative(model.impurity_model, r)
    NQCModels.derivative!(model.impurity_model, D_impurity_model, r)
    @assert ∂V |> length == D_impurity_model |> length
    for I in eachindex(∂V)
        ∂V[I] = D_impurity_model[I][1,1]
    end
end
