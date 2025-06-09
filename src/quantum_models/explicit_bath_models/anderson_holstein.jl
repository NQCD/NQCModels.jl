
struct AndersonHolstein{M<:QuantumModel,Dᵢ,B,T} <: QuantumModel
    impurity_model::M
    impurity_derivative::Dᵢ
    bath_model::B
    fermi_level::T
    nelectrons::Int
end

# include if statement here so that impurity_derivative always has shape Matrix{<:Hermitian} 
# that way it can be populated in derivative!(AndersonHolstien) and the shape is certain
function AndersonHolstein(impurity_model, bath; fermi_level=0.0) 
    impurity_derivative = NQCModels.zero_derivative(impurity_model, hcat(0.0))
#=     if shape(impurity_derivative) <: AbstractMatrix{<:Hermitian}
    elseif shape(impurity_derivative)
    end =#
    fermi_level = austrip(fermi_level)
    nelectrons = count(bath.bathstates .≤ fermi_level)
    return AndersonHolstein(impurity_model, impurity_derivative, bath, fermi_level, nelectrons)
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

#= These multiple dispatch versions exits because the derivative of the 
impurity model may either be of type Hermitian or Matrix{Hermitian} 
depending on the number of spatial dimensions the impurity is defined over =#

"""
    function NQCModels.derivative!(model::AndersonHolstein{M, AbstractMatrix{<:Hermitian}, B, T}, D::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
        output: nothing

Updates the derivative of the Anderson-Holstein Hamiltonian with respect to all spatial degrees of freedom to have the correct values
for a given position R.
    
This function is multiple dispatched over the shape of derivative(model.impurity_model) as these impurities may be defined over different
numbers of spatial degrees of freedom.
"""
function NQCModels.derivative!(model::AndersonHolstein, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix)
    # Get impurity model derivative
    D_impurity_model = NQCModels.zero_derivative(model.impurity_model, r)
    NQCModels.derivative!(model.impurity_model, D_impurity_model, r)

    # Write and apply bath coupling
    for i in axes(r, 1) #All model degrees of freedom
        for j in axes(r, 2) # All particles
            D[i,j][1,1] = D_impurity_model[i,j][2,2] - D_impurity_model[i,j][1,1]
            fillbathcoupling!(D[i,j], D_impurity_model[i,j][2,1], model.bath_model)
        end
    end
    
    return nothing
end

#= """
    function NQCModels.derivative!(model::AndersonHolstein, D::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
        output: nothing

Updates the derivative of the Anderson-Holstein Hamiltonian with respect to all spatial degrees of freedom to have the correct values
for a given position R.
    
This function is multiple dispatched over the shape of derivative(model.impurity_model) as these impurities may be defined over different
numbers of spatial degrees of freedom.
"""
function NQCModels.derivative!(model::AndersonHolstein{M, Hermitian, B, T}, D::AbstractMatrix{<:Hermitian}, r::AbstractMatrix) where {M,B,T}
    # Get impurity model derivative
    D_impurity_model = derivative(model.impurity_model, r)

    # Write and apply bath coupling
    for I in axes(r, 2) # All particles
        D[I][1,1] = D_impurity_model[2,2] - D_impurity_model[1,1]
        fillbathcoupling!(D[I], D_impurity_model[2,1], model.bath_model)
    end
    
    return nothing
end =#

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

#= function get_subsystem_derivative(model::AndersonHolstein, r::AbstractMatrix)
    if size(r) != size(model.tmp_derivative[])
        model.tmp_derivative[] = NQCModels.zero_derivative(model.model, r)
    end
    NQCModels.derivative!(model.model, model.tmp_derivative[], r)
    return model.tmp_derivative[]
end =#