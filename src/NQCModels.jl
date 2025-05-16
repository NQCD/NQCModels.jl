"""
NQCModels define the potentials and derivatives that
govern the dynamics of the particles.
These can exist as analytic models or as interfaces to other codes. 
"""
module NQCModels

using Reexport: @reexport

export potential
export derivative
export nstates
export ndofs

"""
Top-level type for models.

# Implementation
When adding new models, this should not be directly subtyped. Instead, depending on
the intended functionality of the model, one of the child abstract types should be
subtyped.
If an appropriate type is not already available, a new abstract subtype should be created. 
"""
abstract type Model end

Base.broadcastable(model::Model) = Ref(model)

"""
    potential(model::Model, R::AbstractMatrix)

Evaluate the potential at position `R` for the given `model`.
"""
function potential(model::Model, R::AbstractMatrix)
    if ndofs(model) == 1
        if size(R, 2) == 1
            return potential(model, R[1])
        else
            return potential(model, @view R[1,:])
        end
    elseif size(R, 2) == 1
        return potential(model, @view R[:,1])
    else
        throw(MethodError(potential, (model, R)))
    end
end

"""
    potential!(model::Model, V, R::AbstractMatrix)

In-place version of `potential`, used to implement more efficient dynamics.

Currently used only for all `QuantumModels`, see `quantum_models/QuantumModels.jl`.
As `ClassicalModels` store their potential as a Real, it cannot be updated in-place so
potential!(model::ClassicalModel, V, R::AbstractMatrix) returns a new value. This function is 
defined on these models to enable efficient multiple dispatching within NQCDynamics.  
"""
function potential!(model::Model, V, R::AbstractMatrix)
    if ndofs(model) == 1
        if size(R, 2) == 1
            return potential!(model, V, R[1])
        else
            return potential!(model, V, view(R, 1, :))
        end
    elseif size(R, 2) == 1
        return potential!(model, V, view(R, :, 1))
    else
        throw(MethodError(potential!, (model, V, R)))
    end

end

"""
    derivative!(model::Model, D, R::AbstractMatrix)

Fill `D` with the derivative of the electronic potential as a function of the positions `R`.

This must be implemented for all models.
"""
function derivative!(model::Model, D, R::AbstractMatrix)
    if ndofs(model) == 1
        if size(R, 2) == 1
            derivative!(model, D, R::AbstractMatrix)
        else
            derivative!(model, view(D, 1, :), view(R, 1, :))
        end
    elseif size(R, 2) == 1
        derivative!(model, view(D, :, 1), view(R, :, 1))
    else
        throw(MethodError(derivative!, (model, D, R)))
    end
end

"""
    derivative(model::Model, R)

Allocating version of `derivative!`, this definition should be suitable for all models.

Implement `zero_derivative` to allocate an appropriate array then implement `derivative!`
to fill the array.
"""
function derivative(model::Model, R)
    D = zero_derivative(model, R)
    derivative!(model, D, R)
    return D
end

"""
    zero_derivative(model::Model, R)

Create an zeroed array of the right size to match the derivative.
"""
function zero_derivative(model::Model, R) end

"""
    nstates(::Model)

Get the number of electronic states in the model.
"""
nstates(::Model) = error("This should return the total number of electronic states.")

"""
    ndofs(::Model)

Get the number of degrees of freedom for every atom in the model. Usually 1 or 3.
"""
ndofs(::Model) = error("This should return the number of degrees of freedom for each atom.")

dofs(model::Model) = Base.OneTo(ndofs(model))

state_independent_potential(model, r) = 0.0

function state_independent_potential!(model, Vsystem, r) 
    Vsystem .= 0.0
end

function state_independent_derivative(model, r) end
state_independent_derivative!(model, derivative, r) = fill!(derivative, zero(eltype(r)))
nelectrons(::Model) = error("This should return the total number of electrons.")
fermilevel(::Model) = 0.0

eachelectron(model::Model) = Base.OneTo(nelectrons(model))
eachstate(model::Model) = Base.OneTo(nstates(model))

mobileatoms(::Model, n::Int) = Base.OneTo(n)

include("bath_discretisations/BathDiscretisations.jl")
@reexport using .BathDiscretisations

include("classical_models/ClassicalModels.jl")
@reexport using .ClassicalModels

include("quantum_models/QuantumModels.jl")
@reexport using .QuantumModels

include("friction_models/FrictionModels.jl")
@reexport using .FrictionModels

include("plot.jl")

include("subsystems.jl")

end # module
