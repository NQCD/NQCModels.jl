
export Subsystem, CompositeModel
using .FrictionModels
using .AdiabaticModels

"""
Subsystem(M, indices)

A subsystem is a Model which only applies to a subset of the degrees of freedom of the original model. 

**When combined in a CompositeModel**, `potential()`, `derivative!()` and `friction!()` will be sourced from the respective Subsystems. 

Calling `potential()`, `derivative!()`, or `friction!()` on a subsystem directly will output the respective values **for the entire system**. 

The Model specified will be supplied with the positions of the entire system for evaluation. 

"""
struct Subsystem{M,I<:Union{Vector{Int64}, Colon}}
	model::M
	indices::I
end

function Base.show(io::IO, subsystem::Subsystem)
    print(io, "Subsystem:\n\t🏎️ $(subsystem.model)\n\t🔢 $(subsystem.indices)\n")
end

function Subsystem(model, indices=:)
	@debug "Outer Subsystem constructor called"
	# Convert indices to a Vector{Int} or : for consistency
	if isa(indices, Int)
		indices = [indices]
	elseif isa(indices, UnitRange{Int})
		indices = collect(indices)
	end
	Subsystem(model, indices)
end

# Passthrough functions to Model functions
potential(subsystem::Subsystem, R::AbstractMatrix) = potential(subsystem.model, R)
derivative(subsystem::Subsystem, R::AbstractMatrix) = derivative(subsystem.model, R)
derivative!(subsystem::Subsystem, D::AbstractMatrix, R::AbstractMatrix) = derivative!(subsystem.model, D, R)
FrictionModels.friction(subsystem::Subsystem, R::AbstractMatrix) = friction(subsystem.model, R)
FrictionModels.friction!(subsystem::Subsystem, F::AbstractMatrix, R::AbstractMatrix) = friction!(subsystem.model, F, R)
dofs(subsystem::Subsystem) = dofs(subsystem.model)
ndofs(subsystem::Subsystem) = ndofs(subsystem.model)

"""
CompositeModel(Subsystems...)

A CompositeModel is composed of multiple Subsystems, creating an effective model which evaluates each Subsystem for its respective indices. 
"""
struct CompositeModel{S<:Vector{<:Subsystem}, D<:Int} <: AdiabaticModels.AdiabaticModel
	subsystems::S
	dofs::D
end

function Base.show(io::IO, model::CompositeModel)
    print(io, "CompositeModel with subsystems:\n", [system for system in model.subsystems]...)
end

CompositeModel(subsystems::Subsystem...) = CompositeModel(check_models(subsystems...)...) # Check subsystems are a valid combination

get_friction_models(system::Vector{<:Subsystem}) = @view system[findall(x->isa(x.model, FrictionModels.ElectronicFrictionProvider), system)]
get_friction_models(system::CompositeModel) = get_friction_models(system.subsystems)
get_pes_models(system::Vector{<:Subsystem}) = @view system[findall(x->isa(x.model, AdiabaticModels.AdiabaticModel) || isa(x.model, DiabaticModels.DiabaticModel), system)]
get_pes_models(system::CompositeModel) = get_pes_models(system.subsystems)

dofs(system::CompositeModel) = 1:system.dofs
ndofs(system::CompositeModel) = system.dofs

"""
Subsystem combination logic - We only want to allow combination of subsystems:

? 1. with the same number of degrees of freedom

2. without overlapping indices (Build a different type of CompositeModel) to handle these cases separately. 
"""
function check_models(subsystems::Subsystem...)
	systems=vcat(subsystems...)
	# Check unique assignment of potential and derivative to each atom index
	pes_models = get_pes_models(systems)
	pes_model_indices = vcat([subsystem.indices for subsystem in pes_models]...)
	if length(unique(pes_model_indices)) != length(pes_model_indices)
		error("Overlapping indices detected for the assignment of potential energy surfaces.")
	end
	# Check for unique assignment of friction to each atom index
	model_has_friction=systems[findall(x->isa(x.model, FrictionModels.ElectronicFrictionProvider), systems)]
	friction_indices = vcat([subsystem.indices for subsystem in model_has_friction]...)
	if length(unique(friction_indices)) != length(friction_indices)
		error("Overlapping indices detected for the assignment of friction models.")
	end
	
	# Check for different numbers of degrees of freedom in each subsystem
	dofs = [ndofs(subsystem.model) for subsystem in subsystems]
	if length(unique(dofs)) != 1
		error("Subsystems must have the same number of degrees of freedom.")
	end
	return systems, unique(dofs)[1]
end

# Subsystem evaluation of model functions
function potential(system::CompositeModel, R::AbstractMatrix)
	potentials = [potential(subsystem, R[dofs(subsystem.model), subsystem.indices]) for subsystem in get_pes_models(system)]
	return sum(potentials)
end

function derivative!(system::CompositeModel, D::AbstractMatrix, R::AbstractMatrix)
	for subsystem in get_pes_models(system)
		derivative!(subsystem.model, view(D, dofs(subsystem), subsystem.indices), view(R, dofs(subsystem), subsystem.indices))
	end
end

function derivative(system::CompositeModel, R::AbstractMatrix)
	total_derivative=zero(R)
	derivative!(system, total_derivative, R)
	return total_derivative
end

function FrictionModels.friction!(system::CompositeModel, F::AbstractMatrix, R::AbstractMatrix)
	for subsystem in get_friction_models(system)
		eft_indices=vcat([[(j-1)*ndofs(subsystem.model)+i for i in dofs(subsystem.model)] for j in subsystem.indices]...)
		FrictionModels.friction!(subsystem.model, view(F, eft_indices, eft_indices), view(R, dofs(subsystem), subsystem.indices))
	end
end

function FrictionModels.friction(system::CompositeModel, R::AbstractMatrix)
	F=zeros(eltype(R), length(R), length(R))
	FrictionModels.friction!(system, F, R)
	return F
end
	




