using AtomsCalculators
import AtomsBase
using NQCBase
using Unitful, UnitfulAtomic

struct AtomsCalculatorsModel{C} <: ClassicalModel
	calc_object::C
	energy_unit
	length_unit
	ndofs::Int
	atoms::NQCBase.Atoms
	cell::NQCBase.AbstractCell
end

"""
    AtomsCalculatorsModel(calc_object)

Model interface to AtomsCalculators. Supply this function with the calculator object you would use with AtomsCalculators and the correct unit conversions will be automatically applied. 
"""
function AtomsCalculatorsModel(calc_object, structure::AtomsBase.AbstractSystem)
	atoms = NQCBase.Atoms(structure)
	cell = NQCBase.Cell(structure)
	return AtomsCalculatorsModel(
		calc_object,
		AtomsCalculators.energy_unit(calc_object),
		AtomsCalculators.length_unit(calc_object),
		isa(cell, NQCBase.PeriodicCell) ? cell : 3,
		atoms,
		cell,
	)
end

function AtomsCalculatorsModel(calc_object, structure::NQCBase.Structure)
	atoms = structure.atoms
	cell = structure.cell
	return AtomsCalculatorsModel(
		calc_object,
		AtomsCalculators.energy_unit(calc_object),
		AtomsCalculators.length_unit(calc_object),
		isa(cell, NQCBase.PeriodicCell) ? cell : 3,
		atoms,
		cell,
	)
end

NQCModels.ndofs(::AtomsCalculatorsModel) = 3

function NQCModels.potential(model::AtomsCalculatorsModel, R::AbstractMatrix)
	# Convert into system format expected by AtomsBase
	sy = AtomsBase.System(model.atoms, auconvert.(model.length_unit, R), model.cell)
	# Convert AtomsBase calculator energy unit back out. 
	return austrip(potential_energy(sy, model.calc_object) * model.energy_unit)
end

function NQCModels.potential!(model::AtomsCalculatorsModel, V::Matrix{<:Number}, R::AbstractMatrix)
	V .= NQCModels.potential(model, R)
end

function NQCModels.derivative!(model::AtomsCalculatorsModel, D::AbstractMatrix, R::AbstractMatrix)
	# Convert into system format expected by AtomsBase
	sy = NQCBase.System(model.atoms, R, model.cell)
	forces = .- reduce(hcat, AtomsCalculators.forces(sy, model.calc_object)) # Convert to matrix representation rather than Vector{Vector}.
	D .=  austrip.(forces .* model.energy_unit / model.length_unit) # Convert back into atomic units. 
	return D
end


# Minimal AtomsBase Calculator implementation for Classical Models (so we don't have to worry about which state to select. )
AtomsCalculators.energy_unit(ClassicalModel) = energy_unit
AtomsCalculators.length_unit(ClassicalModel) = length_unit
function AtomsCalculators.potential_energy(
	sys::AtomsBase.AbstractSystem,
	model::ClassicalModel,
)
	nqcd_pos = NQCBase.Position(sys)
	return NQCModels.potential(model, nqcd_pos) * energy_unit
end

function AtomsCalculators.forces(
	sys::AtomsBase.AbstractSystem,
	model::ClassicalModel,
)
	nqcd_pos = NQCBase.Position(sys)
	forces_with_unit = -NQCModels.derivative(model, nqcd_pos) .* energy_unit ./ length_unit
	return SVector{3}.(eachcol(forces_with_unit))
end

function AtomsCalculators.virial(
	sys::AtomsBase.AbstractSystem,
	model::ClassicalModel,
)
	nd::Int = NQCModels.ndofs(model)
	return zeros(SMatrix{nd,nd})
end

