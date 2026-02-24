using AtomsCalculators
import AtomsBase
using NQCBase
using Unitful, UnitfulAtomic
using StaticArrays

# NQCD uses these units
const energy_unit = u"hartree"
const length_unit = u"a0_au"

struct AtomsCalculatorsModel{C} <: ClassicalModel
	calc_object::C
	energy_unit
	length_unit
	ndofs::Int
	atoms::NQCBase.Atoms
	cell::NQCBase.AbstractCell
end

"""
    AtomsCalculatorsModel(calc_object, structure)

Wrap an [AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl)-compatible
calculator as an NQCModels `ClassicalModel`.

`calc_object` is any object implementing the AtomsCalculators interface
(`potential_energy`, `forces`, `energy_unit`, `length_unit`).
`structure` can be either an `AtomsBase.AbstractSystem` or an `NQCBase.Structure`,
and is used to extract atomic species and cell information.

Unit conversions between the calculator's units and NQCModels' internal atomic units
(hartree for energy, bohr for length) are applied automatically.

# Example

```julia
using NQCModels, NQCBase

atoms = Atoms([:H, :H])
structure = NQCBase.Structure(atoms, rand(3, 2), InfiniteCell())
model = AtomsCalculatorsModel(Harmonic(), structure)
potential(model, rand(3, 2))
```
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
	sy = NQCBase.System(model.atoms, R, model.cell)
	# Convert AtomsBase calculator energy unit back out. 
	return austrip(AtomsCalculators.potential_energy(sy, model.calc_object) * model.energy_unit)
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

