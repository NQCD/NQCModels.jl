using NQCBase: au_to_ang, eV_to_au, eV_per_ang_to_au, System
using UnitfulAtomic
using .AtomsCalculators: AtomsCalculators

export ACEpotentialsModel

"""
    struct ACEpotentialsModel{A,C,M} <: AdiabaticModel

Model for interfacing with ACEpotentials.jl potentials.
"""
struct ACEpotentialsModel{A,C,M} <: AdiabaticModel
    atoms::A
    cell::C
    model::M
end

NQCModels.ndofs(::ACEpotentialsModel) = 3

function NQCModels.potential(model::ACEpotentialsModel, R::AbstractMatrix)
    system = System(model.atoms, R, model.cell)
    return austrip(auconvert(AtomsCalculators.potential_energy(system, model.model)))
end

function NQCModels.derivative!(model::ACEpotentialsModel, D::AbstractMatrix, R::AbstractMatrix)
    system = System(model.atoms, R, model.cell)
    force = AtomsCalculators.forces(system, model.model)
    D .= -austrip.(auconvert.(reduce(vcat,transpose.(force))))
    return D
end
